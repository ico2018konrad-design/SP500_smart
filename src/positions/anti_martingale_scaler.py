"""Anti-Martingale Scaler — scale-in logic.

NEW POSITION ONLY when previous position is +1% in profit.
NEVER average down into losing positions.

This is the critical difference from martingale/averaging-down:
- Martingale: add more when losing → exponential risk
- Anti-martingale: add more when winning → risk reward maximized
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from src.positions.position_manager import Position, PositionManager, PositionStatus
from src.signals.signal_types import Signal, SignalDirection

logger = logging.getLogger(__name__)


class AntiMartingaleScaler:
    """Manages scale-in logic for position pyramiding.

    Rules:
    1. First position opens on valid signal
    2. Each additional position requires previous to be +threshold% in profit
    3. NEVER add to losing position
    4. Min 1-2 hours between entries
    5. Max 6 positions total
    6. When adding: move previous stop to break-even (at minimum)
    """

    def __init__(
        self,
        profit_threshold: float = 0.01,     # +1% before scaling
        min_hours_between: float = 2.0,      # min time between entries
        max_positions: int = 6,
        position_size_pct: float = 0.1667,   # 16.67% per position (1/6)
        backtest_mode: bool = False,         # True: bypass real-time clock checks
        allocated_capital: Optional[float] = None,  # fixed base capital for equal-dollar sizing
    ):
        self.profit_threshold = profit_threshold
        self.min_hours_between = min_hours_between
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.backtest_mode = backtest_mode
        self._allocated_capital = allocated_capital
        self._last_entry_time: Optional[datetime] = None

    def set_allocated_capital(self, capital: float) -> None:
        """Set the fixed allocated capital for equal-dollar position sizing."""
        self._allocated_capital = capital
        logger.info("Scaler: allocated capital set to %.2f", capital)

    def can_scale_in(
        self,
        position_manager: PositionManager,
        symbol: str,
        direction: str,
        backtest_time: Optional[datetime] = None,
    ) -> tuple:
        """Check if scaling in is allowed.

        Args:
            position_manager: Current position manager state
            symbol: Trading symbol
            direction: LONG or SHORT
            backtest_time: Bar datetime for backtest mode (bypasses datetime.now())

        Returns (can_scale, reason)
        """
        open_positions = position_manager.get_open_positions()

        # Check max positions (use the stricter of scaler limit and position manager limit)
        effective_max = min(self.max_positions, position_manager.max_positions)
        if len(open_positions) >= effective_max:
            return False, f"At max positions ({effective_max})"

        # Check time since last entry
        if self._last_entry_time is not None:
            now = backtest_time if (self.backtest_mode and backtest_time is not None) else datetime.now()
            elapsed = now - self._last_entry_time
            min_elapsed = timedelta(hours=self.min_hours_between)
            if elapsed < min_elapsed:
                remaining = (min_elapsed - elapsed).total_seconds() / 3600
                return False, f"Too soon: {remaining:.1f}h remaining before next entry"

        # If no positions yet, allow first entry
        if len(open_positions) == 0:
            return True, "First position"

        # Check same direction positions for same symbol
        same_dir = [
            p for p in open_positions
            if p.symbol == symbol and p.direction == direction
        ]

        if len(same_dir) == 0:
            return True, "No existing position in this direction"

        # CRITICAL: Check that the most recent position is in profit
        # Sort by entry time, check the most recent
        latest = sorted(same_dir, key=lambda p: p.entry_time)[-1]

        if latest.unrealized_pnl_pct < self.profit_threshold:
            return False, (
                f"Previous position not at +{self.profit_threshold:.1%}: "
                f"current={latest.unrealized_pnl_pct:.2%} "
                f"(ANTI-MARTINGALE: never average down)"
            )

        # Check for any losing positions — never add when any position is losing
        losing = [p for p in same_dir if p.unrealized_pnl_pct < 0]
        if losing:
            return False, f"{len(losing)} losing position(s) — anti-martingale rule"

        return True, "Scale-in allowed (previous +{:.2%})".format(latest.unrealized_pnl_pct)

    def execute_scale_in(
        self,
        position_manager: PositionManager,
        signal: Signal,
        capital: float,
        current_prices: Dict[str, float],
        atr: float = 0.0,
        backtest_time: Optional[datetime] = None,
        allocated_capital: Optional[float] = None,
    ) -> Optional[Position]:
        """Execute scale-in if conditions are met.

        Args:
            position_manager: Current position manager state
            signal: Entry signal
            capital: Available allocated capital
            current_prices: Current prices for all symbols
            atr: Current ATR for trailing stop calculation
            backtest_time: Bar datetime for backtest mode

        Returns:
            New Position if opened, None otherwise
        """
        direction = signal.direction.value
        symbol = signal.symbol

        can_scale, reason = self.can_scale_in(position_manager, symbol, direction, backtest_time=backtest_time)

        if not can_scale:
            logger.debug("Scale-in rejected for %s %s: %s", direction, symbol, reason)
            return None

        # Determine scale number (1-6)
        existing = [
            p for p in position_manager.get_open_positions()
            if p.symbol == symbol and p.direction == direction
        ]
        scale_number = len(existing) + 1

        # Calculate position size — use allocated_capital for equal-dollar sizing
        # This ensures Position #6 is the same dollar size as Position #1
        effective_cap = allocated_capital or self._allocated_capital or capital
        position_capital = effective_cap * self.position_size_pct
        entry_price = current_prices.get(symbol, signal.entry_price)
        shares = max(1, int(position_capital / entry_price))

        # Open the new position
        new_pos = position_manager.add_position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            shares=shares,
            stop_price=signal.stop_price,
            target1=signal.target1,
            target2=signal.target2,
            target3=signal.target3,
            scale_number=scale_number,
        )

        if new_pos is None:
            return None

        # Move previous positions' stops to break-even (anti-martingale protection)
        if len(existing) > 0:
            for prev_pos in existing:
                if not prev_pos.breakeven_stop:
                    position_manager.move_stop_to_breakeven(prev_pos.position_id)
                    logger.info(
                        "Stop moved to BE for %s (scale #%d added)",
                        prev_pos.position_id, scale_number
                    )

        self._last_entry_time = backtest_time if (self.backtest_mode and backtest_time is not None) else datetime.now()
        logger.info(
            "Scale-in #%d executed: %s %s %d shares @ %.2f (reason: %s)",
            scale_number, direction, symbol, shares, entry_price, reason
        )

        return new_pos

    def get_scale_in_status(
        self,
        position_manager: PositionManager,
        symbol: str,
        direction: str,
    ) -> dict:
        """Get current scale-in status for a symbol."""
        open_pos = [
            p for p in position_manager.get_open_positions()
            if p.symbol == symbol and p.direction == direction
        ]

        can_scale, reason = self.can_scale_in(position_manager, symbol, direction)

        return {
            "positions_open": len(open_pos),
            "max_positions": self.max_positions,
            "can_scale": can_scale,
            "reason": reason,
            "positions": [
                {
                    "id": p.position_id,
                    "pnl_pct": p.unrealized_pnl_pct,
                    "scale_number": p.scale_number,
                    "breakeven": p.breakeven_stop,
                }
                for p in open_pos
            ],
        }
