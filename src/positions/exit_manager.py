"""Exit Manager — handles all exit logic.

A) Profit Ladder: T1 (+2%), T2 (+4.5%), T3 (+8%)
B) Trailing Stop: activated after +3% gain
C) Regime-based Forced Exit
D) Time Stop: 10 days without movement
E) Stop Loss hits
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.positions.position_manager import Position, PositionManager, PositionStatus
from src.signals.indicators import calc_atr
from src.regime.regime_types import Regime

logger = logging.getLogger(__name__)


class ExitManager:
    """Manages all position exits."""

    def __init__(
        self,
        trailing_activate_pct: float = 0.03,    # activate after +3%
        trailing_atr_multiplier: float = 1.5,
        time_stop_days: int = 10,
        time_stop_movement: float = 0.01,
        vix_forced_exit_threshold: float = 30.0,
        spy_flash_crash_pct: float = -0.02,
    ):
        self.trailing_activate_pct = trailing_activate_pct
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.time_stop_days = time_stop_days
        self.time_stop_movement = time_stop_movement
        self.vix_forced_exit_threshold = vix_forced_exit_threshold
        self.spy_flash_crash_pct = spy_flash_crash_pct

    def check_exits(
        self,
        position_manager: PositionManager,
        current_prices: Dict[str, float],
        atr_values: Dict[str, float],
        current_regime: Optional[Regime] = None,
        previous_regime: Optional[Regime] = None,
        vix: float = 20.0,
        spy_1h_return: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> List[Tuple[str, str, float]]:
        """Check all exit conditions for open positions.

        Returns list of (position_id, exit_reason, close_price)
        """
        if current_time is None:
            current_time = datetime.now()

        exits = []

        for pos in position_manager.get_open_positions():
            current_price = current_prices.get(pos.symbol, pos.current_price)

            # A) Stop loss hit
            if self._check_stop_loss(pos, current_price):
                exits.append((pos.position_id, "stop_loss", current_price))
                continue

            # C) Flash crash protection
            if spy_1h_return <= self.spy_flash_crash_pct:
                exits.append((pos.position_id, "flash_crash_protection", current_price))
                continue

            # C) VIX forced exit
            if vix > self.vix_forced_exit_threshold and pos.direction == "LONG":
                exits.append((pos.position_id, f"vix_above_{self.vix_forced_exit_threshold:.0f}", current_price))
                continue

            # C) Regime forced exit
            regime_exit = self._check_regime_exit(pos, current_regime, previous_regime)
            if regime_exit:
                exits.append((pos.position_id, f"regime_exit_{regime_exit}", current_price))
                continue

            # D) Time stop
            if self._check_time_stop(pos, current_time):
                exits.append((pos.position_id, "time_stop", current_price))
                continue

        return exits

    def check_partial_exits(
        self,
        position_manager: PositionManager,
        current_prices: Dict[str, float],
        atr_values: Dict[str, float],
    ) -> List[Tuple[str, str, float, float]]:
        """Check for partial exits at targets.

        Returns list of (position_id, reason, close_price, fraction)
        """
        partial_exits = []

        for pos in position_manager.get_open_positions():
            current_price = current_prices.get(pos.symbol, pos.current_price)
            pnl_pct = pos.unrealized_pnl_pct

            # A) Profit ladder — T1 hit (+2%)
            if not pos.target1_hit and current_price >= pos.target1 and pos.direction == "LONG":
                partial_exits.append((pos.position_id, "target1_hit", current_price, 0.333))
                pos.target1_hit = True
                # Move stop to break-even after T1
                pos.stop_price = pos.entry_price
                pos.breakeven_stop = True
                logger.info("T1 hit: %s, stop moved to BE", pos.position_id)

            elif not pos.target1_hit and current_price <= pos.target1 and pos.direction == "SHORT":
                partial_exits.append((pos.position_id, "target1_hit_short", current_price, 0.333))
                pos.target1_hit = True
                pos.stop_price = pos.entry_price
                pos.breakeven_stop = True

            # A) T2 hit (+4.5%)
            if pos.target1_hit and not pos.target2_hit:
                if (pos.direction == "LONG" and current_price >= pos.target2) or \
                   (pos.direction == "SHORT" and current_price <= pos.target2):
                    partial_exits.append((pos.position_id, "target2_hit", current_price, 0.333))
                    pos.target2_hit = True
                    # Move stop to +2%
                    if pos.direction == "LONG":
                        pos.stop_price = pos.entry_price * 1.02
                    else:
                        pos.stop_price = pos.entry_price * 0.98
                    logger.info("T2 hit: %s, stop moved to +2%%", pos.position_id)

            # B) Activate trailing stop after +3%
            if pnl_pct >= self.trailing_activate_pct and not pos.trailing_stop_active:
                pos.trailing_stop_active = True
                pos.trailing_stop_high = current_price
                logger.info("Trailing stop activated: %s at %.2f", pos.position_id, current_price)

        return partial_exits

    def update_trailing_stops(
        self,
        position_manager: PositionManager,
        current_prices: Dict[str, float],
        atr_values: Dict[str, float],
    ) -> None:
        """Update trailing stops for all active trailing stop positions."""
        for pos in position_manager.get_open_positions():
            if not pos.trailing_stop_active:
                continue

            current_price = current_prices.get(pos.symbol, pos.current_price)
            atr = atr_values.get(pos.symbol, current_price * 0.01)

            if pos.direction == "LONG":
                # Update high water mark
                if current_price > pos.trailing_stop_high:
                    pos.trailing_stop_high = current_price

                # New stop = HH - 1.5×ATR
                new_stop = pos.trailing_stop_high - self.trailing_atr_multiplier * atr
                if new_stop > pos.stop_price:
                    pos.stop_price = new_stop
                    logger.debug(
                        "Trailing stop updated: %s → %.2f (HH=%.2f, ATR=%.2f)",
                        pos.position_id, new_stop, pos.trailing_stop_high, atr
                    )

    def _check_stop_loss(self, pos: Position, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        if pos.direction == "LONG":
            return current_price <= pos.stop_price
        else:
            return current_price >= pos.stop_price

    def _check_regime_exit(
        self,
        pos: Position,
        current_regime: Optional[Regime],
        previous_regime: Optional[Regime],
    ) -> Optional[str]:
        """Check regime-based forced exits."""
        if current_regime is None or previous_regime is None:
            return None

        if pos.direction == "LONG":
            # BULL → CHOP: signal UPRO exit (return 'leveraged' for UPRO symbol)
            if previous_regime == Regime.BULL and current_regime == Regime.CHOP:
                if pos.symbol == "UPRO":
                    return "bull_to_chop_leveraged"

            # CHOP → BEAR: flat all longs
            if previous_regime == Regime.CHOP and current_regime == Regime.BEAR:
                return "chop_to_bear"

        return None

    def _check_time_stop(self, pos: Position, current_time: datetime) -> bool:
        """Check time stop: position open > 10 days without ±1% movement."""
        days_open = (current_time - pos.entry_time).days
        if days_open < self.time_stop_days:
            return False

        # If price hasn't moved ±1% from entry
        pnl_pct = abs(pos.unrealized_pnl_pct)
        return pnl_pct < self.time_stop_movement
