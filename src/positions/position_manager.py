"""Position Manager — tracks up to 6 concurrent positions with stops and targets."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


@dataclass
class Position:
    """Individual position with stop, targets, and trailing stop."""
    position_id: str
    symbol: str
    direction: str          # LONG or SHORT
    entry_price: float
    current_price: float
    shares: int
    stop_price: float
    target1: float
    target2: float
    target3: float
    entry_time: datetime
    status: PositionStatus = PositionStatus.OPEN

    # Scaling info
    scale_number: int = 1   # 1-6

    # Target hit tracking
    target1_hit: bool = False
    target2_hit: bool = False
    target3_hit: bool = False

    # Partial closes
    shares_closed: int = 0
    realized_pnl: float = 0.0

    # Stop management
    original_stop: float = 0.0
    breakeven_stop: bool = False  # stop moved to BE
    trailing_stop_active: bool = False
    trailing_stop_high: float = 0.0   # highest price seen

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized PnL as % of entry."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    @property
    def is_in_profit(self) -> bool:
        return self.unrealized_pnl_pct > 0

    @property
    def shares_open(self) -> int:
        return self.shares - self.shares_closed

    def update_price(self, new_price: float) -> None:
        """Update current price and trailing stop if active."""
        self.current_price = new_price

        if self.trailing_stop_active and self.direction == "LONG":
            if new_price > self.trailing_stop_high:
                self.trailing_stop_high = new_price

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "shares": self.shares,
            "shares_open": self.shares_open,
            "stop_price": self.stop_price,
            "target1": self.target1,
            "target2": self.target2,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "status": self.status.value,
            "scale_number": self.scale_number,
            "breakeven_stop": self.breakeven_stop,
            "trailing_stop_active": self.trailing_stop_active,
            "entry_time": self.entry_time.isoformat(),
        }


class PositionManager:
    """Manages up to 6 concurrent positions.

    Tracks positions, their states, and enforces position limits.
    Works with AntiMartingaleScaler for scale-in decisions.
    """

    MAX_POSITIONS = 6

    def __init__(self, max_positions: int = 6):
        self.max_positions = max_positions
        self.positions: Dict[str, Position] = {}
        self._position_counter = 0

    def _generate_id(self, symbol: str) -> str:
        self._position_counter += 1
        return f"{symbol}_{self._position_counter:04d}"

    def can_add_position(self) -> bool:
        """Check if we can open a new position."""
        open_count = sum(
            1 for p in self.positions.values()
            if p.status == PositionStatus.OPEN
        )
        return open_count < self.max_positions

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_position_count(self) -> int:
        """Get count of open positions."""
        return len(self.get_open_positions())

    def has_position(self, symbol: str) -> bool:
        """Check if symbol has an open position."""
        return any(
            p.symbol == symbol and p.status == PositionStatus.OPEN
            for p in self.positions.values()
        )

    def has_short_position(self) -> bool:
        """Check if any short position is open."""
        return any(
            p.direction == "SHORT" and p.status == PositionStatus.OPEN
            for p in self.positions.values()
        )

    def add_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_price: float,
        target1: float,
        target2: float,
        target3: float,
        scale_number: int = 1,
    ) -> Optional[Position]:
        """Open a new position.

        Returns Position if successful, None if at max capacity.
        """
        if not self.can_add_position():
            logger.warning(
                "Cannot add position: at max capacity (%d/%d)",
                self.get_position_count(), self.max_positions
            )
            return None

        pos_id = self._generate_id(symbol)
        position = Position(
            position_id=pos_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=entry_price,
            shares=shares,
            stop_price=stop_price,
            target1=target1,
            target2=target2,
            target3=target3,
            entry_time=datetime.now(),
            scale_number=scale_number,
            original_stop=stop_price,
        )

        self.positions[pos_id] = position
        logger.info(
            "Position opened: %s %s %d @ %.2f (stop=%.2f, #%d)",
            pos_id, direction, shares, entry_price, stop_price, scale_number
        )
        return position

    def close_position(self, position_id: str, close_price: float, reason: str = "manual") -> Optional[float]:
        """Close a position fully.

        Returns realized PnL as % if successful.
        """
        if position_id not in self.positions:
            logger.warning("Position %s not found", position_id)
            return None

        pos = self.positions[position_id]
        if pos.status != PositionStatus.OPEN:
            return None

        if pos.direction == "LONG":
            pnl_pct = (close_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - close_price) / pos.entry_price

        pos.realized_pnl += pnl_pct * pos.shares_open
        pos.shares_closed = pos.shares
        pos.status = PositionStatus.CLOSED
        pos.current_price = close_price

        logger.info(
            "Position closed: %s @ %.2f | PnL: %+.2f%% | Reason: %s",
            position_id, close_price, pnl_pct * 100, reason
        )
        return pnl_pct

    def partial_close(
        self,
        position_id: str,
        close_price: float,
        fraction: float = 0.333,
        reason: str = "target_hit",
    ) -> Optional[float]:
        """Partially close a position (e.g., 1/3 at target).

        Args:
            fraction: Fraction to close (0.333 = close 1/3)
        """
        if position_id not in self.positions:
            return None

        pos = self.positions[position_id]
        if pos.status != PositionStatus.OPEN:
            return None

        shares_to_close = max(1, int(pos.shares_open * fraction))

        if pos.direction == "LONG":
            pnl_pct = (close_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - close_price) / pos.entry_price

        pos.shares_closed += shares_to_close
        pos.realized_pnl += pnl_pct * shares_to_close

        if pos.shares_open <= 0:
            pos.status = PositionStatus.CLOSED

        logger.info(
            "Partial close: %s %d shares @ %.2f | PnL: %+.2f%% | Reason: %s",
            position_id, shares_to_close, close_price, pnl_pct * 100, reason
        )
        return pnl_pct

    def update_prices(self, price_updates: Dict[str, float]) -> None:
        """Update current prices for all open positions.

        Args:
            price_updates: Dict of {symbol: current_price}
        """
        for pos in self.get_open_positions():
            if pos.symbol in price_updates:
                pos.update_price(price_updates[pos.symbol])

    def move_stop_to_breakeven(self, position_id: str) -> bool:
        """Move stop loss to break-even (entry price)."""
        if position_id not in self.positions:
            return False
        pos = self.positions[position_id]
        pos.stop_price = pos.entry_price
        pos.breakeven_stop = True
        logger.info("Stop moved to breakeven for %s: %.2f", position_id, pos.entry_price)
        return True

    def activate_trailing_stop(self, position_id: str, atr: float, multiplier: float = 1.5) -> bool:
        """Activate trailing stop at HH - N×ATR."""
        if position_id not in self.positions:
            return False
        pos = self.positions[position_id]
        pos.trailing_stop_active = True
        pos.trailing_stop_high = pos.current_price
        logger.info("Trailing stop activated for %s (ATR=%.2f)", position_id, atr)
        return True

    def update_trailing_stops(self, atr_values: Dict[str, float], multiplier: float = 1.5) -> None:
        """Update trailing stops for all positions that have them active."""
        for pos in self.get_open_positions():
            if pos.trailing_stop_active and pos.symbol in atr_values:
                atr = atr_values[pos.symbol]
                new_stop = pos.trailing_stop_high - multiplier * atr
                if new_stop > pos.stop_price:
                    pos.stop_price = new_stop

    def flat_all(self, prices: Dict[str, float], reason: str = "forced_exit") -> float:
        """Close all open positions at current prices.

        Returns total realized PnL %.
        """
        total_pnl = 0.0
        for pos in self.get_open_positions():
            price = prices.get(pos.symbol, pos.current_price)
            pnl = self.close_position(pos.position_id, price, reason=reason)
            if pnl is not None:
                total_pnl += pnl

        logger.info("FLAT ALL executed: reason=%s, total_pnl=%.2f%%", reason, total_pnl * 100)
        return total_pnl

    def get_summary(self) -> dict:
        """Get summary of all positions."""
        open_positions = self.get_open_positions()
        total_unrealized = sum(p.unrealized_pnl_pct for p in open_positions)

        return {
            "open_count": len(open_positions),
            "max_positions": self.max_positions,
            "positions": [p.to_dict() for p in open_positions],
            "total_unrealized_pnl_pct": total_unrealized,
        }
