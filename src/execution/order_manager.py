"""Order Manager — tracks order placement and fills."""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    order_id: str
    symbol: str
    action: str        # BUY or SELL
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "filled_price": self.filled_price,
            "created_at": self.created_at.isoformat(),
        }


class OrderManager:
    """Manages order creation, submission, and tracking."""

    def __init__(self):
        self.orders: Dict[str, Order] = {}

    def create_market_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        notes: str = "",
    ) -> Order:
        """Create a market order."""
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=OrderType.MARKET,
            notes=notes,
        )
        self.orders[order.order_id] = order
        logger.info("Order created: %s %s %d %s", order.order_id, action, quantity, symbol)
        return order

    def create_stop_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_price: float,
    ) -> Order:
        """Create a stop order."""
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
        )
        self.orders[order.order_id] = order
        return order

    def simulate_fill(self, order_id: str, fill_price: float) -> bool:
        """Simulate order fill (for paper trading)."""
        if order_id not in self.orders:
            return False
        order = self.orders[order_id]
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.filled_at = datetime.now()
        logger.info(
            "Order filled: %s @ %.2f (%s %d %s)",
            order_id, fill_price, order.action, order.quantity, order.symbol
        )
        return True

    def get_pending_orders(self) -> List[Order]:
        return [o for o in self.orders.values() if o.status == OrderStatus.PENDING]

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
