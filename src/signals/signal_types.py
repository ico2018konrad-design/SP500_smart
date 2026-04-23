"""Signal data classes and enumerations."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from datetime import datetime


class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SignalStrength(Enum):
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


@dataclass
class Signal:
    """Trading signal with full entry details."""
    direction: SignalDirection
    symbol: str
    entry_price: float
    stop_price: float
    target1: float
    target2: float
    target3: float
    timestamp: datetime
    regime_score: int
    regime: str

    # Signal validation details
    setup_valid: bool = False
    trigger_count: int = 0
    confirm_count: int = 0

    # Risk metrics
    risk_pct: float = 0.015
    rr_ratio: float = 0.0

    # Trigger details
    triggers_hit: List[str] = field(default_factory=list)
    confirms_hit: List[str] = field(default_factory=list)

    # Signal strength
    strength: SignalStrength = SignalStrength.MODERATE

    def is_valid(self) -> bool:
        """Check if signal meets minimum requirements."""
        return self.setup_valid and self.trigger_count >= 3 and self.confirm_count >= 2

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target1": self.target1,
            "target2": self.target2,
            "target3": self.target3,
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime,
            "regime_score": self.regime_score,
            "setup_valid": self.setup_valid,
            "trigger_count": self.trigger_count,
            "confirm_count": self.confirm_count,
            "rr_ratio": self.rr_ratio,
            "valid": self.is_valid(),
            "triggers_hit": self.triggers_hit,
            "confirms_hit": self.confirms_hit,
        }


@dataclass
class EventCalendar:
    """Major economic events that trigger blackout periods."""
    event_type: str   # FOMC, CPI, NFP, etc.
    date: datetime
    is_major: bool = True
    hours_blackout: int = 24
