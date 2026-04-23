"""Regime type enumerations and data classes."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class Regime(Enum):
    """Market regime classification."""
    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    CHOP = "CHOP"
    CAUTION = "CAUTION"
    BEAR = "BEAR"


REGIME_LEVERAGE = {
    Regime.STRONG_BULL: 2.5,
    Regime.BULL: 2.0,
    Regime.CHOP: 1.0,
    Regime.CAUTION: 0.5,
    Regime.BEAR: 0.0,
}

REGIME_RISK_PCT = {
    Regime.STRONG_BULL: 0.020,
    Regime.BULL: 0.015,
    Regime.CHOP: 0.010,
    Regime.CAUTION: 0.005,
    Regime.BEAR: 0.010,
}


def score_to_regime(score: int) -> Regime:
    """Convert regime score (0-11) to Regime enum.

    10-11 → STRONG_BULL
    8-9   → BULL
    6-7   → CHOP
    4-5   → CAUTION
    0-3   → BEAR
    """
    if score >= 10:
        return Regime.STRONG_BULL
    elif score >= 8:
        return Regime.BULL
    elif score >= 6:
        return Regime.CHOP
    elif score >= 4:
        return Regime.CAUTION
    else:
        return Regime.BEAR


@dataclass
class RegimeResult:
    """Result of regime detection."""
    score: int
    regime: Regime
    max_leverage: float
    risk_pct: float
    indicator_scores: Dict[str, int] = field(default_factory=dict)
    details: Dict[str, float] = field(default_factory=dict)

    # Individual indicator flags
    spy_above_200sma: bool = False
    spy_above_50sma: bool = False
    sma200_slope_positive: bool = False
    higher_highs_lows: bool = False
    macd_weekly_bullish: bool = False
    adx_bullish: bool = False
    vix_below_20: bool = False
    vix_term_contango: bool = False
    breadth_above_55pct: bool = False
    hy_spread_ok: bool = False
    yield_curve_ok: bool = False

    timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "regime": self.regime.value,
            "max_leverage": self.max_leverage,
            "risk_pct": self.risk_pct,
            "indicators": self.indicator_scores,
            "spy_above_200sma": self.spy_above_200sma,
            "spy_above_50sma": self.spy_above_50sma,
            "vix_below_20": self.vix_below_20,
            "timestamp": self.timestamp,
        }
