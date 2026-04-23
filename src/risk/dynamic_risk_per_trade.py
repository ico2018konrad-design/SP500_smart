"""Dynamic risk per trade calculator.

Risk varies by:
- Regime (0.5% to 2%)
- Recent win rate (last 5 trades)
- Volatility ratio (current ATR vs 252d average)

Hard cap: 2% maximum risk per trade.
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.regime.regime_types import Regime, REGIME_RISK_PCT

logger = logging.getLogger(__name__)


class DynamicRiskCalculator:
    """Calculates dynamic risk per trade.

    Algorithm:
    1. Start with regime-based risk (0.5-2%)
    2. Adjust for recent win rate (last 5 trades)
    3. Adjust for current volatility vs historical average
    4. Cap at 2%
    """

    def __init__(
        self,
        min_risk_pct: float = 0.005,
        max_risk_pct: float = 0.020,
        win_rate_low: float = 0.40,
        win_rate_low_multiplier: float = 0.50,
        win_rate_high: float = 0.70,
        win_rate_high_multiplier: float = 1.20,
        performance_lookback: int = 5,
    ):
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        self.win_rate_low = win_rate_low
        self.win_rate_low_multiplier = win_rate_low_multiplier
        self.win_rate_high = win_rate_high
        self.win_rate_high_multiplier = win_rate_high_multiplier
        self.performance_lookback = performance_lookback

    def calculate(
        self,
        regime: Regime,
        recent_trades: Optional[List[bool]] = None,   # True=win, False=loss
        current_atr: Optional[float] = None,
        avg_atr_252d: Optional[float] = None,
    ) -> float:
        """Calculate dynamic risk for next trade.

        Args:
            regime: Current market regime
            recent_trades: List of True/False for last N trades (True=win)
            current_atr: Current ATR value
            avg_atr_252d: 252-day average ATR for normalization

        Returns:
            Risk percentage as decimal (e.g., 0.015 = 1.5%)
        """
        # Step 1: Base risk from regime
        base = REGIME_RISK_PCT.get(regime, 0.015)
        logger.debug("Base risk for %s: %.1f%%", regime.value, base * 100)

        # Step 2: Adjust for recent performance
        if recent_trades and len(recent_trades) >= 3:
            lookback = recent_trades[-self.performance_lookback:]
            win_rate = sum(1 for t in lookback if t) / len(lookback)

            if win_rate < self.win_rate_low:
                base *= self.win_rate_low_multiplier
                logger.info(
                    "Risk reduced: win rate %.0f%% < %.0f%% → risk=%.1f%%",
                    win_rate * 100, self.win_rate_low * 100, base * 100
                )
            elif win_rate > self.win_rate_high:
                base *= self.win_rate_high_multiplier
                logger.info(
                    "Risk increased: win rate %.0f%% > %.0f%% → risk=%.1f%%",
                    win_rate * 100, self.win_rate_high * 100, base * 100
                )

        # Step 3: Volatility normalization
        if current_atr is not None and avg_atr_252d is not None and avg_atr_252d > 0:
            vol_ratio = current_atr / avg_atr_252d
            if vol_ratio > 0:
                base /= vol_ratio
                logger.debug(
                    "Vol adjustment: current_atr=%.2f, avg=%.2f, ratio=%.2f → risk=%.1f%%",
                    current_atr, avg_atr_252d, vol_ratio, base * 100
                )

        # Step 4: Apply hard limits
        final_risk = max(self.min_risk_pct, min(base, self.max_risk_pct))

        logger.debug("Final risk: %.2f%%", final_risk * 100)
        return final_risk

    def get_consecutive_losses(self, trades: List[bool]) -> int:
        """Count consecutive losses at end of trade list."""
        count = 0
        for trade in reversed(trades):
            if not trade:
                count += 1
            else:
                break
        return count
