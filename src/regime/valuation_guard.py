"""Valuation Guard — KEY INNOVATION.

Applies penalty multiplier to leverage based on:
- SPY overextension vs 200 SMA
- CAPE ratio (Shiller P/E)
- VIX complacency (too-low VIX)
- Months since last 10% correction
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    """Result of valuation guard check."""
    penalty: float               # 0.0-1.0 multiplier
    final_leverage: float        # base_leverage * penalty
    overextended: bool           # SPY > 200SMA * 1.15
    cape_elevated: bool          # CAPE > 35
    vix_complacent: bool         # VIX < 13
    correction_overdue: bool     # months since correction > 18
    details: dict


class ValuationGuard:
    """Applies penalty multipliers to target leverage based on market valuation.

    The friend's bot lacked this — we add it to avoid entering fully
    leveraged at market tops (like April 2026 situation).
    """

    def __init__(
        self,
        overextension_threshold: float = 1.15,
        overextension_penalty: float = 0.75,
        cape_high: float = 35.0,
        cape_penalty: float = 0.75,
        vix_complacency: float = 13.0,
        vix_complacency_penalty: float = 0.80,
        correction_drought_months: int = 18,
        correction_drought_penalty: float = 0.70,
    ):
        self.overextension_threshold = overextension_threshold
        self.overextension_penalty = overextension_penalty
        self.cape_high = cape_high
        self.cape_penalty = cape_penalty
        self.vix_complacency = vix_complacency
        self.vix_complacency_penalty = vix_complacency_penalty
        self.correction_drought_months = correction_drought_months
        self.correction_drought_penalty = correction_drought_penalty

    def compute(
        self,
        base_leverage: float,
        spy_price: float,
        spy_200sma: float,
        vix: float,
        cape_ratio: float = 28.0,
        spy_prices_history: Optional[pd.Series] = None,
    ) -> ValuationResult:
        """Compute valuation-adjusted leverage.

        Args:
            base_leverage: Regime-determined base leverage
            spy_price: Current SPY price
            spy_200sma: Current SPY 200-day SMA
            vix: Current VIX value
            cape_ratio: Shiller CAPE ratio (default 28)
            spy_prices_history: Historical SPY prices for correction check

        Returns:
            ValuationResult with penalty and adjusted leverage
        """
        penalty = 1.0
        details = {}

        # Check 1: SPY overextension
        overextension_ratio = spy_price / spy_200sma if spy_200sma > 0 else 1.0
        overextended = overextension_ratio > self.overextension_threshold
        if overextended:
            penalty *= self.overextension_penalty
            logger.info(
                "Valuation Guard: SPY overextended (%.1f%% above 200SMA), penalty %.2f",
                (overextension_ratio - 1) * 100,
                self.overextension_penalty
            )
        details["overextension_ratio"] = overextension_ratio

        # Check 2: CAPE elevated
        cape_elevated = cape_ratio > self.cape_high
        if cape_elevated:
            penalty *= self.cape_penalty
            logger.info(
                "Valuation Guard: CAPE elevated (%.1f > %.0f), penalty %.2f",
                cape_ratio, self.cape_high, self.cape_penalty
            )
        details["cape_ratio"] = cape_ratio

        # Check 3: VIX complacency
        vix_complacent = vix < self.vix_complacency
        if vix_complacent:
            penalty *= self.vix_complacency_penalty
            logger.info(
                "Valuation Guard: VIX complacency (%.1f < %.0f), penalty %.2f",
                vix, self.vix_complacency, self.vix_complacency_penalty
            )
        details["vix"] = vix

        # Check 4: Correction overdue
        months_since_correction = self._months_since_last_correction(spy_prices_history)
        correction_overdue = months_since_correction > self.correction_drought_months
        if correction_overdue:
            penalty *= self.correction_drought_penalty
            logger.info(
                "Valuation Guard: Correction overdue (%d months), penalty %.2f",
                months_since_correction, self.correction_drought_penalty
            )
        details["months_since_correction"] = months_since_correction

        final_leverage = base_leverage * penalty
        details["penalty"] = penalty

        logger.info(
            "Valuation Guard: base_leverage=%.2f × penalty=%.2f → final=%.2f",
            base_leverage, penalty, final_leverage
        )

        return ValuationResult(
            penalty=penalty,
            final_leverage=final_leverage,
            overextended=overextended,
            cape_elevated=cape_elevated,
            vix_complacent=vix_complacent,
            correction_overdue=correction_overdue,
            details=details,
        )

    def _months_since_last_correction(
        self,
        prices: Optional[pd.Series],
        correction_threshold: float = 0.10,
    ) -> int:
        """Calculate months since last 10%+ correction.

        A correction is defined as a peak-to-trough decline of >= 10%.
        """
        if prices is None or len(prices) < 20:
            return 0

        prices = prices.dropna()
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak

        # Find last time drawdown exceeded -10%
        correction_dates = drawdown[drawdown <= -correction_threshold].index

        if len(correction_dates) == 0:
            # No correction found — use full history length
            return len(prices) // 21  # approximate trading days to months

        last_correction_date = correction_dates[-1]
        current_date = prices.index[-1]

        # Convert to months
        delta = current_date - last_correction_date
        months = int(delta.days / 30.44)
        return max(0, months)
