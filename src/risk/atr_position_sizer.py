"""ATR-based position sizer — volatility-adjusted sizing.

Instead of fixed % allocation:
  position_size = target_risk_per_trade / ATR

This keeps risk constant regardless of volatility:
- Low volatility (ATR low) → larger position
- High volatility (ATR high) → smaller position

This is the Renaissance/CTA approach to position sizing.
"""
import logging

import pandas as pd

from src.signals.indicators import calc_atr

logger = logging.getLogger(__name__)


class ATRPositionSizer:
    """Calculates position size based on ATR for constant risk per trade.

    Formula:
        shares = (capital × risk_pct) / (ATR × atr_multiplier)
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 1.0,  # ATR as stop distance
        min_shares: int = 1,
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.min_shares = min_shares

    def calculate_shares(
        self,
        capital: float,
        risk_pct: float,
        entry_price: float,
        stop_price: float,
        max_position_pct: float = 0.25,
    ) -> dict:
        """Calculate number of shares based on risk and stop distance.

        Args:
            capital: Available capital
            risk_pct: Risk per trade as decimal (e.g., 0.015 = 1.5%)
            entry_price: Entry price per share
            stop_price: Stop loss price per share
            max_position_pct: Max % of capital in single position (default 25%)

        Returns:
            dict with shares, position_value, actual_risk_pct
        """
        if entry_price <= 0 or stop_price <= 0:
            logger.warning("Invalid prices: entry=%.2f, stop=%.2f", entry_price, stop_price)
            return {"shares": self.min_shares, "position_value": entry_price, "actual_risk_pct": 0}

        dollar_risk = capital * risk_pct
        stop_distance = abs(entry_price - stop_price)

        if stop_distance <= 0:
            logger.warning("Stop distance is zero, using 1%% stop")
            stop_distance = entry_price * 0.01

        # Primary calculation: shares based on dollar risk
        shares_by_risk = int(dollar_risk / stop_distance)

        # Cap by max position size
        max_shares = int((capital * max_position_pct) / entry_price)
        shares = max(self.min_shares, min(shares_by_risk, max_shares))

        position_value = shares * entry_price
        actual_risk = shares * stop_distance
        actual_risk_pct = actual_risk / capital if capital > 0 else 0

        logger.debug(
            "ATR sizer: capital=$%.0f, risk=%.1f%%, stop_dist=$%.2f → "
            "%d shares (value=$%.0f, actual_risk=%.2f%%)",
            capital, risk_pct * 100, stop_distance,
            shares, position_value, actual_risk_pct * 100,
        )

        return {
            "shares": shares,
            "position_value": position_value,
            "actual_risk_pct": actual_risk_pct,
            "dollar_risk": actual_risk,
            "stop_distance": stop_distance,
        }

    def calculate_shares_atr(
        self,
        capital: float,
        risk_pct: float,
        entry_price: float,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        max_position_pct: float = 0.25,
    ) -> dict:
        """Calculate shares using ATR as stop distance.

        Stop distance = ATR(14) × multiplier
        """
        atr = calc_atr(high, low, close, self.atr_period)
        current_atr = float(atr.dropna().iloc[-1]) if len(atr.dropna()) > 0 else entry_price * 0.01

        stop_price = entry_price - current_atr * self.atr_multiplier
        return self.calculate_shares(
            capital, risk_pct, entry_price, stop_price, max_position_pct
        )

    def get_atr_stop(
        self,
        entry_price: float,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        direction: str = "LONG",
        multiplier: float = 1.5,
    ) -> float:
        """Get ATR-based stop price.

        LONG: entry - N×ATR
        SHORT: entry + N×ATR
        """
        atr = calc_atr(high, low, close, self.atr_period)
        current_atr = float(atr.dropna().iloc[-1]) if len(atr.dropna()) > 0 else entry_price * 0.01

        if direction == "LONG":
            return entry_price - multiplier * current_atr
        else:
            return entry_price + multiplier * current_atr
