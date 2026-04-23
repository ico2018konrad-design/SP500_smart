"""Market breadth indicators.

Approximates:
- % of S&P 500 stocks above 50 SMA
- A/D Line (Advance/Decline)

Note: Real breadth data requires a proper data provider.
This module provides:
1. Actual data via Yahoo Finance (RSP/SPY ratio as proxy)
2. Approximate breadth from available data
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_rsp_data(start: str = "2005-01-01") -> pd.DataFrame:
    """Load RSP (Invesco S&P 500 Equal Weight) as breadth proxy."""
    try:
        from src.data.yahoo_loader import load_ohlcv
        return load_ohlcv("RSP", start)
    except Exception:
        return pd.DataFrame()


def estimate_breadth_from_etfs(
    spy_data: pd.DataFrame,
    lookback: int = 50,
) -> pd.Series:
    """Estimate % of stocks above 50 SMA using SPY momentum proxy.

    A simple proxy: if SPY itself is trending well, breadth is likely > 55%.
    More accurate would require individual stock data (expensive).

    This is a simplified proxy for backtesting.
    """
    close = spy_data["Close"]
    sma50 = close.rolling(50).mean()

    # Count "breadth-like" indicator: SPY above SMA50 and momentum positive
    momentum = close.pct_change(20)

    # When SPY is strong, estimate ~65% breadth; when weak, ~35%
    breadth = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        spy_pct_above = (close.iloc[i] - sma50.iloc[i]) / sma50.iloc[i] if i >= 50 else 0
        mom = momentum.iloc[i] if i >= 20 else 0
        # Map to breadth estimate
        breadth.iloc[i] = 0.50 + 0.20 * np.tanh(spy_pct_above * 10) + 0.10 * np.tanh(mom * 20)

    return breadth.clip(0.10, 0.90)


def calc_advance_decline_line(
    spy_data: pd.DataFrame,
    rsp_data: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Calculate approximate A/D line.

    Uses RSP/SPY ratio as a proxy for equal-weight vs cap-weight breadth.
    When RSP outperforms SPY, breadth is improving.
    """
    if rsp_data is not None and not rsp_data.empty:
        try:
            common = spy_data.index.intersection(rsp_data.index)
            spy_close = spy_data.loc[common, "Close"]
            rsp_close = rsp_data.loc[common, "Close"]
            ratio = rsp_close / spy_close
            # Normalize to start at 100
            ad_line = ratio / ratio.iloc[0] * 100
            ad_line.name = "AD_Line_Proxy"
            return ad_line
        except Exception as exc:
            logger.warning("A/D line calc error: %s", exc)

    # Fallback: use SPY momentum as proxy
    close = spy_data["Close"]
    ad = close / close.iloc[0] * 100 if len(close) > 0 else close
    ad.name = "AD_Line_Proxy"
    return ad


def is_breadth_rising(ad_line: pd.Series, lookback: int = 5) -> bool:
    """Check if A/D line is in uptrend over lookback period."""
    if len(ad_line) < lookback + 1:
        return True
    recent = ad_line.dropna().tail(lookback)
    return float(recent.iloc[-1]) > float(recent.iloc[0])


def get_current_breadth_pct() -> float:
    """Get current estimated breadth (% of stocks above 50 SMA).

    Returns approximate value in 0.0-1.0 range.
    Falls back to 0.55 (neutral) if data unavailable.
    """
    try:
        from src.data.yahoo_loader import load_spy
        spy = load_spy()
        if not spy.empty:
            breadth = estimate_breadth_from_etfs(spy)
            if not breadth.empty:
                return float(breadth.dropna().iloc[-1])
    except Exception as exc:
        logger.warning("Breadth estimation failed: %s", exc)
    return 0.55
