"""FRED (Federal Reserve Economic Data) macro data loader.

Requires FRED_API_KEY in environment variables.
Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs
SERIES_HY_SPREAD = "BAMLH0A0HYM2"   # BofA US HY OAS (bps)
SERIES_YIELD_CURVE = "T10Y2Y"         # 10Y-2Y Treasury spread
SERIES_T10Y = "DGS10"                 # 10-Year Treasury
SERIES_T2Y = "DGS2"                   # 2-Year Treasury


def _get_fred_client():
    """Create FRED client, warn if API key not set."""
    try:
        from fredapi import Fred
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            logger.warning(
                "FRED_API_KEY not set in environment. "
                "Macro data will use fallback values. "
                "Get free key: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return None
        return Fred(api_key=api_key)
    except ImportError:
        logger.error("fredapi not installed. Run: pip install fredapi")
        return None


def load_hy_spread(
    start: str = "2005-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Load BofA US High Yield OAS spread (in basis points).

    Returns Series indexed by date. Values are in basis points.
    > 400 bps = elevated credit risk
    """
    fred = _get_fred_client()
    if fred is None:
        return _fallback_hy_spread(start, end)

    try:
        data = fred.get_series(SERIES_HY_SPREAD, observation_start=start,
                               observation_end=end)
        data.name = "hy_spread_bps"
        logger.info("Loaded HY spread: %d observations", len(data))
        return data
    except Exception as exc:
        logger.error("Failed to load HY spread from FRED: %s", exc)
        return _fallback_hy_spread(start, end)


def load_yield_curve(
    start: str = "2005-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Load 10Y-2Y Treasury yield spread.

    Negative = inverted yield curve (bearish signal).
    """
    fred = _get_fred_client()
    if fred is None:
        return _fallback_yield_curve(start, end)

    try:
        data = fred.get_series(SERIES_YIELD_CURVE, observation_start=start,
                               observation_end=end)
        data.name = "yield_curve_10y2y"
        logger.info("Loaded yield curve: %d observations", len(data))
        return data
    except Exception as exc:
        logger.error("Failed to load yield curve from FRED: %s", exc)
        return _fallback_yield_curve(start, end)


def load_macro_data(
    start: str = "2005-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load all macro indicators and combine into DataFrame.

    Returns DataFrame with columns: hy_spread_bps, yield_curve_10y2y
    """
    hy = load_hy_spread(start, end)
    yc = load_yield_curve(start, end)

    df = pd.DataFrame({"hy_spread_bps": hy, "yield_curve_10y2y": yc})
    df = df.ffill().bfill()
    return df


def get_current_hy_spread() -> float:
    """Get current HY spread value (most recent available).

    Returns value in basis points.
    """
    fred = _get_fred_client()
    if fred is None:
        logger.warning("Using fallback HY spread: 350 bps")
        return 350.0

    try:
        data = fred.get_series(SERIES_HY_SPREAD)
        return float(data.dropna().iloc[-1])
    except Exception as exc:
        logger.error("Failed to get current HY spread: %s", exc)
        return 350.0


def get_current_yield_curve() -> float:
    """Get current 10Y-2Y yield spread.

    Returns value in percentage points.
    """
    fred = _get_fred_client()
    if fred is None:
        logger.warning("Using fallback yield curve: 0.5%")
        return 0.5

    try:
        data = fred.get_series(SERIES_YIELD_CURVE)
        return float(data.dropna().iloc[-1])
    except Exception as exc:
        logger.error("Failed to get current yield curve: %s", exc)
        return 0.5


def is_yield_curve_freshly_inverted(lookback_days: int = 30) -> bool:
    """Check if yield curve became freshly inverted in last N days.

    Fresh inversion = curve crossed below zero in lookback window.
    """
    fred = _get_fred_client()
    if fred is None:
        return False

    try:
        data = fred.get_series(SERIES_YIELD_CURVE)
        recent = data.dropna().tail(lookback_days)
        if len(recent) < 2:
            return False
        # Check if any crossover from positive to negative occurred
        for i in range(1, len(recent)):
            if recent.iloc[i - 1] >= 0 and recent.iloc[i] < 0:
                return True
        return False
    except Exception as exc:
        logger.error("Failed to check yield curve inversion: %s", exc)
        return False


# ─── FALLBACK FUNCTIONS (when FRED API not available) ────────────────────────

def _fallback_hy_spread(start: str, end: Optional[str]) -> pd.Series:
    """Return reasonable historical HY spread approximation."""
    logger.warning("Using fallback HY spread data (FRED API unavailable)")
    idx = pd.date_range(start=start, end=end or datetime.now(), freq="B")
    # Rough historical average ~350 bps
    data = pd.Series(350.0, index=idx, name="hy_spread_bps")
    return data


def _fallback_yield_curve(start: str, end: Optional[str]) -> pd.Series:
    """Return reasonable yield curve approximation."""
    logger.warning("Using fallback yield curve data (FRED API unavailable)")
    idx = pd.date_range(start=start, end=end or datetime.now(), freq="B")
    data = pd.Series(0.5, index=idx, name="yield_curve_10y2y")
    return data


def get_cape_ratio() -> float:
    """Fetch Shiller CAPE ratio.

    Tries FRED first, falls back to reasonable estimate.
    Note: FRED series 'CAPE' may not be available. Returns current estimate.
    """
    fred = _get_fred_client()
    if fred is not None:
        try:
            data = fred.get_series("CAPE")
            val = float(data.dropna().iloc[-1])
            logger.info("Current CAPE ratio: %.1f", val)
            return val
        except Exception:
            pass

    # Fallback: use yfinance to compute P/E approximation
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        info = spy.info
        pe = info.get("trailingPE", 28.0)
        logger.warning("Using trailing P/E as CAPE proxy: %.1f", pe)
        return float(pe) if pe else 28.0
    except Exception:
        logger.warning("CAPE unavailable, using default 28.0")
        return 28.0
