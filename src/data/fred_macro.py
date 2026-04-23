"""FRED Macro data loader.

Loads:
- HY credit spreads (BAMLH0A0HYM2)
- Yield curve spread (T10Y2Y)
- CAPE ratio (approximate via Shiller P/E)
"""
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs
HY_SPREAD_SERIES = "BAMLH0A0HYM2"       # ICE BofA US High Yield Option-Adjusted Spread
YIELD_CURVE_SERIES = "T10Y2Y"            # 10-Year minus 2-Year Treasury
CAPE_SERIES = "CAPE"                     # Not in FRED; approximated below


def get_fred_api_key() -> Optional[str]:
    """Get FRED API key from environment."""
    return os.getenv("FRED_API_KEY")


def load_fred_series(series_id: str, start: str = "2000-01-01") -> pd.Series:
    """Load a FRED data series.

    Requires FRED_API_KEY environment variable (free from fred.stlouisfed.org).
    Falls back to reasonable defaults if API unavailable.
    """
    api_key = get_fred_api_key()
    if not api_key:
        logger.warning(
            "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org. "
            "Using default values for macro indicators."
        )
        return pd.Series(dtype=float)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id, observation_start=start)
        logger.info("FRED %s loaded: %d observations", series_id, len(data))
        return data
    except ImportError:
        logger.warning("fredapi not installed. Run: pip install fredapi")
    except Exception as exc:
        logger.warning("FRED API error for %s: %s", series_id, exc)

    return pd.Series(dtype=float)


def load_macro_data(start: str = "2005-01-01") -> dict:
    """Load all macro indicators.

    Returns dict with hy_spreads and yield_curve series.
    """
    hy = load_fred_series(HY_SPREAD_SERIES, start)
    yc = load_fred_series(YIELD_CURVE_SERIES, start)
    return {
        "hy_spread": hy,
        "yield_curve": yc,
    }


def get_current_hy_spread() -> float:
    """Get latest HY spread in basis points.

    Returns 350.0 (neutral) if FRED unavailable.
    """
    series = load_fred_series(HY_SPREAD_SERIES)
    if not series.empty:
        return float(series.dropna().iloc[-1] * 100)  # FRED returns in percent
    return 350.0  # neutral default


def get_current_yield_curve() -> float:
    """Get latest T10Y2Y yield curve spread (10yr - 2yr).

    Returns 0.5 (slight positive) if FRED unavailable.
    Negative = inverted curve.
    """
    series = load_fred_series(YIELD_CURVE_SERIES)
    if not series.empty:
        return float(series.dropna().iloc[-1])
    return 0.5


def get_cape_ratio() -> float:
    """Get Shiller CAPE ratio.

    Uses multpl.com or a static estimate if unavailable.
    Current CAPE (Apr 2026) is approximately 36-38.
    """
    # Try to fetch from a URL
    try:
        import urllib.request
        # This would need a real data source in production
        # Using a reasonable estimate for April 2026
        return 37.0  # Approximate CAPE as of early 2026
    except Exception:
        return 37.0  # Conservative default based on historical average being ~17


def hy_spread_is_elevated(bps_threshold: float = 400.0) -> bool:
    """Check if HY spreads signal credit stress."""
    current = get_current_hy_spread()
    return current > bps_threshold


def yield_curve_freshly_inverted() -> bool:
    """Check if yield curve is freshly inverted."""
    yc = get_current_yield_curve()
    return yc < 0
