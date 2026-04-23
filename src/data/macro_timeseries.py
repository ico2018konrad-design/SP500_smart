"""Macro time-series helpers for backtest.

Provides date-indexed lookup for:
- HY OAS spread (BAMLH0A0HYM2 from FRED)
- Yield curve T10Y2Y (from FRED)
- Breadth proxy (SPY / SPY_50SMA mapping)

If FRED API key is not available, falls back to constants with a warning.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.fred_macro import load_fred_series, get_fred_api_key

logger = logging.getLogger(__name__)

# FRED series IDs
HY_SERIES = "BAMLH0A0HYM2"   # HY OAS in percent → multiply by 100 for bps
YC_SERIES = "T10Y2Y"          # 10Y-2Y in percent

# Fallback constants (neutral defaults)
_DEFAULT_HY_BPS = 350.0
_DEFAULT_YC = 0.50


class MacroTimeSeries:
    """Cache and serve historical macro data by date.

    Usage:
        macro = MacroTimeSeries(start="2005-01-01")
        hy = macro.get_hy_spread_on(pd.Timestamp("2008-10-15"))  # returns ~2000+ bps
    """

    def __init__(self, start: str = "2000-01-01"):
        self._start = start
        self._hy: Optional[pd.Series] = None
        self._yc: Optional[pd.Series] = None
        self._loaded = False
        self._has_real_data = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if get_fred_api_key():
            hy_raw = load_fred_series(HY_SERIES, start=self._start)
            yc_raw = load_fred_series(YC_SERIES, start=self._start)

            if not hy_raw.empty and not yc_raw.empty:
                # FRED returns % values (e.g. 3.5 means 350 bps for HY)
                self._hy = hy_raw.astype(float).ffill().bfill()
                self._yc = yc_raw.astype(float).ffill().bfill()
                self._has_real_data = True
                logger.info(
                    "MacroTimeSeries loaded: HY %d obs, YC %d obs",
                    len(self._hy), len(self._yc),
                )
            else:
                logger.warning(
                    "FRED data empty — backtest uses constant macro defaults "
                    "(HY=%.0f bps, YC=%.2f). Regime scores during stress periods "
                    "may be inflated.",
                    _DEFAULT_HY_BPS, _DEFAULT_YC,
                )
        else:
            logger.warning(
                "FRED_API_KEY not set — backtest uses constant macro defaults "
                "(HY=%.0f bps, YC=%.2f). Get a free key at fred.stlouisfed.org. "
                "Regime scores during 2008/2020 stress periods will be inflated.",
                _DEFAULT_HY_BPS, _DEFAULT_YC,
            )

        self._loaded = True

    def _lookup(self, series: Optional[pd.Series], date: pd.Timestamp, default: float) -> float:
        """Lookup nearest value in series for a given date."""
        if series is None or series.empty:
            return default
        # Use asof (last observation before or on date)
        val = series.asof(date)
        if pd.isna(val):
            return default
        return float(val)

    def get_hy_spread_on(self, date: pd.Timestamp) -> float:
        """Return HY OAS in basis points for a given date.

        Historical examples:
        - 2007 (normal):  ~280-350 bps
        - 2008 crisis:    ~1900-2000 bps
        - 2020 COVID:     ~1000+ bps
        - 2022 hike cycle: ~500 bps
        """
        self._ensure_loaded()
        pct = self._lookup(self._hy, date, _DEFAULT_HY_BPS / 100.0)
        return pct * 100.0  # convert from % to bps

    def get_yield_curve_on(self, date: pd.Timestamp) -> float:
        """Return T10Y2Y spread in % for a given date.

        Positive = normal curve, negative = inverted (recession signal).
        """
        self._ensure_loaded()
        return self._lookup(self._yc, date, _DEFAULT_YC)

    def is_yield_curve_freshly_inverted(self, date: pd.Timestamp) -> bool:
        """True if yield curve inverted (T10Y2Y < 0) on the given date."""
        self._ensure_loaded()
        return self.get_yield_curve_on(date) < 0.0

    def get_breadth_on(self, date: pd.Timestamp, spy_slice: pd.DataFrame) -> float:
        """Compute breadth proxy from SPY price relative to its 50-day SMA.

        Uses (SPY/SPY_50SMA) mapped to a [0.3, 0.8] breadth range.
        This is a rough proxy — not true % of stocks above 50 SMA,
        but captures the same directional signal historically.

        Args:
            date: Bar date (unused, included for API consistency)
            spy_slice: SPY OHLCV DataFrame up to and including current bar

        Returns:
            breadth_pct: float in [0.0, 1.0]
        """
        if spy_slice is None or len(spy_slice) < 10:
            return 0.55  # neutral default

        close = spy_slice["Close"]
        sma50 = close.rolling(50).mean()

        if pd.isna(sma50.iloc[-1]) or sma50.iloc[-1] == 0:
            return 0.55

        ratio = float(close.iloc[-1]) / float(sma50.iloc[-1])
        # Map ratio [0.80, 1.20] → breadth [0.30, 0.80]
        ratio_clipped = float(np.clip(ratio, 0.80, 1.20))
        breadth = 0.30 + (ratio_clipped - 0.80) / (1.20 - 0.80) * (0.80 - 0.30)
        return float(np.clip(breadth, 0.0, 1.0))

    @property
    def has_real_data(self) -> bool:
        """True if real FRED data was successfully loaded."""
        self._ensure_loaded()
        return self._has_real_data


# Module-level singleton for convenience
_default_macro: Optional[MacroTimeSeries] = None


def _get_macro(start: str = "2000-01-01") -> MacroTimeSeries:
    global _default_macro
    if _default_macro is None:
        _default_macro = MacroTimeSeries(start=start)
    return _default_macro


def get_hy_spread_on(date: pd.Timestamp, start: str = "2000-01-01") -> float:
    """Module-level convenience: HY spread in bps for a date."""
    return _get_macro(start).get_hy_spread_on(date)


def get_yield_curve_on(date: pd.Timestamp, start: str = "2000-01-01") -> float:
    """Module-level convenience: T10Y2Y for a date."""
    return _get_macro(start).get_yield_curve_on(date)


def is_yield_curve_freshly_inverted(date: pd.Timestamp, start: str = "2000-01-01") -> bool:
    """Module-level convenience: True if yield curve inverted."""
    return _get_macro(start).is_yield_curve_freshly_inverted(date)


def get_breadth_on(date: pd.Timestamp, spy_slice: pd.DataFrame, start: str = "2000-01-01") -> float:
    """Module-level convenience: breadth proxy for a date."""
    return _get_macro(start).get_breadth_on(date, spy_slice)
