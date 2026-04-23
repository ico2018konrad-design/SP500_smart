"""Yahoo Finance data loader for historical OHLCV.

Loads SPY, VIX, UPRO, SH, SPXS using yfinance.
Caches data locally to reduce API calls.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = ".data_cache"


def _cache_path(symbol: str, start: str, end: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}_{start}_{end}.parquet")


def load_ohlcv(
    symbol: str,
    start: str = "2005-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data for a symbol from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g., 'SPY', 'VIX', 'UPRO')
        start: Start date string 'YYYY-MM-DD'
        end: End date string (defaults to today)
        use_cache: Cache results to disk

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    cache_file = _cache_path(symbol, start, end)

    # Try cache first (within 1 day)
    if use_cache and os.path.exists(cache_file):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        if file_age < timedelta(hours=4):
            try:
                df = pd.read_parquet(cache_file)
                logger.debug("Loaded %s from cache (%d rows)", symbol, len(df))
                return df
            except Exception:
                pass

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            logger.warning("No data returned for %s", symbol)
            return pd.DataFrame()

        # Ensure standard column names
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

        if use_cache and not df.empty:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass

        logger.info("Loaded %s: %d bars (%s to %s)", symbol, len(df), start, end)
        return df

    except Exception as exc:
        logger.error("Failed to load %s: %s", symbol, exc)
        return pd.DataFrame()


def load_spy(
    start: str = "2005-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load SPY (S&P 500 ETF) data."""
    return load_ohlcv("SPY", start, end, use_cache)


def load_vix(
    start: str = "2005-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load VIX (CBOE Volatility Index) data."""
    df = load_ohlcv("^VIX", start, end, use_cache)
    if df.empty:
        # Fallback symbol
        df = load_ohlcv("VIX", start, end, use_cache)
    return df


def load_upro(start: str = "2009-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load UPRO (ProShares UltraPro S&P500 3x) data."""
    return load_ohlcv("UPRO", start, end)


def load_sh(start: str = "2006-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load SH (ProShares Short S&P500 -1x) data."""
    return load_ohlcv("SH", start, end)


def load_spxs(start: str = "2008-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load SPXS (Direxion Daily S&P 500 Bear 3x) data."""
    return load_ohlcv("SPXS", start, end)


def load_vxx(start: str = "2009-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load VXX (iPath Series B S&P 500 VIX) data. Full mode only."""
    return load_ohlcv("VXX", start, end)


def get_current_price(symbol: str) -> Optional[float]:
    """Get the latest closing price for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.error("Failed to get current price for %s: %s", symbol, exc)
    return None
