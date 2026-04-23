"""Yahoo Finance data loader for SPY, VIX, UPRO, SH, SPXS."""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_ohlcv(
    ticker: str,
    start: str = "2005-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
        interval: Data interval ('1d', '1h', '5m', etc.)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    logger.info("Loading %s from %s to %s (interval=%s)", ticker, start, end, interval)
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval,
                           auto_adjust=True, progress=False)
        if data.empty:
            logger.warning("No data returned for %s", ticker)
            return pd.DataFrame()
        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index = pd.to_datetime(data.index)
        data.index.name = "Date"
        logger.info("Loaded %d rows for %s", len(data), ticker)
        return data
    except Exception as exc:
        logger.error("Failed to load %s: %s", ticker, exc)
        return pd.DataFrame()


def load_spy(start: str = "2005-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load SPY OHLCV data."""
    return load_ohlcv("SPY", start=start, end=end)


def load_vix(start: str = "2005-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load VIX index data (^VIX)."""
    return load_ohlcv("^VIX", start=start, end=end)


def load_upro(start: str = "2011-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load UPRO (3x S&P 500 ETF) data. Available from 2011."""
    return load_ohlcv("UPRO", start=start, end=end)


def load_sh(start: str = "2006-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load SH (inverse S&P 500 ETF) data."""
    return load_ohlcv("SH", start=start, end=end)


def load_spxs(start: str = "2008-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load SPXS (3x inverse S&P 500 ETF) data."""
    return load_ohlcv("SPXS", start=start, end=end)


def load_vxx(start: str = "2018-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load VXX (VIX short-term futures ETN) data."""
    return load_ohlcv("VXX", start=start, end=end)


def load_intraday(
    ticker: str,
    days_back: int = 30,
    interval: str = "5m",
) -> pd.DataFrame:
    """Load intraday data (limited history for short intervals)."""
    end = datetime.now()
    start = end - timedelta(days=days_back)
    return load_ohlcv(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
    )


def load_multiple(
    tickers: list,
    start: str = "2005-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> dict:
    """Load multiple tickers at once.

    Returns dict of {ticker: DataFrame}.
    """
    result = {}
    for ticker in tickers:
        result[ticker] = load_ohlcv(ticker, start=start, end=end, interval=interval)
    return result


def get_spy_intraday_hourly(days_back: int = 30) -> pd.DataFrame:
    """Get SPY hourly data for signal generation."""
    return load_intraday("SPY", days_back=days_back, interval="1h")


def get_spy_intraday_5min(days_back: int = 7) -> pd.DataFrame:
    """Get SPY 5-minute data for VWAP calculation."""
    return load_intraday("SPY", days_back=days_back, interval="5m")
