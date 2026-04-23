"""Market breadth indicators.

Calculates:
- % of S&P 500 stocks above 50 SMA
- Advance/Decline line approximation
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Sample of S&P 500 component tickers used for breadth approximation
# Using sector ETFs and major components as proxies for full breadth
SP500_SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"
]

# Representative large-cap sample (30 Dow-like components)
SP500_SAMPLE_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "LLY",
    "MRK", "PEP", "KO", "BAC", "ABBV", "WMT", "COST", "MCD",
    "DIS", "CSCO", "TMO", "ABT", "CRM"
]


def get_pct_above_sma(
    period: int = 50,
    sample_tickers: Optional[list] = None,
    lookback_days: int = 300,
) -> float:
    """Calculate % of S&P 500 stocks above N-day SMA.

    Args:
        period: SMA period (default 50)
        sample_tickers: List of tickers to use (defaults to SP500_SAMPLE_TICKERS)
        lookback_days: Days of history to download

    Returns:
        Float 0.0-1.0 representing % above SMA (e.g., 0.65 = 65%)
    """
    if sample_tickers is None:
        sample_tickers = SP500_SAMPLE_TICKERS

    import datetime
    end = datetime.datetime.now().strftime("%Y-%m-%d")
    start = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        data = yf.download(
            sample_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            logger.warning("No breadth data available")
            return 0.55

        # Get Close prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]] if "Close" in data else data

        above_count = 0
        total_count = 0

        for ticker in sample_tickers:
            try:
                if ticker not in close.columns:
                    continue
                prices = close[ticker].dropna()
                if len(prices) < period:
                    continue
                sma = prices.rolling(period).mean()
                # Check most recent value
                last_price = prices.iloc[-1]
                last_sma = sma.iloc[-1]
                if not pd.isna(last_sma) and not pd.isna(last_price):
                    total_count += 1
                    if last_price > last_sma:
                        above_count += 1
            except Exception:
                continue

        if total_count == 0:
            return 0.55

        pct = above_count / total_count
        logger.info(
            "Breadth: %d/%d stocks above %d-SMA = %.1f%%",
            above_count, total_count, period, pct * 100
        )
        return pct

    except Exception as exc:
        logger.error("Failed to calculate breadth: %s", exc)
        return 0.55


def get_breadth_series(
    period: int = 50,
    sample_tickers: Optional[list] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Calculate historical breadth series using sector ETFs as proxy.

    This is faster than downloading all 500 stocks.
    Uses sector ETF performance relative to their SMAs.
    """
    if sample_tickers is None:
        sample_tickers = SP500_SECTOR_ETFS

    try:
        data = yf.download(
            sample_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data

        result_dates = []
        result_values = []

        # Get common dates
        common_dates = close.dropna(how="all").index

        for date in common_dates:
            above = 0
            total = 0
            for ticker in sample_tickers:
                if ticker not in close.columns:
                    continue
                prices = close[ticker][:date].dropna()
                if len(prices) < period:
                    continue
                sma_val = prices.rolling(period).mean().iloc[-1]
                last_val = prices.iloc[-1]
                if not pd.isna(sma_val) and not pd.isna(last_val):
                    total += 1
                    if last_val > sma_val:
                        above += 1
            if total > 0:
                result_dates.append(date)
                result_values.append(above / total)

        series = pd.Series(result_values, index=result_dates, name="breadth_pct_above_50sma")
        return series

    except Exception as exc:
        logger.error("Failed to calculate breadth series: %s", exc)
        return pd.Series(dtype=float)


def get_advance_decline_line(
    start: str = "2005-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Approximate A/D line using SPY vs equal-weight RSP.

    When RSP (equal-weight S&P 500) outperforms SPY, breadth is improving.
    Returns normalized A/D proxy: RSP/SPY ratio.
    """
    try:
        data = yf.download(
            ["SPY", "RSP"],
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            return pd.Series(dtype=float)

        spy = close["SPY"].dropna()
        rsp = close["RSP"].dropna() if "RSP" in close.columns else spy.copy()

        # Normalize
        spy_norm = spy / spy.iloc[0]
        rsp_norm = rsp / rsp.iloc[0]

        ad_ratio = rsp_norm / spy_norm
        ad_ratio.name = "ad_line_proxy"

        return ad_ratio

    except Exception as exc:
        logger.error("Failed to calculate A/D line: %s", exc)
        return pd.Series(dtype=float)


def is_ad_line_rising(lookback: int = 5) -> bool:
    """Check if A/D line is rising over last N days."""
    ad = get_advance_decline_line()
    if len(ad) < lookback + 1:
        return True  # assume positive if insufficient data

    recent = ad.dropna().tail(lookback + 1)
    return float(recent.iloc[-1]) > float(recent.iloc[0])
