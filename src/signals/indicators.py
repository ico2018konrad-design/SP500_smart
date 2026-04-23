"""Technical indicators: RSI, MACD, Stochastic, Bollinger Bands, ATR, ADX, VWAP."""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index).

    Args:
        prices: Close price series
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"RSI_{period}"
    return rsi


def calc_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram.

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_line.name = "MACD"
    signal_line.name = "MACD_Signal"
    histogram.name = "MACD_Hist"

    return macd_line, signal_line, histogram


def calc_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator (%K and %D).

    Returns:
        Tuple of (slow_k, slow_d)
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()

    fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    slow_k = fast_k.rolling(smooth_k).mean()
    slow_d = slow_k.rolling(d_period).mean()

    slow_k.name = f"Stoch_K"
    slow_d.name = f"Stoch_D"
    return slow_k, slow_d


def calc_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (upper, middle, lower).

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std

    upper.name = "BB_Upper"
    middle.name = "BB_Middle"
    lower.name = "BB_Lower"
    return upper, middle, lower


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR).

    Args:
        high, low, close: OHLC price series
        period: ATR period (default 14)

    Returns:
        ATR series
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    atr.name = f"ATR_{period}"
    return atr


def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, DI+, DI-.

    Returns:
        Tuple of (adx, plus_di, minus_di)
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index
    )

    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    adx.name = "ADX"
    plus_di.name = "DI_Plus"
    minus_di.name = "DI_Minus"
    return adx, plus_di, minus_di


def calc_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate Volume-Weighted Average Price (VWAP).

    Resets daily for intraday use.
    For daily bars, acts as cumulative VWAP.
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap.name = "VWAP"
    return vwap


def calc_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    ema = prices.ewm(span=period, adjust=False).mean()
    ema.name = f"EMA_{period}"
    return ema


def calc_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    sma = prices.rolling(period).mean()
    sma.name = f"SMA_{period}"
    return sma


def calc_atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate ATR as percentage of current price."""
    atr = calc_atr(high, low, close, period)
    atr_pct = atr / close
    atr_pct.name = f"ATR_Pct_{period}"
    return atr_pct


def is_rsi_crossing_up(rsi: pd.Series, threshold: float = 40.0, lookback: int = 3) -> bool:
    """Check if RSI crossed above threshold recently."""
    recent = rsi.dropna().tail(lookback + 1)
    if len(recent) < 2:
        return False
    for i in range(1, len(recent)):
        if recent.iloc[i - 1] < threshold <= recent.iloc[i]:
            return True
    return False


def is_rsi_crossing_down(rsi: pd.Series, threshold: float = 68.0, lookback: int = 3) -> bool:
    """Check if RSI crossed below threshold recently."""
    recent = rsi.dropna().tail(lookback + 1)
    if len(recent) < 2:
        return False
    for i in range(1, len(recent)):
        if recent.iloc[i - 1] >= threshold > recent.iloc[i]:
            return True
    return False


def is_price_touching_ema(
    close: pd.Series,
    ema: pd.Series,
    tolerance_pct: float = 0.005,
) -> bool:
    """Check if price recently touched EMA (within tolerance %)."""
    recent_close = close.dropna().tail(3)
    recent_ema = ema.dropna().tail(3)

    for c, e in zip(recent_close, recent_ema):
        if abs(c - e) / e <= tolerance_pct:
            return True
    return False


def is_macd_hist_rising(histogram: pd.Series, lookback: int = 3) -> bool:
    """Check if MACD histogram is rising (from negative territory)."""
    recent = histogram.dropna().tail(lookback)
    if len(recent) < 2:
        return False
    return bool(recent.iloc[-1] > recent.iloc[-2])


def is_price_bb_lower(
    close: pd.Series,
    lower_band: pd.Series,
    lookback: int = 3,
) -> bool:
    """Check if price touched or crossed below lower Bollinger Band recently."""
    recent_close = close.dropna().tail(lookback)
    recent_lower = lower_band.dropna().tail(lookback)

    for c, bb in zip(recent_close, recent_lower):
        if c <= bb * 1.002:  # slight tolerance
            return True
    return False


def is_volume_elevated(volume: pd.Series, multiplier: float = 1.3, lookback: int = 20) -> bool:
    """Check if current volume > multiplier × average of last N bars."""
    if len(volume) < lookback + 1:
        return False
    avg_vol = volume.dropna().tail(lookback + 1).iloc[:-1].mean()
    current_vol = volume.dropna().iloc[-1]
    return bool(current_vol > avg_vol * multiplier)


def is_price_above_vwap(close: pd.Series, vwap: pd.Series) -> bool:
    """Check if latest price is above VWAP."""
    if len(close) == 0 or len(vwap) == 0:
        return True
    return bool(close.dropna().iloc[-1] > vwap.dropna().iloc[-1])
