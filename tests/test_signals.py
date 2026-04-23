"""Tests for signal generation."""
import numpy as np
import pandas as pd
import pytest

from src.signals.indicators import (
    calc_rsi, calc_macd, calc_stochastic, calc_bollinger_bands,
    calc_atr, is_rsi_crossing_up, is_volume_elevated,
)
from src.signals.long_signals import LongSignalGenerator
from src.signals.short_signals import ShortSignalGenerator


def make_ohlcv(n: int = 100, trend: float = 0.0) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    np.random.seed(123)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 400.0 * np.exp(np.cumsum(trend + 0.01 * np.random.randn(n)))
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.random.randint(50_000_000, 200_000_000, n),
    }, index=dates)


class TestIndicators:
    def test_rsi_range(self):
        data = make_ohlcv(100)
        rsi = calc_rsi(data["Close"])
        valid = rsi.dropna()
        assert all(0 <= v <= 100 for v in valid)

    def test_macd_returns_three_series(self):
        data = make_ohlcv(100)
        macd, signal, hist = calc_macd(data["Close"])
        assert len(macd) == len(data)
        assert len(signal) == len(data)
        assert len(hist) == len(data)

    def test_bollinger_bands_upper_above_lower(self):
        data = make_ohlcv(100)
        upper, mid, lower = calc_bollinger_bands(data["Close"])
        valid_upper = upper.dropna()
        valid_lower = lower.dropna()
        assert all(u > l for u, l in zip(valid_upper, valid_lower))

    def test_atr_positive(self):
        data = make_ohlcv(100)
        atr = calc_atr(data["High"], data["Low"], data["Close"])
        assert all(v > 0 for v in atr.dropna())

    def test_stochastic_range(self):
        data = make_ohlcv(100)
        k, d = calc_stochastic(data["High"], data["Low"], data["Close"])
        valid_k = k.dropna()
        assert all(0 <= v <= 100 for v in valid_k)

    def test_rsi_crossing_up(self):
        # Manually craft crossing up at 40
        rsi = pd.Series([35, 37, 39, 41, 43])
        assert is_rsi_crossing_up(rsi, 40.0, lookback=3) is True

    def test_rsi_not_crossing_up(self):
        rsi = pd.Series([50, 52, 53, 55])
        assert is_rsi_crossing_up(rsi, 40.0) is False

    def test_volume_elevated(self):
        vol = pd.Series([100] * 20 + [160])  # last bar is 1.6x average
        assert is_volume_elevated(vol, multiplier=1.3) is True

    def test_volume_not_elevated(self):
        vol = pd.Series([100] * 21)
        assert is_volume_elevated(vol, multiplier=1.3) is False


class TestLongSignals:
    def test_setup_fails_low_regime(self):
        gen = LongSignalGenerator()
        valid, reasons = gen.check_setup(
            regime_score=5,  # below 8
            spy_above_200sma=True,
            atr_pct=0.01,
        )
        assert not valid
        assert any("regime_score" in r for r in reasons)

    def test_setup_fails_spy_below_200sma(self):
        gen = LongSignalGenerator()
        valid, reasons = gen.check_setup(
            regime_score=9,
            spy_above_200sma=False,  # below 200 SMA
            atr_pct=0.01,
        )
        assert not valid

    def test_setup_fails_high_atr(self):
        gen = LongSignalGenerator()
        valid, reasons = gen.check_setup(
            regime_score=9,
            spy_above_200sma=True,
            atr_pct=0.04,  # above 3% threshold
        )
        assert not valid

    def test_setup_passes_all_conditions(self):
        gen = LongSignalGenerator()
        valid, reasons = gen.check_setup(
            regime_score=9,
            spy_above_200sma=True,
            atr_pct=0.015,
            has_major_event=False,
            has_short_position=False,
        )
        assert valid
        assert len(reasons) == 0

    def test_setup_fails_major_event(self):
        gen = LongSignalGenerator()
        valid, _ = gen.check_setup(
            regime_score=9,
            spy_above_200sma=True,
            atr_pct=0.015,
            has_major_event=True,
        )
        assert not valid


class TestShortSignals:
    def test_setup_fails_high_regime(self):
        gen = ShortSignalGenerator()
        valid, reasons = gen.check_setup(
            regime_score=8,  # above 5 = no shorting
            vix=25.0,
            vix_rising=True,
            spy_below_50sma=True,
        )
        assert not valid

    def test_setup_passes_bear_conditions(self):
        gen = ShortSignalGenerator()
        valid, reasons = gen.check_setup(
            regime_score=3,  # bear
            vix=25.0,
            vix_rising=True,
            spy_below_50sma=True,
            panic_rebound=False,
        )
        assert valid
