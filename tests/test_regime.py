"""Tests for regime detection."""
import numpy as np
import pandas as pd
import pytest

from src.regime.regime_types import Regime, score_to_regime, REGIME_LEVERAGE
from src.regime.detector import RegimeDetector
from src.regime.valuation_guard import ValuationGuard


def make_spy_data(n: int = 300, start_price: float = 400.0, trend: float = 0.0003) -> pd.DataFrame:
    """Create synthetic SPY OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = start_price * np.exp(
        np.cumsum(trend + 0.01 * np.random.randn(n))
    )
    return pd.DataFrame({
        "Open": prices * 0.999,
        "High": prices * 1.005,
        "Low": prices * 0.995,
        "Close": prices,
        "Volume": np.random.randint(50_000_000, 200_000_000, n),
    }, index=dates)


def make_vix_data(n: int = 300, vix_level: float = 18.0) -> pd.DataFrame:
    """Create synthetic VIX data."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    vix = np.abs(vix_level + np.random.randn(n) * 2)
    return pd.DataFrame({
        "Open": vix,
        "High": vix * 1.02,
        "Low": vix * 0.98,
        "Close": vix,
        "Volume": np.ones(n) * 0,
    }, index=dates)


class TestScoreToRegime:
    def test_strong_bull(self):
        assert score_to_regime(10) == Regime.STRONG_BULL
        assert score_to_regime(11) == Regime.STRONG_BULL

    def test_bull(self):
        assert score_to_regime(8) == Regime.BULL
        assert score_to_regime(9) == Regime.BULL

    def test_chop(self):
        assert score_to_regime(6) == Regime.CHOP
        assert score_to_regime(7) == Regime.CHOP

    def test_caution(self):
        assert score_to_regime(4) == Regime.CAUTION
        assert score_to_regime(5) == Regime.CAUTION

    def test_bear(self):
        assert score_to_regime(0) == Regime.BEAR
        assert score_to_regime(3) == Regime.BEAR

    def test_leverage_decreases_with_score(self):
        leverages = [REGIME_LEVERAGE[score_to_regime(s)] for s in [11, 9, 7, 5, 2]]
        for i in range(len(leverages) - 1):
            assert leverages[i] >= leverages[i + 1]


class TestRegimeDetector:
    def test_bull_market(self):
        """Strong uptrending market should score high."""
        spy = make_spy_data(300, trend=0.001)  # strong uptrend
        vix = make_vix_data(300, vix_level=14.0)  # low VIX
        detector = RegimeDetector()
        result = detector.detect(spy, vix, breadth_pct=0.70, hy_spread_bps=300.0)
        assert result.score >= 5
        assert result.regime in (Regime.STRONG_BULL, Regime.BULL, Regime.CHOP)

    def test_bear_market(self):
        """Downtrending market with high VIX should score low."""
        spy = make_spy_data(300, trend=-0.001)  # downtrend
        vix = make_vix_data(300, vix_level=35.0)  # high VIX
        detector = RegimeDetector()
        result = detector.detect(spy, vix, breadth_pct=0.30, hy_spread_bps=600.0)
        assert result.score <= 6
        assert result.regime in (Regime.BEAR, Regime.CAUTION, Regime.CHOP)

    def test_result_has_all_fields(self):
        spy = make_spy_data(250)
        vix = make_vix_data(250)
        detector = RegimeDetector()
        result = detector.detect(spy, vix)
        assert 0 <= result.score <= 11
        assert result.regime is not None
        assert result.max_leverage >= 0
        assert result.risk_pct > 0

    def test_insufficient_data_returns_result(self):
        """Should not crash with minimal data."""
        spy = make_spy_data(50)  # not enough for SMA200
        vix = make_vix_data(50)
        detector = RegimeDetector()
        result = detector.detect(spy, vix)
        assert result is not None


class TestValuationGuard:
    def test_no_penalty_normal_conditions(self):
        guard = ValuationGuard()
        result = guard.compute(
            base_leverage=2.0,
            spy_price=400.0,
            spy_200sma=400.0,  # at 200 SMA, not overextended
            vix=18.0,          # normal VIX
            cape_ratio=25.0,   # normal CAPE
        )
        assert result.penalty == 1.0
        assert result.final_leverage == 2.0

    def test_overextension_penalty(self):
        guard = ValuationGuard()
        result = guard.compute(
            base_leverage=2.0,
            spy_price=470.0,    # 17.5% above 200 SMA
            spy_200sma=400.0,
            vix=18.0,
            cape_ratio=25.0,
        )
        assert result.overextended is True
        assert result.penalty < 1.0
        assert result.final_leverage < 2.0

    def test_cape_penalty(self):
        guard = ValuationGuard()
        result = guard.compute(
            base_leverage=2.0,
            spy_price=400.0,
            spy_200sma=400.0,
            vix=18.0,
            cape_ratio=40.0,  # elevated CAPE
        )
        assert result.cape_elevated is True
        assert result.penalty < 1.0

    def test_vix_complacency_penalty(self):
        guard = ValuationGuard()
        result = guard.compute(
            base_leverage=2.0,
            spy_price=400.0,
            spy_200sma=400.0,
            vix=10.0,   # very low VIX = complacency
            cape_ratio=25.0,
        )
        assert result.vix_complacent is True
        assert result.penalty < 1.0

    def test_multiple_penalties_compound(self):
        guard = ValuationGuard()
        result = guard.compute(
            base_leverage=2.0,
            spy_price=470.0,   # overextended
            spy_200sma=400.0,
            vix=10.0,          # complacent
            cape_ratio=40.0,   # high CAPE
        )
        # Should have multiple penalties applied
        assert result.penalty < 0.80
        assert result.final_leverage < 1.5
