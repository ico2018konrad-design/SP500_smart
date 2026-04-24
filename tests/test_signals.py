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


class TestLongSignalV13:
    """Tests for v1.3 enhancements: trigger #6, adaptive stops, regime normalization."""

    def test_trigger6_ema50_above_ema200_fires_in_bull(self):
        """Trigger #6 (ema50 > ema200) fires in a sustained uptrend."""
        gen = LongSignalGenerator()
        data = make_ohlcv(n=300, trend=0.001)  # clear uptrend → EMA50 > EMA200
        count, triggers = gen._trend_follow_triggers(
            data["Close"], data["High"], data["Low"], data["Volume"], data["Open"]
        )
        assert "ema50_above_ema200_uptrend" in triggers, (
            f"Trigger #6 should fire in a 300-bar uptrend. Got: {triggers}"
        )
        assert count >= 1

    def test_min_triggers_1_produces_signal_in_bull(self):
        """With min_triggers=1 (relaxed), a bull-market day always produces a signal."""
        gen = LongSignalGenerator()
        data = make_ohlcv(n=300, trend=0.001)
        signal = gen.generate(
            symbol="SPY",
            daily_data=data,
            regime_score=10,
            regime="STRONG_BULL",
            spy_above_200sma=True,
            breadth_rising=True,
        )
        assert signal is not None, "STRONG_BULL market should produce a signal with relaxed triggers"

    def test_adaptive_stop_daily_wider_than_intraday(self):
        """Daily stop (2.5%) must be wider than intraday stop (1.5%)."""
        gen = LongSignalGenerator()
        daily_stop, *_ = gen._stop_pct_for_timeframe("daily")
        intra_stop, *_ = gen._stop_pct_for_timeframe("intraday")
        assert daily_stop > intra_stop, "Daily stop should be wider than intraday stop"
        assert daily_stop == pytest.approx(0.025)
        assert intra_stop == pytest.approx(gen.stop_pct)

    def test_adaptive_targets_daily(self):
        """Daily timeframe uses T1=5%, T2=9%, T3=15%."""
        gen = LongSignalGenerator()
        _, t1, t2, t3 = gen._stop_pct_for_timeframe("daily")
        assert t1 == pytest.approx(0.050)
        assert t2 == pytest.approx(0.090)
        assert t3 == pytest.approx(0.150)

    def test_adaptive_stop_timeframe_case_insensitive(self):
        """_stop_pct_for_timeframe must handle case variants."""
        gen = LongSignalGenerator()
        stop_lower, *_ = gen._stop_pct_for_timeframe("daily")
        stop_upper, *_ = gen._stop_pct_for_timeframe("Daily")
        stop_mixed, *_ = gen._stop_pct_for_timeframe("DAILY")
        assert stop_lower == stop_upper == stop_mixed

    def test_regime_normalization_enum(self):
        """generate() accepts Regime enum directly without error."""
        from src.regime.regime_types import Regime
        gen = LongSignalGenerator()
        data = make_ohlcv(n=300, trend=0.001)
        # Should not raise — regime enum is handled gracefully
        signal = gen.generate(
            symbol="SPY",
            daily_data=data,
            regime_score=10,
            regime=Regime.STRONG_BULL,
            spy_above_200sma=True,
            breadth_rising=True,
        )
        assert signal is not None

    def test_regime_normalization_lowercase(self):
        """generate() handles lowercase regime strings."""
        gen = LongSignalGenerator()
        data = make_ohlcv(n=300, trend=0.001)
        signal = gen.generate(
            symbol="SPY",
            daily_data=data,
            regime_score=10,
            regime="strong_bull",
            spy_above_200sma=True,
            breadth_rising=True,
        )
        assert signal is not None, "Lowercase regime string should be normalized and work"

    def test_signal_stop_is_daily_width_on_daily_data(self):
        """Signal generated on daily data must use the wider 2.5% stop."""
        gen = LongSignalGenerator()
        data = make_ohlcv(n=300, trend=0.001)
        signal = gen.generate(
            symbol="SPY",
            daily_data=data,
            regime_score=10,
            regime="STRONG_BULL",
            spy_above_200sma=True,
            breadth_rising=True,
        )
        assert signal is not None
        entry = signal.entry_price
        stop = signal.stop_price
        stop_pct = (entry - stop) / entry
        assert stop_pct == pytest.approx(0.025, abs=0.001), (
            f"Daily signal should have 2.5% stop, got {stop_pct:.2%}"
        )


class TestExitManagerV13:
    """Tests for v1.3 regime-softened exits for core positions."""

    def _make_core_position(self, pm, symbol="UPRO", entry=100.0):
        """Helper: add a core position (scale_number=0)."""
        from src.positions.position_manager import PositionManager
        pos = pm.add_position(symbol, "LONG", entry, 10, entry * 0.90, entry * 1.06, entry * 1.12, entry * 1.20)
        pos.scale_number = 0
        return pos

    def _make_scaled_position(self, pm, symbol="UPRO", entry=100.0):
        """Helper: add a scaled position (scale_number=1)."""
        pos = pm.add_position(symbol, "LONG", entry, 5, entry * 0.90, entry * 1.06, entry * 1.12, entry * 1.20)
        pos.scale_number = 1
        return pos

    def test_core_not_exited_on_bull_to_chop(self):
        """Core positions must NOT exit on BULL→CHOP transition."""
        from src.positions.position_manager import PositionManager
        from src.positions.exit_manager import ExitManager
        from src.regime.regime_types import Regime

        pm = PositionManager()
        exit_mgr = ExitManager()
        core = self._make_core_position(pm)

        exits = exit_mgr.check_exits(
            pm,
            {"UPRO": 100.0},
            {"UPRO": 2.0},
            current_regime=Regime.CHOP,
            previous_regime=Regime.BULL,
            consecutive_bear_days=0,
            current_time=__import__("datetime").datetime(2023, 6, 1),
        )
        exited_ids = {e[0] for e in exits}
        assert core.position_id not in exited_ids, "Core should NOT exit on BULL→CHOP"

    def test_scaled_upro_exits_on_bull_to_chop(self):
        """Scaled UPRO positions SHOULD still exit on BULL→CHOP."""
        from src.positions.position_manager import PositionManager
        from src.positions.exit_manager import ExitManager
        from src.regime.regime_types import Regime

        pm = PositionManager()
        exit_mgr = ExitManager()
        scaled = self._make_scaled_position(pm)

        exits = exit_mgr.check_exits(
            pm,
            {"UPRO": 100.0},
            {"UPRO": 2.0},
            current_regime=Regime.CHOP,
            previous_regime=Regime.BULL,
            consecutive_bear_days=0,
            current_time=__import__("datetime").datetime(2023, 6, 1),
        )
        exited_ids = {e[0] for e in exits}
        assert scaled.position_id in exited_ids, "Scaled UPRO should still exit on BULL→CHOP"

    def test_core_not_exited_on_single_bear_day(self):
        """Core positions must NOT exit on a single BEAR day (consecutive < 2)."""
        from src.positions.position_manager import PositionManager
        from src.positions.exit_manager import ExitManager
        from src.regime.regime_types import Regime

        pm = PositionManager()
        exit_mgr = ExitManager()
        core = self._make_core_position(pm)

        exits = exit_mgr.check_exits(
            pm,
            {"UPRO": 100.0},
            {"UPRO": 2.0},
            current_regime=Regime.BEAR,
            previous_regime=Regime.CHOP,
            consecutive_bear_days=1,
            current_time=__import__("datetime").datetime(2023, 6, 1),
        )
        exited_ids = {e[0] for e in exits}
        assert core.position_id not in exited_ids, "Core should NOT exit after just 1 BEAR day"

    def test_core_exits_after_two_consecutive_bear_days(self):
        """Core positions MUST exit after 2+ consecutive BEAR days."""
        from src.positions.position_manager import PositionManager
        from src.positions.exit_manager import ExitManager
        from src.regime.regime_types import Regime

        pm = PositionManager()
        exit_mgr = ExitManager()
        core = self._make_core_position(pm)

        exits = exit_mgr.check_exits(
            pm,
            {"UPRO": 100.0},
            {"UPRO": 2.0},
            current_regime=Regime.BEAR,
            previous_regime=Regime.CHOP,
            consecutive_bear_days=2,
            current_time=__import__("datetime").datetime(2023, 6, 1),
        )
        exited_ids = {e[0] for e in exits}
        assert core.position_id in exited_ids, "Core MUST exit after 2+ consecutive BEAR days"


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
