"""Integration test for backtest pipeline.

Runs a short backtest (synthetic data) verifying the full strategy stack
integrates correctly:
- LongSignalGenerator, ShortSignalGenerator
- AntiMartingaleScaler, PositionManager
- BaselineHedge, CircuitBreakers
- ValuationGuard, RegimeDetector
- capital accounting (no phantom profits)

Should complete in < 30 seconds.
"""
import json
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch


def make_synthetic_spy(n_days: int = 400, start_price: float = 400.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic SPY-like OHLCV data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2022-01-01", periods=n_days)
    returns = rng.normal(0.0003, 0.012, n_days)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.01, n_days))
    low = close * (1 - rng.uniform(0.001, 0.01, n_days))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    volume = rng.integers(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_synthetic_vix(spy_dates: pd.DatetimeIndex, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic VIX data aligned to SPY dates."""
    rng = np.random.default_rng(seed)
    vix_vals = 15.0 + rng.normal(0, 3, len(spy_dates))
    vix_vals = np.clip(vix_vals, 10, 45)
    return pd.DataFrame({"Close": vix_vals}, index=spy_dates)


class TestBacktestIntegration:
    """Integration tests for the backtest pipeline using synthetic data."""

    def test_backtest_produces_valid_output_structure(self, tmp_path):
        """Backtest returns dict with expected keys."""
        from src.backtest.honest_backtest import run_backtest

        spy = make_synthetic_spy(n_days=400)
        vix = make_synthetic_vix(spy.index)

        with patch("src.backtest.honest_backtest.load_spy", return_value=spy), \
             patch("src.backtest.honest_backtest.load_vix", return_value=vix), \
             patch("src.data.macro_timeseries.load_fred_series", return_value=pd.Series(dtype=float)), \
             patch("src.data.macro_timeseries.get_fred_api_key", return_value=None):

            result = run_backtest(
                start_date="2022-01-01",
                end_date="2023-12-31",
                starting_capital=5000.0,
                verbose=False,
            )

        assert isinstance(result, dict), "run_backtest must return a dict"
        assert "metrics" in result
        assert "equity_series" in result
        assert "trades" in result

    def test_backtest_capital_not_phantom(self, tmp_path):
        """Final equity must not be wildly inflated (no phantom profit bug)."""
        from src.backtest.honest_backtest import run_backtest

        spy = make_synthetic_spy(n_days=400)
        vix = make_synthetic_vix(spy.index)

        with patch("src.backtest.honest_backtest.load_spy", return_value=spy), \
             patch("src.backtest.honest_backtest.load_vix", return_value=vix), \
             patch("src.data.macro_timeseries.load_fred_series", return_value=pd.Series(dtype=float)), \
             patch("src.data.macro_timeseries.get_fred_api_key", return_value=None):

            result = run_backtest(
                start_date="2022-01-01",
                end_date="2023-12-31",
                starting_capital=5000.0,
                verbose=False,
            )

        metrics = result.get("metrics", {})
        start_capital = 5000.0
        end_equity = metrics.get("end_equity", 0)

        # Should be within -80% to +300% of starting capital (realistic range)
        assert end_equity > start_capital * 0.20, f"Too much loss: {end_equity:.0f}"
        assert end_equity < start_capital * 4.0, (
            f"Unrealistically high equity {end_equity:.0f} — "
            f"likely phantom profit bug still present"
        )

    def test_backtest_equity_curve_length(self):
        """Equity curve should have one point per bar after warmup."""
        from src.backtest.honest_backtest import run_backtest

        spy = make_synthetic_spy(n_days=400)
        vix = make_synthetic_vix(spy.index)

        with patch("src.backtest.honest_backtest.load_spy", return_value=spy), \
             patch("src.backtest.honest_backtest.load_vix", return_value=vix), \
             patch("src.data.macro_timeseries.load_fred_series", return_value=pd.Series(dtype=float)), \
             patch("src.data.macro_timeseries.get_fred_api_key", return_value=None):

            result = run_backtest(
                start_date="2022-01-01",
                end_date="2023-12-31",
                starting_capital=5000.0,
                verbose=False,
            )

        eq = result["equity_series"]
        # warmup = 200, so we get 400-200+1 = 201 bars (plus initial point)
        assert len(eq) >= 100, f"Equity curve too short: {len(eq)} points"
        assert len(eq) <= len(spy) + 1, "Equity curve too long"

    def test_backtest_metrics_sane(self):
        """Metrics should be in realistic range."""
        from src.backtest.honest_backtest import run_backtest

        spy = make_synthetic_spy(n_days=400)
        vix = make_synthetic_vix(spy.index)

        with patch("src.backtest.honest_backtest.load_spy", return_value=spy), \
             patch("src.backtest.honest_backtest.load_vix", return_value=vix), \
             patch("src.data.macro_timeseries.load_fred_series", return_value=pd.Series(dtype=float)), \
             patch("src.data.macro_timeseries.get_fred_api_key", return_value=None):

            result = run_backtest(
                start_date="2022-01-01",
                end_date="2023-12-31",
                starting_capital=5000.0,
                verbose=False,
            )

        metrics = result.get("metrics", {})
        # Sharpe ratio should not be insanely high
        assert metrics.get("sharpe_ratio", 0) < 20.0, "Sharpe ratio suspiciously high"
        # Max drawdown should be negative (or zero)
        assert metrics.get("max_drawdown", 0) <= 0.01, "Max drawdown should be non-positive"
        # Win rate between 0 and 1
        wr = metrics.get("win_rate", 0.5)
        assert 0.0 <= wr <= 1.0, f"Win rate out of range: {wr}"


class TestPaperTraderEquityMath:
    """Verify paper trader equity math at the integration level."""

    def test_flat_round_trip_no_growth(self):
        """Open and close at same price → no capital growth."""
        from src.execution.paper_trader import PaperTrader
        from src.signals.signal_types import Signal, SignalDirection

        pt = PaperTrader(starting_capital=10_000.0)
        initial = pt.capital

        price = 100.0
        pt.get_current_price = lambda sym: price

        signal = Signal(
            direction=SignalDirection.LONG,
            symbol="SPY",
            entry_price=price,
            stop_price=price * 0.985,
            target1=price * 1.02,
            target2=price * 1.045,
            target3=price * 1.08,
            timestamp=datetime.now(),
            regime_score=9,
            regime="BULL",
        )

        pos = pt.execute_signal(signal, allocated_capital=10_000.0)
        assert pos is not None

        pt.close_position(pos.position_id, reason="test", current_prices={"SPY": price})

        # After flat round-trip, capital should be at most initial (minus costs)
        assert pt.capital <= initial, f"Capital grew on flat round-trip: {pt.capital:.2f} > {initial:.2f}"


class TestMacroTimeSeries:
    """Verify MacroTimeSeries fallback behavior."""

    def test_returns_defaults_without_fred_key(self):
        """Without FRED key, returns sensible defaults."""
        from src.data.macro_timeseries import MacroTimeSeries

        with patch("src.data.macro_timeseries.get_fred_api_key", return_value=None):
            macro = MacroTimeSeries(start="2005-01-01")
            hy = macro.get_hy_spread_on(pd.Timestamp("2020-03-15"))
            yc = macro.get_yield_curve_on(pd.Timestamp("2020-03-15"))
            breadth = macro.get_breadth_on(pd.Timestamp("2020-03-15"), make_synthetic_spy(300))

        assert hy > 0, "HY spread should be positive"
        assert isinstance(yc, float)
        assert 0.0 <= breadth <= 1.0, f"Breadth out of range: {breadth}"

    def test_breadth_proxy_varies_with_price(self):
        """Breadth proxy should be lower when price is below its 50 SMA."""
        from src.data.macro_timeseries import MacroTimeSeries

        with patch("src.data.macro_timeseries.get_fred_api_key", return_value=None):
            macro = MacroTimeSeries()

            # Declining market: price well below SMA
            dates = pd.bdate_range("2022-01-01", periods=100)
            prices_declining = pd.Series(
                [400.0 - i * 1.5 for i in range(100)], index=dates
            )
            df_declining = pd.DataFrame({"Close": prices_declining, "High": prices_declining * 1.005, "Low": prices_declining * 0.995, "Volume": 1e8})

            # Rising market: price well above SMA
            prices_rising = pd.Series(
                [300.0 + i * 1.5 for i in range(100)], index=dates
            )
            df_rising = pd.DataFrame({"Close": prices_rising, "High": prices_rising * 1.005, "Low": prices_rising * 0.995, "Volume": 1e8})

            b_declining = macro.get_breadth_on(dates[-1], df_declining)
            b_rising = macro.get_breadth_on(dates[-1], df_rising)

        assert b_declining < b_rising, "Breadth should be lower in declining market"
