"""Tests for backtest components."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.performance_metrics import (
    sharpe_ratio, max_drawdown, cagr, profit_factor,
    win_rate, calmar_ratio, compute_all_metrics,
)


def make_equity(returns_list):
    """Build equity series from returns."""
    dates = pd.date_range("2020-01-01", periods=len(returns_list) + 1, freq="B")
    equity = [5000.0]
    for r in returns_list:
        equity.append(equity[-1] * (1 + r))
    return pd.Series(equity, index=dates)


class TestPerformanceMetrics:
    def test_sharpe_positive_returns(self):
        equity = make_equity([0.002] * 252)  # consistent gains
        returns = equity.pct_change().dropna()
        sr = sharpe_ratio(returns, risk_free_rate=0.05)
        assert sr > 0

    def test_sharpe_negative_returns(self):
        equity = make_equity([-0.002] * 252)
        returns = equity.pct_change().dropna()
        sr = sharpe_ratio(returns, risk_free_rate=0.05)
        assert sr < 0

    def test_max_drawdown_flat(self):
        equity = make_equity([0.001] * 100)
        mdd = max_drawdown(equity)
        assert mdd >= -0.01  # minimal drawdown in uptrend

    def test_max_drawdown_crash(self):
        # 50% crash then recovery
        returns = [-0.01] * 50 + [0.01] * 100
        equity = make_equity(returns)
        mdd = max_drawdown(equity)
        assert mdd < -0.30  # significant drawdown

    def test_cagr_positive(self):
        equity = make_equity([0.001] * 252)
        c = cagr(equity)
        assert c > 0

    def test_profit_factor_all_wins(self):
        returns = pd.Series([0.01] * 50)
        pf = profit_factor(returns)
        assert pf == float("inf")

    def test_profit_factor_all_losses(self):
        returns = pd.Series([-0.01] * 50)
        pf = profit_factor(returns)
        assert pf == 0.0

    def test_win_rate(self):
        returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01])
        wr = win_rate(returns)
        assert abs(wr - 0.60) < 0.01  # 3 wins out of 5

    def test_compute_all_metrics_structure(self):
        equity = make_equity([0.001, -0.002, 0.003] * 84)
        metrics = compute_all_metrics(equity)
        required_keys = ["cagr", "sharpe_ratio", "max_drawdown", "win_rate",
                         "profit_factor", "calmar_ratio", "total_return"]
        for key in required_keys:
            assert key in metrics
