"""Performance metrics: Sharpe, Sortino, Calmar, MDD, Profit Factor."""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def calc_returns(equity_series: pd.Series) -> pd.Series:
    """Calculate daily returns from equity series."""
    return equity_series.pct_change().dropna()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: Daily return series
        risk_free_rate: Annual risk-free rate (default 5%)
        periods_per_year: Number of trading periods per year
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    return float(sharpe)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Calculate annualized Sortino Ratio (uses only downside deviation)."""
    if returns.empty:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    downside = excess_returns[excess_returns < 0]

    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess_returns.mean() > 0 else 0.0

    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside.std()
    return float(sortino)


def max_drawdown(equity_series: pd.Series) -> float:
    """Calculate Maximum Drawdown.

    Returns negative value (e.g., -0.15 = -15% drawdown).
    """
    if equity_series.empty:
        return 0.0

    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    return float(drawdown.min())


def calmar_ratio(
    equity_series: pd.Series,
    returns: Optional[pd.Series] = None,
) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
    if equity_series.empty or len(equity_series) < 2:
        return 0.0

    mdd = abs(max_drawdown(equity_series))
    if mdd == 0:
        return float("inf")

    years = len(equity_series) / TRADING_DAYS
    if years <= 0:
        return 0.0

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / years) - 1

    return float(cagr / mdd)


def cagr(equity_series: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate."""
    if equity_series.empty or len(equity_series) < 2:
        return 0.0
    years = len(equity_series) / TRADING_DAYS
    if years <= 0:
        return 0.0
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    return float((1 + total_return) ** (1 / years) - 1)


def profit_factor(returns: pd.Series) -> float:
    """Calculate Profit Factor (gross profit / gross loss)."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 1.0
    return float(wins / losses)


def win_rate(returns: pd.Series) -> float:
    """Calculate win rate (% of positive return periods)."""
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_all_metrics(equity_series: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Compute all performance metrics.

    Args:
        equity_series: Daily equity series (indexed by date)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dict with all metrics
    """
    returns = calc_returns(equity_series)

    metrics = {
        "cagr": cagr(equity_series),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "calmar_ratio": calmar_ratio(equity_series),
        "max_drawdown": max_drawdown(equity_series),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "total_return": float(equity_series.iloc[-1] / equity_series.iloc[0] - 1) if len(equity_series) > 1 else 0.0,
        "volatility_annual": float(returns.std() * np.sqrt(TRADING_DAYS)) if not returns.empty else 0.0,
        "num_periods": len(returns),
        "start_equity": float(equity_series.iloc[0]) if not equity_series.empty else 0.0,
        "end_equity": float(equity_series.iloc[-1]) if not equity_series.empty else 0.0,
    }

    logger.info(
        "Performance: CAGR=%.1f%%, Sharpe=%.2f, MDD=%.1f%%, WinRate=%.0f%%",
        metrics["cagr"] * 100,
        metrics["sharpe_ratio"],
        metrics["max_drawdown"] * 100,
        metrics["win_rate"] * 100,
    )

    return metrics
