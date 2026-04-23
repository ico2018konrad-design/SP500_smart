"""Rolling Out-of-Sample Validation — 16-window consecutive OOS testing.

Each window: 1-year out-of-sample test, sliding 1 year at a time.
No parameter optimization is performed on the training period (honest OOS naming).

To add true walk-forward with parameter optimization, see TODO in code.
Reports Sharpe/MDD per window.

Note: Previously named run_walk_forward(). Renamed to run_rolling_oos_validation()
to reflect that no in-sample parameter tuning occurs — these are purely
consecutive 1-year out-of-sample tests.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from src.backtest.honest_backtest import run_backtest
from src.backtest.performance_metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_rolling_oos_validation(
    full_start: str = "2005-01-01",
    full_end: str = "2025-12-31",
    train_years: int = 3,
    test_years: int = 1,
    starting_capital: float = 5000.0,
) -> dict:
    """Run 16-window rolling out-of-sample validation.

    Runs consecutive 1-year OOS tests without parameter optimization.
    The 'train_years' parameter is recorded for documentation purposes only —
    no in-sample optimization is performed (honest OOS).

    TODO (walk-forward v2): Add grid-search over min_regime_score, stop_pct,
    profit_threshold on the train period, then apply best params to test period.

    Args:
        full_start: Start of full data range
        full_end: End of full data range
        train_years: Training window size in years (recorded but not used for optimization)
        test_years: Test window size in years
        starting_capital: Starting capital per window

    Returns:
        Dict with results per window + aggregate stats
    """
    logger.info(
        "Rolling OOS validation: %d-year test windows (no train-period optimization)",
        test_years
    )

    windows = []
    start_year = int(full_start[:4])
    end_year = int(full_end[:4])
    num_windows = 0
    current_year = start_year

    # Generate windows
    while current_year + train_years + test_years <= end_year + 1:
        train_start = f"{current_year}-01-01"
        train_end = f"{current_year + train_years - 1}-12-31"
        test_start = f"{current_year + train_years}-01-01"
        test_end = f"{current_year + train_years + test_years - 1}-12-31"
        windows.append({
            "window": num_windows + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        current_year += 1
        num_windows += 1
        if num_windows >= 16:
            break

    logger.info("Running %d walk-forward windows...", len(windows))

    results = []
    for w in windows:
        logger.info(
            "Window %d: Train %s→%s | Test %s→%s",
            w["window"], w["train_start"], w["train_end"],
            w["test_start"], w["test_end"]
        )

        # Run backtest on test period (out-of-sample)
        result = run_backtest(
            start_date=w["test_start"],
            end_date=w["test_end"],
            starting_capital=starting_capital,
            verbose=False,
        )

        if result and result.get("metrics"):
            m = result["metrics"]
            window_result = {
                "window": w["window"],
                "train_period": f"{w['train_start']} to {w['train_end']}",
                "test_period": f"{w['test_start']} to {w['test_end']}",
                "sharpe_ratio": m.get("sharpe_ratio", 0),
                "max_drawdown": m.get("max_drawdown", 0),
                "cagr": m.get("cagr", 0),
                "total_return": m.get("total_return", 0),
                "win_rate": m.get("win_rate", 0),
                "num_trades": m.get("num_trades", 0),
            }
            results.append(window_result)

            logger.info(
                "  → Sharpe: %.2f | MDD: %.1f%% | CAGR: %.1f%% | Trades: %d",
                window_result["sharpe_ratio"],
                window_result["max_drawdown"] * 100,
                window_result["cagr"] * 100,
                window_result["num_trades"],
            )
        else:
            logger.warning("  → No data for window %d", w["window"])

    # Aggregate statistics
    if results:
        sharpes = [r["sharpe_ratio"] for r in results]
        mdds = [r["max_drawdown"] for r in results]
        cagrs = [r["cagr"] for r in results]

        aggregate = {
            "num_windows": len(results),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_min": float(np.min(sharpes)),
            "mdd_mean": float(np.mean(mdds)),
            "mdd_worst": float(np.min(mdds)),
            "cagr_mean": float(np.mean(cagrs)),
            "positive_sharpe_pct": float(sum(1 for s in sharpes if s > 0) / len(sharpes)),
            "windows_above_1_sharpe": sum(1 for s in sharpes if s > 1.0),
        }
    else:
        aggregate = {}

    output = {
        "config": {
            "full_start": full_start,
            "full_end": full_end,
            "train_years": train_years,
            "test_years": test_years,
        },
        "windows": results,
        "aggregate": aggregate,
    }

    print("\n" + "=" * 60)
    print("ROLLING OOS VALIDATION RESULTS")
    print("(No parameter optimization — pure out-of-sample)")
    print("=" * 60)
    for r in results:
        print(
            f"W{r['window']:02d} {r['test_period']}: "
            f"Sharpe={r['sharpe_ratio']:+.2f} | "
            f"MDD={r['max_drawdown']:.1%} | "
            f"CAGR={r['cagr']:.1%}"
        )
    if aggregate:
        print("─" * 60)
        print(f"Avg Sharpe: {aggregate['sharpe_mean']:.2f} ± {aggregate['sharpe_std']:.2f}")
        print(f"Worst MDD:  {aggregate['mdd_worst']:.1%}")
        print(f"Avg CAGR:   {aggregate['cagr_mean']:.1%}")
        print(f"Windows > Sharpe 1.0: {aggregate['windows_above_1_sharpe']}/{len(results)}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    with open("results/walk_forward.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to results/walk_forward.json")

    return output


# Backward-compatible alias
def run_walk_forward(**kwargs) -> dict:
    """Backward-compatible alias for run_rolling_oos_validation().

    Deprecated: use run_rolling_oos_validation() directly.
    """
    return run_rolling_oos_validation(**kwargs)


if __name__ == "__main__":
    run_rolling_oos_validation()
