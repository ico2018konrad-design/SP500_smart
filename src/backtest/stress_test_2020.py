"""Stress Test: COVID-19 Crash (Feb-Apr 2020 — V-shape recovery)."""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.honest_backtest import run_backtest
from src.backtest.performance_metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START = "2019-01-01"
END = "2021-12-31"
CRISIS_START = "2020-02-19"
CRISIS_END = "2020-04-30"


def run_stress_test_2020() -> dict:
    logger.info("Running stress test: COVID-19 Crash (%s to %s)", CRISIS_START, CRISIS_END)

    result = run_backtest(start_date=START, end_date=END, starting_capital=5000.0, verbose=False)
    if not result:
        return {}

    metrics = result.get("metrics", {})
    equity = result.get("equity_series")
    crisis_metrics = {}

    if equity is not None and not equity.empty:
        crisis_slice = equity[(equity.index >= CRISIS_START) & (equity.index <= CRISIS_END)]
        if not crisis_slice.empty:
            crisis_metrics = compute_all_metrics(crisis_slice)

    output = {
        "test": "COVID19_Crash_2020",
        "crisis_period": f"{CRISIS_START} to {CRISIS_END}",
        "full_period": f"{START} to {END}",
        "full_period_metrics": metrics,
        "crisis_period_metrics": crisis_metrics,
        "key_findings": [
            f"Max Drawdown during COVID: {crisis_metrics.get('max_drawdown', 0):.1%}",
            f"Recovery speed captured: {metrics.get('cagr', 0):.1%} CAGR",
        ],
    }

    print("\n" + "=" * 60)
    print("STRESS TEST: COVID-19 Crash (V-Shape)")
    print("=" * 60)
    print(f"Full Period Return: {metrics.get('total_return', 0):.1%}")
    print(f"Max Drawdown:       {metrics.get('max_drawdown', 0):.1%}")
    print(f"Sharpe:             {metrics.get('sharpe_ratio', 0):.2f}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    with open("results/stress_test_2020.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to results/stress_test_2020.json")
    return output


if __name__ == "__main__":
    run_stress_test_2020()
