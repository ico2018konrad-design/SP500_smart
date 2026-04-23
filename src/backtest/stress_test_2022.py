"""Stress Test: 2022 Slow Bear Market (Jan-Oct 2022)."""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.honest_backtest import run_backtest
from src.backtest.performance_metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START = "2021-01-01"
END = "2023-12-31"
CRISIS_START = "2022-01-03"
CRISIS_END = "2022-10-12"


def run_stress_test_2022() -> dict:
    logger.info("Running stress test: 2022 Slow Bear (%s to %s)", CRISIS_START, CRISIS_END)

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
        "test": "2022_Slow_Bear",
        "crisis_period": f"{CRISIS_START} to {CRISIS_END}",
        "full_period": f"{START} to {END}",
        "full_period_metrics": metrics,
        "crisis_period_metrics": crisis_metrics,
        "key_findings": [
            f"2022 bear max drawdown: {crisis_metrics.get('max_drawdown', 0):.1%}",
            f"Bot survived slow bleed: {metrics.get('end_equity', 0) > 4000}",
        ],
    }

    print("\n" + "=" * 60)
    print("STRESS TEST: 2022 Slow Bear Market")
    print("=" * 60)
    print(f"Full Period Return: {metrics.get('total_return', 0):.1%}")
    print(f"Max Drawdown:       {metrics.get('max_drawdown', 0):.1%}")
    print(f"Bear MDD:           {crisis_metrics.get('max_drawdown', 0):.1%}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    with open("results/stress_test_2022.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to results/stress_test_2022.json")
    return output


if __name__ == "__main__":
    run_stress_test_2022()
