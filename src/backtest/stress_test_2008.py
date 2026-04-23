"""Stress Test: 2008 Financial Crisis (Sep 2008 - Mar 2009).

Tests bot behavior during the worst financial crisis since 1929.
Output: results/stress_test_2008.json
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtest.honest_backtest import run_backtest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START = "2007-01-01"
END = "2010-12-31"
CRISIS_START = "2008-09-01"
CRISIS_END = "2009-03-31"


def run_stress_test_2008() -> dict:
    logger.info("Running stress test: 2008 Financial Crisis (%s to %s)", CRISIS_START, CRISIS_END)

    result = run_backtest(
        start_date=START,
        end_date=END,
        starting_capital=5000.0,
        verbose=False,
    )

    if not result:
        return {}

    metrics = result.get("metrics", {})
    equity = result.get("equity_series")

    # Extract crisis period metrics if available
    crisis_metrics = {}
    if equity is not None and not equity.empty:
        crisis_slice = equity[
            (equity.index >= CRISIS_START) &
            (equity.index <= CRISIS_END)
        ]
        if not crisis_slice.empty:
            from src.backtest.performance_metrics import compute_all_metrics
            crisis_metrics = compute_all_metrics(crisis_slice)

    output = {
        "test": "2008_Financial_Crisis",
        "crisis_period": f"{CRISIS_START} to {CRISIS_END}",
        "full_period": f"{START} to {END}",
        "full_period_metrics": metrics,
        "crisis_period_metrics": crisis_metrics,
        "survival": metrics.get("end_equity", 0) > 0,
        "key_findings": [
            f"Bot {'survived' if metrics.get('end_equity', 0) > 0 else 'blown up'} the 2008 crisis",
            f"Max Drawdown: {metrics.get('max_drawdown', 0):.1%}",
            f"Crisis period MDD: {crisis_metrics.get('max_drawdown', 0):.1%}",
        ],
    }

    print("\n" + "=" * 60)
    print("STRESS TEST: 2008 Financial Crisis")
    print("=" * 60)
    print(f"Full Period Return: {metrics.get('total_return', 0):.1%}")
    print(f"Max Drawdown:       {metrics.get('max_drawdown', 0):.1%}")
    print(f"Sharpe:             {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Survived:           {'YES ✓' if output['survival'] else 'NO ✗'}")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    with open("results/stress_test_2008.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("Results saved to results/stress_test_2008.json")

    return output


if __name__ == "__main__":
    run_stress_test_2008()
