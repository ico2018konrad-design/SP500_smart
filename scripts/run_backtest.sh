#!/usr/bin/env bash
set -e
echo "Running SP500 Smart Bot — Full Backtest"
echo "========================================"
cd "$(dirname "$0")/.."
python src/backtest/honest_backtest.py
echo "Done. Results in results/backtest_full.json"
