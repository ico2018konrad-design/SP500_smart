#!/usr/bin/env bash
set -e
echo "Running SP500 Smart Bot — Stress Tests"
echo "========================================"
cd "$(dirname "$0")/.."
echo "--- 2008 Financial Crisis ---"
python src/backtest/stress_test_2008.py
echo ""
echo "--- 2020 COVID Crash ---"
python src/backtest/stress_test_2020.py
echo ""
echo "--- 2022 Slow Bear ---"
python src/backtest/stress_test_2022.py
echo ""
echo "All stress tests complete. Results in results/"
