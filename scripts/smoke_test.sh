#!/bin/bash
# SP500 Smart Scalper — Smoke Test
# Runs unit tests + basic import checks to verify the strategy stack is intact.
#
# Usage: bash scripts/smoke_test.sh

set -euo pipefail

echo "========================================"
echo " SP500 Smart Scalper — Smoke Tests"
echo "========================================"

# Ensure we run from repo root
cd "$(dirname "$0")/.."

echo ""
echo "▶ Running unit tests..."
pytest tests/ -v --tb=short || {
    echo "✗ Unit tests FAILED"
    exit 1
}

echo ""
echo "▶ Checking strategy module imports..."
python -c "
from src.execution.paper_trader import PaperTrader
from src.backtest.honest_backtest import run_backtest
from src.backtest.walk_forward import run_rolling_oos_validation
from src.data.macro_timeseries import MacroTimeSeries
from src.positions.anti_martingale_scaler import AntiMartingaleScaler
from src.hedge.baseline_hedge import BaselineHedge
from src.signals.long_signals import LongSignalGenerator
from src.signals.short_signals import ShortSignalGenerator
from src.risk.circuit_breakers import CircuitBreakers
print('✓ All strategy modules import OK')
"

echo ""
echo "========================================"
echo " ✓ All smoke tests passed"
echo "========================================"
