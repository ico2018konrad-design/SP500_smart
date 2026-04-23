#!/usr/bin/env bash
set -e
echo "Starting SP500 Smart Bot — Paper Trading Mode"
echo "================================================"
echo "Make sure IBKR TWS/Gateway is running on port 7497"
echo ""
cd "$(dirname "$0")/.."
python src/main.py
