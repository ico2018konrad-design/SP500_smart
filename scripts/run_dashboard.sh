#!/usr/bin/env bash
set -e
echo "Starting SP500 Smart Bot Dashboard"
echo "===================================="
cd "$(dirname "$0")/.."
streamlit run src/monitoring/dashboard.py --server.port 8501
