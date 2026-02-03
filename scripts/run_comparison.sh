#!/usr/bin/env bash
# Run probe comparison (ID vs OOD) for all probe_types in config.
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -m src.probe.run_comparison --config "$CONFIG"
