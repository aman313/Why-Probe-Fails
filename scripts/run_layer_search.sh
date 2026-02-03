#!/usr/bin/env bash
# Run layer search; writes best_layer.json. Optionally run extraction per layer with --run_extraction.
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -m src.layer_search --config "$CONFIG"
