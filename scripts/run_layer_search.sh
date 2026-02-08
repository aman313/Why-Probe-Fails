#!/usr/bin/env bash
# Run layer search on the probe dataset (config data section): linear probe on each candidate layer, write best_layer.json.
# Use --run_extraction when memmaps for each layer do not exist yet (extracts probe data at every layer, then probes).
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
RUN_EXTRACT="${2:-}"  # Pass --run_extraction to extract activations at each layer first
python -m src.layer_search --config "$CONFIG" $RUN_EXTRACT
