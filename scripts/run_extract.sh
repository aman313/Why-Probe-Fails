#!/usr/bin/env bash
# Extract activations for all splits at the layer given in config (or best_layer if available).
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -m src.extract_activations --config "$CONFIG" --split all
