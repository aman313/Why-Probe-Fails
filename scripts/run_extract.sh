#!/usr/bin/env bash
# Extract activations for all splits at the layer given in config (or best_layer if available).
# For a separate pretrain dataset (config pretrain.data + pretrain.memmap_dir), run with --for_pretrain
# and best layer, e.g.:
#   python -m src.extract_activations --config "$CONFIG" --for_pretrain --layer_index $(jq -r .layer_index outputs/best_layer.json) --split all
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -m src.extract_activations --config "$CONFIG" --split all
