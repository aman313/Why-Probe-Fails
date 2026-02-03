#!/usr/bin/env bash
# Pretrain ActFormer on extracted activations (use config's layer or best_layer).
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -m src.actformer.train --config "$CONFIG"
