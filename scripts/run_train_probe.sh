#!/usr/bin/env bash
# Train probe (set USE_ACTFORMER=true to use ActFormer features).
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
USE_ACTFORMER="${USE_ACTFORMER:-false}"
python -m src.probe.train_probe --config "$CONFIG" --use_actformer "$USE_ACTFORMER"
