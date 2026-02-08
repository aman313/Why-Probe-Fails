#!/usr/bin/env bash
# Train probe. PROBE_TYPE: raw_linear | raw_mlp | actformer_finetuned (default: raw_linear).
# Legacy: USE_ACTFORMER=true => actformer_finetuned, USE_ACTFORMER=false => raw_linear.
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
if [ -z "$PROBE_TYPE" ]; then
  [ "$USE_ACTFORMER" = "true" ] && PROBE_TYPE="actformer_finetuned" || PROBE_TYPE="raw_linear"
fi
python -m src.probe.train_probe --config "$CONFIG" --probe_type "${PROBE_TYPE:-raw_linear}"
