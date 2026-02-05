#!/usr/bin/env bash
# Run ActFormer pretrain then probe comparison. Use after layer search and extraction at best layer.
# For linear vs ActFormer only, use: ./scripts/run_actformer_and_comparison.sh configs/linear_vs_actformer.yaml
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
echo "[pipeline] Config: $CONFIG"
echo "[pipeline] Step 1: ActFormer pretrain (at best layer)"
python -m src.actformer.train --config "$CONFIG"
echo "[pipeline] Step 2: Probe comparison"
python -m src.probe.run_comparison --config "$CONFIG"
echo "[pipeline] Done. Results: outputs/probe_comparison/comparison_metrics.json"
