#!/usr/bin/env bash
# Pretrain ActFormer on extracted activations (use config's layer or best_layer).
# When config has pretrain.memmap_dir set, activations are read from there at best layer; run extraction
# for the pretrain dataset first (see run_extract.sh --for_pretrain), or pass --run_extraction to run it here.
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
python -m src.actformer.train --config "$CONFIG"
