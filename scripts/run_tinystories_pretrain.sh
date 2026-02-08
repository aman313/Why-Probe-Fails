#!/usr/bin/env bash
# Run TinyStories pretraining from scratch: prepare data, layer search (probe), extract pretrain at best layer, ActFormer train.
# Usage: ./scripts/run_tinystories_pretrain.sh [CONFIG] [--limit N]
#   CONFIG defaults to configs/tinystories_pretrain.yaml
#   --limit N passed to prepare_tinystories.py to cap number of stories (optional)
set -e
cd "$(dirname "$0")/.."
CONFIG="configs/tinystories_pretrain.yaml"
LIMIT_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT_ARGS="--limit $2"
      shift 2
      ;;
    *)
      CONFIG="$1"
      shift
      ;;
  esac
done

echo "[run_tinystories_pretrain] 1. Prepare TinyStories ..."
python scripts/prepare_tinystories.py --out_dir data/tinystories/text $LIMIT_ARGS

echo "[run_tinystories_pretrain] 2. Layer search (probe = RS1) ..."
bash scripts/run_layer_search.sh "$CONFIG" --run_extraction

echo "[run_tinystories_pretrain] 3. Extract TinyStories at best layer ..."
python -m src.extract_activations --config "$CONFIG" --for_pretrain --split all

echo "[run_tinystories_pretrain] 4. ActFormer pretrain ..."
# MPS fallback for transformer key_padding_mask op (not implemented on MPS)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python -m src.actformer.train --config "$CONFIG"

echo "[run_tinystories_pretrain] Done."
