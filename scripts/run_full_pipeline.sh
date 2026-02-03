#!/usr/bin/env bash
# Full pipeline: layer search -> extract at best layer -> pretrain ActFormer -> run comparison.
# For quick test: use config with limit_docs (e.g. configs/tiny.yaml).
set -e
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/default.yaml}"
echo "[pipeline] Config: $CONFIG"
echo "[pipeline] Step 1: Layer search"
python -m src.layer_search --config "$CONFIG"
echo "[pipeline] Step 2: Extract at best layer (all splits)"
LAYER=$(PYTHONPATH=. python -c "
from pathlib import Path
from src.utils.io import load_json, load_yaml
config = load_yaml(Path(\"$CONFIG\"))
p = Path(config.get('layer_search',{}).get('layer_search_output','outputs/best_layer.json'))
print(load_json(p)['layer_index'])
")
python -m src.extract_activations --config "$CONFIG" --split all --layer_index "$LAYER"
echo "[pipeline] Step 3: Pretrain ActFormer"
python -m src.actformer.train --config "$CONFIG"
echo "[pipeline] Step 4: Probe comparison"
python -m src.probe.run_comparison --config "$CONFIG"
echo "[pipeline] Done."
