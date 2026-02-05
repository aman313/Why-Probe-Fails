#!/usr/bin/env bash
# Run full pipeline with run-scoped output directory.
# Usage:
#   ./scripts/run_with_run_dir.sh configs/alpaca_beaver_ood.yaml   # new run -> outputs/run_<timestamp>/
#   ./scripts/run_with_run_dir.sh outputs/run_<id>/config.yaml      # reproduce (reuse that run dir)
set -e
cd "$(dirname "$0")/.."
# Allow ActFormer to run on Apple Silicon (MPS) where some ops fall back to CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
USER_CONFIG="${1:?Usage: run_with_run_dir.sh <config.yaml>}"
USER_CONFIG="$(cd "$(dirname "$USER_CONFIG")" && pwd)/$(basename "$USER_CONFIG")"

# Detect reproduction mode: config already under outputs/run_*
RUN_DIR=""
CONFIG="$USER_CONFIG"
RUN_DIR=$(PYTHONPATH=. python -c "
from pathlib import Path
from src.run_dir import get_run_dir_from_config_path
rd = get_run_dir_from_config_path(Path('$USER_CONFIG'))
if rd is not None:
    print(rd.resolve())
" 2>/dev/null)
if [ -n "$RUN_DIR" ]; then
    echo "[run_with_run_dir] Reproduction mode: run_dir=$RUN_DIR"
else
    echo "[run_with_run_dir] Creating new run dir from $USER_CONFIG"
    CREATE_OUT=$(PYTHONPATH=. python -c "
from pathlib import Path
from src.run_dir import create_run_dir
rd, cfg = create_run_dir(Path('$USER_CONFIG'))
print(rd.resolve(), cfg.resolve())
")
    RUN_DIR=$(echo "$CREATE_OUT" | awk '{print $1}')
    CONFIG=$(echo "$CREATE_OUT" | awk '{print $2}')
    echo "[run_with_run_dir] Run dir: $RUN_DIR"
fi

echo "[run_with_run_dir] Config: $CONFIG"
echo "[run_with_run_dir] Step 1: Layer search (with extraction)"
python -m src.layer_search --config "$CONFIG" --run_extraction

echo "[run_with_run_dir] Step 2: Update run config with best layer and extract ID at best layer"
LAYER=$(PYTHONPATH=. python -c "
from pathlib import Path
from src.utils.io import load_json
p = Path('$RUN_DIR') / 'best_layer.json'
print(load_json(p)['layer_index'])
")
PYTHONPATH=. python -c "
from pathlib import Path
from src.run_dir import update_run_config_after_layer_search
update_run_config_after_layer_search(Path('$RUN_DIR'), $LAYER)
"
python -m src.extract_activations --config "$CONFIG" --split all --layer_index "$LAYER"

# Step 3: OOD extraction if comparison.activations.ood is non-empty
echo "[run_with_run_dir] Step 3: OOD extraction (if any)"
PYTHONPATH=. python -c "
from pathlib import Path
from src.utils.io import load_yaml
from src.run_dir import run_ood_extraction
cfg = load_yaml(Path('$CONFIG'))
ood = cfg.get('comparison', {}).get('activations', {}).get('ood', [])
if ood:
    run_ood_extraction(Path('$RUN_DIR'), $LAYER, ood)
else:
    print('  No OOD datasets in config, skipping.')
"

echo "[run_with_run_dir] Step 4: Pretrain ActFormer"
python -m src.actformer.train --config "$CONFIG"

echo "[run_with_run_dir] Step 5: Probe comparison"
python -m src.probe.run_comparison --config "$CONFIG"

echo "[run_with_run_dir] Done. Run dir: $RUN_DIR"
