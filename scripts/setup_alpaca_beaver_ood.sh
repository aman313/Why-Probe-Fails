#!/usr/bin/env bash
# 1) Create data/id_alpaca_beaver and OOD dirs with symlinks to source CSVs.
# 2) Run OOD extraction for dolly, harmbench, advbench at the best layer.
# Run from repo root. Requires outputs/best_layer.json (run layer search first on ID config).
set -e
cd "$(dirname "$0")/.."

echo "[setup] Creating ID and OOD data dirs and symlinks ..."
mkdir -p data/id_alpaca_beaver/benign data/id_alpaca_beaver/malicious
mkdir -p data/ood_dolly/benign data/ood_harmbench/malicious data/ood_advbench/malicious

cd data
ln -sf ../../RS1/benign/alpaca.csv id_alpaca_beaver/benign/alpaca.csv 2>/dev/null || true
ln -sf ../../RS1/malicious/beaver.csv id_alpaca_beaver/malicious/beaver.csv 2>/dev/null || true
ln -sf ../../RS1/benign/dolly.csv ood_dolly/benign/dolly.csv 2>/dev/null || true
ln -sf ../../RS2/malicious/harmbench.csv ood_harmbench/malicious/harmbench.csv 2>/dev/null || true
ln -sf ../../RS2/malicious/advbench.csv ood_advbench/malicious/advbench.csv 2>/dev/null || true
cd ..

echo "[setup] Reading best layer from outputs/best_layer.json ..."
LAYER=$(PYTHONPATH=. python -c "
from pathlib import Path
from src.utils.io import load_json
p = Path('outputs/best_layer.json')
if not p.exists():
    raise SystemExit('outputs/best_layer.json not found. Run layer search first (e.g. with configs/alpaca_beaver_ood.yaml).')
print(load_json(p)['layer_index'])
")

echo "[setup] Extracting OOD activations at layer $LAYER (dolly, harmbench, advbench) ..."
python -m src.extract_activations --config configs/ood_dolly.yaml --split all --layer_index "$LAYER"
python -m src.extract_activations --config configs/ood_harmbench.yaml --split all --layer_index "$LAYER"
python -m src.extract_activations --config configs/ood_advbench.yaml --split all --layer_index "$LAYER"

ID_LAYER_DIR="outputs/activations/layer_$LAYER"
for ood in dolly harmbench advbench; do
  if [ -f "$ID_LAYER_DIR/train_mean.npy" ] && [ -d "outputs/activations_ood/$ood/layer_$LAYER" ]; then
    cp "$ID_LAYER_DIR/train_mean.npy" "$ID_LAYER_DIR/train_std.npy" "outputs/activations_ood/$ood/layer_$LAYER/"
  fi
done

echo "[setup] Done. OOD activations: outputs/activations_ood/{dolly,harmbench,advbench}/layer_$LAYER/"
echo "[setup] Run comparison with: python -m src.probe.run_comparison --config configs/alpaca_beaver_ood.yaml"
echo "[setup] If best layer is not 3, update comparison.activations.ood paths in configs/alpaca_beaver_ood.yaml to use layer_$LAYER"
