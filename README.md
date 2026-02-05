<div align="center">
    <h1>False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize</h1>


[![arxiv](https://img.shields.io/badge/Arxiv-2509.03888-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2509.03888) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
</div>



## 📝 Overview  
Large Language Models (LLMs) can comply with harmful instructions, raising critical safety concerns.  
This project systematically re-examines probing-based methods for malicious input detection.  

Our study reveals that probing classifiers:  
- Achieve near-perfect accuracy in in-domain evaluations but collapse on out-of-distribution data.  
- Rely mainly on **instructional patterns** and **trigger words**, not true harmfulness semantics.  
- Create a *false sense of security*, highlighting the need for more principled safety detection approaches.  

---

## ⚙️ Reproduction  
We provide scripts to reproduce all key experiments:  

1. **Dataset preparation** – We have provided the dataset in `data` folder.
2. **Hidden state extraction** – Collect internal representations from target LLMs using `get_hidden_states.py`
3. **Classification** – Train SVMs classifiers on the extracted representations and evaluate the results using `classify.py`
 

---

## Activation-sequence probe pipeline

This repo includes an **activation-sequence probe** pipeline: extract per-token activations from a frozen LM layer, pretrain a small transformer (ActFormer) to predict the next-step activation, then train classifier probes (linear, MLP, or transformer-from-scratch) on pooled features and compare them on in-domain (ID) vs out-of-domain (OOD) data.

### Quickstart (small run)

```bash
pip install -r requirements.txt
# Use tiny config (2 docs, 2 layers) for speed. Set model.cache_dir in config if HF cache has permission issues.
python -m src.extract_activations --config configs/tiny.yaml --split all --layer_index 0
python -m src.extract_activations --config configs/tiny.yaml --split all --layer_index 1
python -m src.layer_search --config configs/tiny.yaml
python -m src.actformer.train --config configs/tiny.yaml
python -m src.probe.run_comparison --config configs/tiny.yaml
```

Or run the full pipeline script:

```bash
./scripts/run_full_pipeline.sh configs/tiny.yaml
```

### Pipeline order (no leakage)

1. **Layer search** – Linear probe on each candidate layer; writes `outputs/best_layer.json`.
2. **Extract** – Extract activations at the best layer for ID (and optionally OOD) data. Splits are by **document** (or by file); train/val/test are disjoint.
3. **ActFormer pretrain** – Next-step activation prediction on ID activations (optional subsequence sampling).
4. **Probe comparison** – Train each probe type (e.g. raw_linear, actformer_linear) on ID; evaluate on ID test and each OOD set; save `comparison_metrics.json` and table.

### Artifacts

- `outputs/activations_*/layer_<L>/` – Memmap `activations.dat`, `index.json`, `meta.json`, `train_mean.npy`, `train_std.npy`.
- `outputs/best_layer.json` – Best layer index and per-layer scores.
- `outputs/actformer*/best.pt` – ActFormer checkpoint.
- `outputs/probe_comparison*/comparison_metrics.json` – ID and OOD metrics per probe type.

### Apple Silicon (MPS)

On Macs with Apple Silicon, the pipeline uses the MPS backend when available. If you see CPU being used instead, run the diagnostic from the repo root:

```bash
PYTHONPATH=. python scripts/diagnose_mps.py
```

It will print PyTorch version, platform (arm64 vs x86_64), and whether MPS is built and available. Ensure PyTorch is installed for Mac ARM (e.g. `pip install torch` from a native arm64 shell) and macOS is 12.3+.

### Switching the base model

In `configs/default.yaml`, set `model.base_model_name` to any HuggingFace causal LM (e.g. `gpt2`, `distilgpt2`, `meta-llama/Llama-3.1-8B-Instruct`). Use `model.cache_dir: .cache/huggingface` (or another writable path) if the default HF cache is not writable.

### Tests

```bash
pytest tests/test_pipeline_e2e.py -v -m slow   # full e2e (needs extraction/network first time)
```

---

## 📚 Citation  
If you find this work useful, please cite:  

```bibtex
@misc{wang2025falsesensesecurityprobingbased,
      title={False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize}, 
      author={Cheng Wang and Zeming Wei and Qin Liu and Muhao Chen},
      year={2025},
      eprint={2509.03888},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03888}, 
}
```
