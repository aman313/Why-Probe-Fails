# Activation-Sequence Probe Pipeline: Current Plan

This document is the single plan: design overview first, then step-by-step implementation with testing and validation after each step.

---

## Design overview

**Goal**  
Extract per-token activations from a frozen LM layer, pretrain a small transformer (ActFormer) to predict the next-step activation, then use ActFormer hidden states (or raw activations) as features to train classifier probes. Compare probe types on in-domain (ID) vs out-of-domain (OOD) data.

**Repo fit**  
- Data: existing CSV layout under `data/RS1` (etc.) with `prompt` column; labels from folder (`benign` → 0, `malicious` → 1). No HuggingFace datasets; use pandas + glob.
- **Base model**: Any HuggingFace causal LM (default `distilgpt2`) whose activations we extract. Switch by changing `model.base_model_name` in config (e.g. `gpt2`, `meta-llama/Llama-3.1-8B-Instruct`). Set `model.cache_dir` to a path inside the repo (e.g. `.cache/huggingface`) if the default HF cache has permission issues.
- **Probe**: The classifier trained to predict labels from (pooled) activations is **linear**, **mlp**, or **transformer**; the **transformer probe is implemented from scratch** (no pretrained weights).
- Document: one row = one document. Doc-level split = by row (or by `source_file` for strict ID/OOD). Train/val/test doc_ids disjoint.
- Layer search runs first; extraction for ActFormer and probes is only at the **best layer** (one activation dataset per run).
- Probe comparison assumes **layer search and extraction are already done** for ID and each OOD set; comparison script only loads pre-extracted memmaps.

**Activation dataset for ActFormer pretraining**  
- Built from memmap + index (from extraction at the chosen layer). Per sequence: load slice `[start:start+length]`, normalize with train mean/std, form (x, y) with x = positions 0..T-1, y = positions 1..T (next-step). Optional **subsequence sampling**: random contiguous slice (start, length) for more views per document.
- ActFormer trains on this self-supervised task only; probe training uses the same memmap (pooled raw or ActFormer pooled).

**Probe comparison (ID vs OOD)**  
- **Config**: `activations.id` (memmap_dir + index_path), `activations.ood` (list of `name`, memmap_dir, index_path), `probe_types` (e.g. raw_linear, actformer_linear), `actformer_checkpoint`, `pooling`, `probe_train`, `metrics`, `output_dir`.
- **Main function**: For each probe type, get pooled features for ID train/val/test and each OOD set; train probe on ID train only; evaluate on ID test and each OOD; build table (probe_type × metric × dataset); save `comparison_metrics.json`, `comparison_table.csv`, optional plot.

**Augmentation**  
- ActFormer: **subsequence sampling** (random contiguous slice) as main augmentation; optional input noise/dropout.  
- Probe: optional Gaussian noise and same-class mixup on pooled features.

**Pipeline order**  
1. Layer search (on ID) → `best_layer.json`.  
2. Extract activations at best layer for ID and for each OOD set.  
3. (Optional) ActFormer pretrain on ID activations.  
4. Run probe comparison (train each probe type on ID, evaluate on ID test + each OOD).

**Probe comparison config (reference)**  
```yaml
activations:
  id: { memmap_dir: ..., index_path: ... }
  ood:
    - name: RS2
      memmap_dir: ...
      index_path: ...
probe_types: [raw_linear, raw_mlp, actformer_linear, actformer_mlp]
actformer_checkpoint: ...
pooling: mean
probe_train: { lr, epochs, batch_size, l2 }
metrics: [accuracy, macro_f1, auroc]
output_dir: ...
```

**How to build**

- **Implement the plan (build the codebase)**  
  Ask the agent: *"Implement the plan in IMPLEMENTATION_PLAN.md step by step"* or *"Execute IMPLEMENTATION_PLAN.md"*. Implementation proceeds in order: Step 1 → Step 12, with testing after each step.

- **Run the pipeline (after implementation)**  
  Use the scripts in order:  
  `run_layer_search.sh` → `run_extract.sh` (at best layer, for ID and OOD) → `run_pretrain_actformer.sh` → `run_train_probe.sh` (optional single-probe) → `run_comparison.sh`.  
  Or add a single `run_full_pipeline.sh` that runs these in sequence (Step 10 deliverable).

---

## Step-by-step implementation

Each step lists deliverables, then **Testing and validation** to run before moving on.

---

## Step 1: Project setup and foundations

**Deliverables**

- `requirements.txt` (or `pyproject.toml`): PyTorch, transformers, pandas, numpy, scipy, scikit-learn, PyYAML; wandb optional.
- Directory layout: `configs/`, `src/`, `src/actformer/`, `src/probe/`, `src/utils/`, `scripts/`, `tests/`.
- `src/utils/seed.py`: `set_seed(seed)` setting PyTorch, numpy, random.
- `src/utils/io.py`: helpers to load/save YAML, JSON; ensure dir exists.
- `src/utils/metrics.py`: `compute_metrics(y_true, y_pred, y_prob, metrics_list)` returning dict (accuracy, macro_f1, auroc).
- `configs/default.yaml`: skeleton with all sections (data, model, extraction, layer_search, actformer, probe, comparison, seeds) and dev defaults (distilgpt2, layer 3, resid_post, data/RS1, etc.).

**Testing and validation**

- From repo root: `pip install -r requirements.txt` (or editable install); `python -c "import src.utils.seed; import src.utils.io; import src.utils.metrics; from src.utils.metrics import compute_metrics"` succeeds.
- Load config: `python -c "from src.utils.io import load_yaml; c = load_yaml('configs/default.yaml'); assert 'base_model_name' in c and 'seed' in c"`.
- Run a quick seed check: set seed, generate two random numbers, reset and set same seed again, assert same two numbers.

---

## Step 2: Data loading and splits (doc-level, no leakage)

**Deliverables**

- `src/utils/data_loading.py` (or under `src/`): load all CSVs from `data_dir` by `categories`; build one table with columns: `doc_id`, `text` (prompt), `label` (from folder), `source_file` (e.g. `benign/alpaca.csv`). Support `label_map` from config.
- Split logic: either (a) stratified split by `doc_id` (train/val/test ratios), or (b) `split_by_file`: assign each `source_file` to one of train/val/test (stratified by label when possible). Return or save split assignments so the same split is reused everywhere.
- Assertion: document that train/val/test doc_id sets are disjoint; add `assert_disjoint_splits(train_ids, val_ids, test_ids)` and call it in tests.

**Testing and validation**

- Unit test: load `data/RS1` with categories `[benign, malicious]`; assert number of rows > 0, labels in {0,1}, `source_file` contains folder names.
- With `split_by_file=False`: run split, assert `len(set(train_ids) & set(val_ids)) == 0` and same for train/test, val/test.
- With `split_by_file=True`: run split, assert no `source_file` appears in two splits; assert disjoint doc_ids.
- Optional: save a small fixture (e.g. 2 CSVs with 2 rows each) under `tests/fixtures/` and run data loading + split on it.

---

## Step 3: Activation extraction

**Deliverables**

- `src/extract_activations.py`: entrypoint `python -m src.extract_activations --config configs/default.yaml [--split train|val|test|all] [--layer_index L]`.
- Use data loading from Step 2 to get docs and splits; filter by `--split` when not `all`.
- For each doc: tokenize, chunk with `max_seq_len` and `stride` (no cross-doc mixing). Run frozen base model with `output_hidden_states=True`; take `tap_point` (e.g. `resid_post` → `hidden_states[layer_index]`). Write activations to sharded memmap `[N_tokens, d]`; append to index: `doc_id`, `seq_id`, `start`, `length`, `label`, `split`, optional `text_ref`.
- Compute and save **train-only** mean/std for normalization (when `--split all` or when processing train).
- Support `limit_docs` for quick runs. Progress logging.

**Testing and validation**

- Run extraction on **2 documents**, one layer (e.g. 0), `max_seq_len=64`, `stride=32`: `python -m src.extract_activations --config configs/default.yaml --split all --layer_index 0` with `limit_docs: 2` in config.
- Check: memmap file(s) exist; index has 2 docs’ entries; for each entry `start + length` <= total tokens; `load_memmap` + index and read one sequence, assert shape `[length, d]`.
- Assert train mean/std file exists when train split was included; load and assert shape `(d,)` for mean and std.

---

## Step 4: Layer search

**Deliverables**

- `src/layer_search.py`: CLI `python -m src.layer_search --config configs/default.yaml`.
- For each layer in `layer_search_layers`: run extraction for that layer (or load from existing per-layer memmap if available); load train/val activations from index; **pool** (e.g. mean over tokens) per document to get one vector per doc; fit sklearn LogisticRegression (or equivalent) on train; evaluate on val (macro_f1 or accuracy). Record metric per layer.
- Write `best_layer.json` with `{"layer_index": best_layer, "metric": "macro_f1", "scores": {0: 0.8, 1: 0.82, ...}}`.

**Testing and validation**

- With 2 layers and 2 documents (reuse Step 3’s tiny run): run layer search pointing to pre-extracted memmaps for layer 0 and layer 1 (or run extraction twice with `limit_docs=2`).
- Assert `best_layer.json` exists; `layer_index` is one of the candidate layers; `scores` has one entry per candidate layer.
- Unit test: mock index + small arrays, run “probe per layer” logic, assert best layer is chosen by val metric.

---

## Step 5: ActFormer dataset and model

**Deliverables**

- `src/actformer/data.py`: PyTorch `Dataset` that reads from memmap + index; returns (x, y) where x = normalized activations `[T-1, d]`, y = next-step target `[T-1, d]` (or full `[T,d]` with shift in training loop). Use train mean/std for normalization. **Subsequence sampling**: optional random contiguous slice `(start, length)` so each sample is `x[s:s+L-1]`, `y[s+1:s+L]` with `L >= 2` and `s + L <= T`; config `min_subseq_len`, `max_subseq_len`.
- Collator: pad to batch max length, build attention mask for loss.
- `src/actformer/model.py`: input projection Linear(d → d_model), causal decoder-only transformer, output head for MSE (one Linear) or Gaussian NLL (mu + log_sigma); expose `last_hidden_state [B, T, d_model]`. Causal mask in attention.

**Testing and validation**

- Unit test: create a tiny memmap (e.g. 3 sequences, 10 tokens each, d=4); index with start/length. Dataset returns (x, y) with consistent shapes; with subsequence sampling, lengths vary.
- One batch: collate a few samples, run model forward; assert `last_hidden_state.shape` and loss scalar; `loss.backward()` runs.
- Optional: test Gaussian NLL path (two heads, clamp log_sigma).

---

## Step 6: ActFormer training and evaluation

**Deliverables**

- `src/actformer/train.py`: training loop (optionally mixed precision); gradient accumulation; eval every `eval_every`; save checkpoints and best by val loss; optional wandb. CLI: `python -m src.actformer.train --config configs/default.yaml`.
- `src/actformer/eval.py`: load checkpoint, run on test split; return dict with MSE, cosine similarity, NLL (if Gaussian).

**Testing and validation**

- Run pretraining for **50 steps** on the 2-doc activations from Step 3: `python -m src.actformer.train --config configs/default.yaml` with config pointing to that memmap and `max_steps=50` or small epochs.
- Assert checkpoint or best checkpoint is written; `eval.py` runs and returns a dict with at least one of (mse, cosine_sim, nll).
- Sanity check: loss decreases over the 50 steps (or stays stable).

---

## Step 7: Probe training (single probe type, ID only)

**Deliverables**

- `src/probe/train_probe.py`: load ID memmap + index; for each split (train/val/test) get **pooled features** (mean / last / attention_pool over tokens). Feature source: **raw** (normalized activations) or **ActFormer** (run sequences through ActFormer, pool `last_hidden_state`). Train linear or MLP classifier on ID train; evaluate on ID val and ID test. Save probe checkpoint and metrics JSON. CLI: `python -m src.probe.train_probe --config configs/default.yaml --use_actformer true|false` (and probe type from config).
- Shared helper: `get_pooled_features(memmap, index_entries, feature_source, actformer, pooling)`.

**Testing and validation**

- Train **linear probe on raw** for 1 epoch on 2-doc data: `python -m src.probe.train_probe --config configs/default.yaml --use_actformer false`. Assert metrics JSON and probe checkpoint (or sklearn model dump) exist; metrics contain accuracy and macro_f1 (and auroc if binary).
- Train **linear probe on ActFormer** for 1 epoch (use ActFormer from Step 6): same config, `--use_actformer true`. Assert run completes and metrics are saved.
- Assert no data leakage: probe is trained only on train split; val/test are not used for fitting.

---

## Step 8: Baselines and probe comparison (ID vs OOD)

**Deliverables**

- `src/probe/baselines.py`: raw probe (already in train_probe); optional PCA on pooled train activations, then linear probe. Function that runs baseline and returns metrics.
- `src/probe/run_comparison.py` (or extend train_probe): `run_probe_comparison(config)` as in the earlier design. Config: `activations.id`, `activations.ood` (list of name + memmap_dir + index_path), `probe_types` (e.g. raw_linear, raw_mlp, actformer_linear, actformer_mlp), `actformer_checkpoint`, `pooling`, `probe_train`, `metrics`, `output_dir`. For each probe type: get features for ID train/val/test and each OOD set; train on ID train; evaluate on ID test and each OOD; build table (probe_type × metric × dataset); save `comparison_metrics.json`, `comparison_table.csv`, optional plot.
- CLI: `python -m src.probe.run_comparison --config configs/default.yaml` (or a dedicated comparison config).

**Testing and validation**

- With **tiny ID and one OOD** memmap (e.g. 2-doc ID and 2-doc OOD from same extraction layout): run comparison for `probe_types: [raw_linear, actformer_linear]`. Assert `comparison_metrics.json` has structure `{ probe_type: { id: {...}, ood: { OOD_NAME: {...} } } }`; table has columns for id and OOD_NAME; no crash.
- Assert baselines (raw_linear, raw_mlp) and actformer_linear all appear in the table with numeric values.

---

## Step 9: Augmentation (subsequence sampling already in Step 5; optional noise/mixup)

**Deliverables**

- ActFormer: subsequence sampling already implemented in Step 5; add config knobs `min_subseq_len`, `max_subseq_len` and document.
- Optional: Gaussian noise on ActFormer input in dataset (`noise_std`) and/or input dropout; config in `actformer.augment`.
- Probe: optional Gaussian noise on pooled features and same-class mixup in probe training; config in `probe.augment`. Implement in `train_probe` / `get_pooled_features` or DataLoader.

**Testing and validation**

- With `augment.noise_std > 0` and `augment.mixup_alpha > 0`: run probe training for 1 epoch; assert no crash and metrics are still computed.
- With ActFormer subsequence sampling enabled: run 50-step pretrain again; assert different sequence lengths in batches (or that sampling is used); loss still decreases.

---

## Step 10: CLI and run scripts

**Deliverables**

- `scripts/run_extract.sh`: example calling `python -m src.extract_activations --config configs/default.yaml --split all`.
- `scripts/run_layer_search.sh`: run layer search.
- `scripts/run_pretrain_actformer.sh`: run ActFormer train.
- `scripts/run_train_probe.sh`: run probe train (with/without ActFormer).
- `scripts/run_comparison.sh`: run probe comparison.
- `scripts/run_full_pipeline.sh`: run the above in order (layer search → extract at best layer for ID + OOD → pretrain ActFormer → run comparison); optional `--limit_docs` for a quick build.
- All CLIs support `--config` and print `--help`.

**Testing and validation**

- Run each script with `--help` or dry-run (e.g. `limit_docs=2`) and confirm they invoke the right module with the right args.
- Optional: run `run_extract.sh` then `run_layer_search.sh` then `run_pretrain_actformer.sh` then `run_train_probe.sh` then `run_comparison.sh` in sequence with tiny config and assert all complete.

---

## Step 11: End-to-end test

**Deliverables**

- `tests/test_pipeline_e2e.py`: single test that (1) loads config with `limit_docs=2`, one layer, (2) runs extraction for that layer, (3) runs layer search over 2 layers (using that extraction or minimal extraction), (4) runs ActFormer pretrain for 50 steps, (5) runs probe train for 1 epoch (raw and optionally ActFormer), (6) runs comparison with ID + one OOD (can reuse same 2-doc data as “OOD” for test). Assert: no exceptions; extraction artifacts exist; best_layer.json exists; ActFormer checkpoint exists; probe/comparison metrics and table exist. Prefer CPU or small device so CI can run.

**Testing and validation**

- `pytest tests/test_pipeline_e2e.py -v` (or `python -m pytest`). Test passes on a clean environment with dependencies installed.
- Optional: mark as `@pytest.mark.slow` and run in CI only when requested.

---

## Step 12: README and documentation

**Deliverables**

- Update `README.md`: add section “Activation-sequence probe pipeline” with quickstart (small data_dir, limit_docs, distilgpt2); order of steps: layer search → extract at best layer (ID + OOD) → pretrain ActFormer → run comparison. Document doc-level (and file-level) split, no leakage; where artifacts live; storage tips; how to switch to a larger base model. Mention subsequence sampling and optional augmentation.

**Testing and validation**

- Manually follow quickstart in README on a fresh clone (or in a clean venv) and confirm commands run and produce the described outputs.

---

## Dependency order (summary)

```
Step 1 (setup) → Step 2 (data/splits) → Step 3 (extraction)
       → Step 4 (layer search, uses 3)
       → Step 5 (ActFormer data/model) → Step 6 (ActFormer train/eval)
       → Step 7 (probe train) → Step 8 (baselines + comparison)
       → Step 9 (augmentation) → Step 10 (scripts) → Step 11 (e2e) → Step 12 (README)
```

Layer search (Step 4) assumes extraction (Step 3) can be run per layer (or that per-layer memmaps exist). Comparison (Step 8) assumes layer search is already done and extraction has been run for ID and OOD at the chosen layer.
