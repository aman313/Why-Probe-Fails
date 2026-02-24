# AGENTS.md

## Cursor Cloud specific instructions

This is a Python ML research codebase (no web server, database, or Docker required). All computation runs locally on CPU or GPU.

### Running the pipeline

See `README.md` for the quickstart commands. The fast-path test config is `configs/tiny.yaml` (2 docs, 2 layers, ~30s on CPU). The pipeline steps are:

1. `python3 -m src.extract_activations --config configs/tiny.yaml --split all --layer_index 0`
2. `python3 -m src.extract_activations --config configs/tiny.yaml --split all --layer_index 1`
3. `python3 -m src.layer_search --config configs/tiny.yaml`
4. `python3 -m src.actformer.train --config configs/tiny.yaml`
5. `python3 -m src.probe.run_comparison --config configs/tiny.yaml`

### Tests

- **Unit/integration tests** (fast, no network needed): `python3 -m pytest tests/ --ignore=tests/test_pipeline_e2e.py -v`
- **E2e tests** (slow, needs HuggingFace model download on first run): `python3 -m pytest tests/test_pipeline_e2e.py -v -m slow`

### Gotchas

- Use `python3` not `python` (the VM has no `python` symlink).
- The actformer branch saves checkpoints to timestamped subdirectories (e.g., `outputs/actformer_tiny/run_<timestamp>/best.pt`), not directly under the configured `output_dir`. The e2e tests in `test_pipeline_e2e.py` have known assertion failures due to this mismatch on the `actformer` branch.
- No linter is configured in the repo. Use `python3 -m py_compile <file>` for syntax checking if needed.
- The `distilgpt2` model is public and downloads without `HF_TOKEN`. Larger gated models (e.g., Llama) require `HF_TOKEN`.
- wandb is used for experiment tracking. If `WANDB_API_KEY` is not set, training still works but wandb will run in offline mode or prompt for login.
- No GPU in the Cloud VM; all pipeline steps run on CPU which is fine for the tiny config.
