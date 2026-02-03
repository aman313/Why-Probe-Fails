"""
End-to-end test: run pipeline with tiny config (2 docs, 2 layers, 50 steps pretrain, 1 epoch probe).
Requires: data/RS1. For extraction (first run) needs network + HF cache.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/tiny.yaml"


@pytest.mark.slow
def test_pipeline_e2e():
    """Run layer search -> extract at best -> ActFormer train -> comparison. Assumes layer 0/1 already extracted or runs extraction."""
    def run(cmd, env=None):
        env = env or {}
        env["PYTHONPATH"] = str(REPO_ROOT)
        return subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env={**__import__("os").environ, **env},
            capture_output=True,
            text=True,
            timeout=300,
        )

    # 0) Extract layer 0 and 1 if not present (needs network first time)
    out_tiny = REPO_ROOT / "outputs/activations_tiny"
    if not (out_tiny / "layer_0/index.json").exists():
        r = run([sys.executable, "-m", "src.extract_activations", "--config", CONFIG, "--split", "all", "--layer_index", "0"], env={"PYTHONPATH": str(REPO_ROOT)})
        if r.returncode != 0:
            pytest.skip(f"Extraction failed (need network?): {r.stderr[:500]}")
        run([sys.executable, "-m", "src.extract_activations", "--config", CONFIG, "--split", "all", "--layer_index", "1"])
    assert (out_tiny / "layer_0/index.json").exists()

    # 1) Layer search
    r = run([sys.executable, "-m", "src.layer_search", "--config", CONFIG])
    assert r.returncode == 0, r.stderr
    best_path = REPO_ROOT / "outputs/best_layer.json"
    assert best_path.exists()

    # 2) ActFormer train (uses best layer)
    r = run([sys.executable, "-m", "src.actformer.train", "--config", CONFIG])
    assert r.returncode == 0, r.stderr
    assert (REPO_ROOT / "outputs/actformer_tiny/best.pt").exists() or (REPO_ROOT / "outputs/actformer_tiny/last.pt").exists()

    # 3) Probe comparison
    r = run([sys.executable, "-m", "src.probe.run_comparison", "--config", CONFIG])
    assert r.returncode == 0, r.stderr
    assert (REPO_ROOT / "outputs/probe_comparison_tiny/comparison_metrics.json").exists()
