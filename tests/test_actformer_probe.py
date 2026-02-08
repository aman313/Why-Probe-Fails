"""
Unit tests for ActFormerWithHead and doc-level probe dataset.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.actformer.model import ActFormer, ActFormerWithHead, load_actformer_with_head
from src.probe.data import DocLevelProbeDataset, collate_doc_level_probe


def test_actformer_with_head_forward_no_mlm_mask():
    """ActFormerWithHead forward uses (x, mask) only; no mlm_mask."""
    B, T, d_in, d_model, n_classes = 2, 5, 8, 16, 2
    actformer = ActFormer(d_in=d_in, d_model=d_model, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0)
    model = ActFormerWithHead(actformer, n_classes=n_classes, pool="mean", freeze_body=False)
    x = torch.randn(B, T, d_in)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1:] = False
    logits = model(x, mask=mask)
    assert logits.shape == (B, n_classes)
    loss = logits.sum()
    loss.backward()


def test_actformer_with_head_load_checkpoint():
    """Load pretrained ActFormer from checkpoint and build ActFormerWithHead."""
    d_in, d_model, n_classes = 8, 16, 2
    actformer = ActFormer(d_in=d_in, d_model=d_model, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        torch.save({"model": actformer.state_dict(), "step": 0}, ckpt_path)
        model = load_actformer_with_head(
            checkpoint_path=ckpt_path,
            d_in=d_in,
            d_model=d_model,
            n_layers=1,
            n_heads=2,
            n_classes=n_classes,
            ff_mult=2,
            dropout=0.0,
            pool="mean",
            freeze_body=False,
        )
        x = torch.randn(2, 4, d_in)
        logits = model(x, mask=torch.ones(2, 4, dtype=torch.bool))
        assert logits.shape == (2, n_classes)
    finally:
        Path(ckpt_path).unlink(missing_ok=True)


def test_actformer_with_head_freeze_body_leaves_body_unchanged():
    """When freeze_body=True, ActFormer parameters have no grad and are unchanged after step."""
    B, T, d_in, d_model, n_classes = 2, 4, 8, 16, 2
    actformer = ActFormer(d_in=d_in, d_model=d_model, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0)
    body_params_before = {k: v.clone() for k, v in actformer.named_parameters()}
    model = ActFormerWithHead(actformer, n_classes=n_classes, pool="mean", freeze_body=True)
    x = torch.randn(B, T, d_in)
    y = torch.randint(0, n_classes, (B,))
    logits = model(x, mask=torch.ones(B, T, dtype=torch.bool))
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    for name, p in actformer.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"Body param {name} should not receive gradients"
        assert torch.allclose(p, body_params_before[name]), f"Body param {name} should be unchanged"


def test_doc_level_probe_dataset_yields_seq_label_per_doc(tmp_path):
    """DocLevelProbeDataset yields (seq, label) per doc; labels match index."""
    d, n_docs = 4, 3
    total = 0
    index_entries = []
    for doc_id in range(n_docs):
        L = 5 + doc_id * 2
        index_entries.append({"doc_id": doc_id, "split": "train", "start": total, "length": L, "label": doc_id % 2})
        total += L
    meta = {"total_tokens": total, "hidden_size": d, "dtype": "float32"}
    arr = np.random.randn(total, d).astype(np.float32)
    memmap_path = tmp_path / "activations.dat"
    arr.tofile(str(memmap_path))
    (tmp_path / "index.json").write_text(json.dumps(index_entries))
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    mean = np.zeros(d, dtype=np.float32)
    std = np.ones(d, dtype=np.float32)
    np.save(tmp_path / "train_mean.npy", mean)
    np.save(tmp_path / "train_std.npy", std)
    ds = DocLevelProbeDataset(
        tmp_path / "activations.dat",
        tmp_path / "index.json",
        tmp_path / "meta.json",
        tmp_path / "train_mean.npy",
        tmp_path / "train_std.npy",
        split="train",
    )
    assert len(ds) == n_docs
    for i in range(len(ds)):
        seq, label = ds[i]
        assert seq.dim() == 2 and seq.shape[1] == d
        assert label == index_entries[i]["label"]
        expected_len = index_entries[i]["length"]
        assert seq.shape[0] == expected_len


def test_collate_doc_level_probe_shapes_and_mask():
    """Collate produces (x, mask, labels) with correct shapes; mask True only on valid positions."""
    d = 4
    batch = [
        (torch.randn(5, d), 0),
        (torch.randn(3, d), 1),
        (torch.randn(7, d), 0),
    ]
    x, mask, labels = collate_doc_level_probe(batch)
    assert x.shape == (3, 7, d)
    assert mask.shape == (3, 7)
    assert labels.shape == (3,)
    assert mask[0, :5].all() and not mask[0, 5:].any()
    assert mask[1, :3].all() and not mask[1, 3:].any()
    assert mask[2, :7].all()
    assert labels.tolist() == [0, 1, 0]


def test_train_actformer_finetuned_saves_checkpoint_and_metrics(tmp_path, monkeypatch):
    """Run fine-tune for 1 epoch on tiny data; assert checkpoint and metrics JSON saved; train-only fit."""
    monkeypatch.setattr("src.probe.train_probe.get_device", lambda: torch.device("cpu"))
    from src.probe.train_probe import train_actformer_finetuned

    d, n_train, n_val, n_test = 4, 4, 2, 2
    total = 0
    index_entries = []
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        for i in range(n):
            doc_id = len(index_entries)
            L = 6
            index_entries.append({"doc_id": doc_id, "split": split, "start": total, "length": L, "label": i % 2})
            total += L
    meta = {"total_tokens": total, "hidden_size": d, "dtype": "float32"}
    arr = np.random.randn(total, d).astype(np.float32)
    (tmp_path / "activations.dat").write_bytes(arr.tobytes())
    (tmp_path / "index.json").write_text(json.dumps(index_entries))
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    np.save(tmp_path / "train_mean.npy", np.zeros(d, dtype=np.float32))
    np.save(tmp_path / "train_std.npy", np.ones(d, dtype=np.float32))
    actformer = ActFormer(d_in=d, d_model=8, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0)
    ckpt_path = tmp_path / "actformer_best.pt"
    torch.save({"model": actformer.state_dict(), "step": 0}, ckpt_path)
    config = {
        "seed": 42,
        "actformer": {"d_model": 8, "n_layers": 1, "n_heads": 2, "ff_mult": 2, "dropout": 0.0, "loss_type": "mse", "causal": True},
        "comparison": {"actformer_finetune": {"lr": 1e-3, "epochs": 1, "batch_size": 2, "freeze_body": False, "pooling": "mean"}},
    }
    out_dir = tmp_path / "probe_out"
    metrics = train_actformer_finetuned(tmp_path, config, ckpt_path, n_classes=2, out_dir=out_dir)
    assert (out_dir / "actformer_probe.pt").exists()
    assert (out_dir / "probe_metrics.json").exists()
    assert "accuracy" in metrics and "macro_f1" in metrics


def test_run_comparison_raw_linear_and_actformer_finetuned(tmp_path, monkeypatch):
    """Run comparison with probe_types [raw_linear, actformer_finetuned]; assert metrics and table."""
    monkeypatch.setattr("src.probe.train_probe.get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("src.probe.run_comparison.get_device", lambda: torch.device("cpu"))
    from src.probe.run_comparison import run_probe_comparison
    from src.utils.io import load_json

    d, n_train, n_val, n_test = 4, 4, 2, 2
    total = 0
    index_entries = []
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        for i in range(n):
            doc_id = len(index_entries)
            L = 6
            index_entries.append({"doc_id": doc_id, "split": split, "start": total, "length": L, "label": i % 2})
            total += L
    meta = {"total_tokens": total, "hidden_size": d, "dtype": "float32"}
    arr = np.random.randn(total, d).astype(np.float32)
    (tmp_path / "activations.dat").write_bytes(arr.tobytes())
    (tmp_path / "index.json").write_text(json.dumps(index_entries))
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    np.save(tmp_path / "train_mean.npy", np.zeros(d, dtype=np.float32))
    np.save(tmp_path / "train_std.npy", np.ones(d, dtype=np.float32))
    actformer = ActFormer(d_in=d, d_model=8, n_layers=1, n_heads=2, ff_mult=2, dropout=0.0)
    ckpt_path = tmp_path / "actformer_best.pt"
    torch.save({"model": actformer.state_dict(), "step": 0}, ckpt_path)
    out_dir = tmp_path / "probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": 42,
        "actformer": {"d_model": 8, "n_layers": 1, "n_heads": 2, "ff_mult": 2, "dropout": 0.0, "loss_type": "mse", "causal": True},
        "comparison": {
            "activations": {"id": {"memmap_dir": str(tmp_path), "index_path": str(tmp_path / "index.json")}, "ood": []},
            "probe_types": ["raw_linear", "actformer_finetuned"],
            "actformer_checkpoint": str(ckpt_path),
            "actformer_finetune": {"lr": 1e-3, "epochs": 1, "batch_size": 2, "freeze_body": False, "pooling": "mean"},
            "probe_train": {"lr": 1e-3, "epochs": 1, "batch_size": 2, "l2": 0.01},
            "metrics": ["accuracy", "macro_f1", "auroc"],
            "output_dir": str(out_dir),
        },
        "model": {"layer_index": 0},
        "layer_search": {},
    }
    results = run_probe_comparison(config)
    assert "raw_linear" in results
    assert "actformer_finetuned" in results
    assert "id" in results["raw_linear"] and "id" in results["actformer_finetuned"]
    for pt in ["raw_linear", "actformer_finetuned"]:
        for m in ["accuracy", "macro_f1"]:
            assert m in results[pt]["id"]
            assert isinstance(results[pt]["id"][m], (int, float))
    assert (out_dir / "comparison_metrics.json").exists()
    saved = load_json(out_dir / "comparison_metrics.json")
    assert "raw_linear" in saved and "actformer_finetuned" in saved
