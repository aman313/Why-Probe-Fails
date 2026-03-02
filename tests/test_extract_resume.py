"""Tests for incremental extraction: progress save/load, data loading parity, chunk_size and batch_size."""
from pathlib import Path

import numpy as np
import pytest

from src.extract_activations import (
    _entry_idx_at_write_offset,
    _load_progress,
    _rebuild_welford_from_memmap,
    _save_progress,
)
from src.utils.data_loading import (
    build_doc_id_to_split,
    iter_docs,
    load_docs_metadata,
    load_and_split,
    split_docs,
)


def test_entry_idx_at_write_offset():
    """_entry_idx_at_write_offset returns count of entries fully written."""
    index_entries = [
        {"start": 0, "length": 10},
        {"start": 10, "length": 5},
        {"start": 15, "length": 20},
        {"start": 35, "length": 3},
    ]
    assert _entry_idx_at_write_offset(index_entries, 0) == 0
    assert _entry_idx_at_write_offset(index_entries, 10) == 1
    assert _entry_idx_at_write_offset(index_entries, 15) == 2
    assert _entry_idx_at_write_offset(index_entries, 35) == 3
    assert _entry_idx_at_write_offset(index_entries, 38) == 4
    assert _entry_idx_at_write_offset(index_entries, 100) == 4


def test_load_docs_metadata_and_iter_docs_match_full_load(tmp_path):
    """iter_docs with split=all yields same count and doc_ids as load_and_split."""
    (tmp_path / "cat_a").mkdir()
    (tmp_path / "cat_b").mkdir()
    # Need enough docs so stratified train/val/test each have >= 2 per class (sklearn requirement)
    (tmp_path / "cat_a" / "1.csv").write_text("prompt\n" + "\n".join(["a" + str(i) for i in range(4)]) + "\n")
    (tmp_path / "cat_b" / "2.csv").write_text("prompt\n" + "\n".join(["b" + str(i) for i in range(4)]) + "\n")
    categories = ["cat_a", "cat_b"]
    label_map = {"cat_a": 0, "cat_b": 1}
    full_df, train_df, val_df, test_df = load_and_split(
        tmp_path, categories, label_map, split_by_file=False,
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42,
    )
    doc_id_to_split = build_doc_id_to_split(train_df, val_df, test_df)
    streamed = list(iter_docs(tmp_path, categories, label_map, doc_id_to_split, split="all"))
    assert len(streamed) == len(full_df)
    streamed_ids = [doc_id for (_, doc_id, _, _, _) in streamed]
    assert streamed_ids == full_df["doc_id"].tolist()


def test_progress_save_and_load(tmp_path):
    """_save_progress and _load_progress round-trip."""
    _save_progress(tmp_path, {"pass": 1, "last_doc_index": 5, "total_tokens": 100})
    p = _load_progress(tmp_path)
    assert p is not None and p["pass"] == 1 and p["last_doc_index"] == 5
    assert _load_progress(tmp_path / "nonexistent") is None


def test_rebuild_welford_from_memmap(tmp_path):
    """_rebuild_welford_from_memmap recovers Welford state from memmap."""
    hidden_size, total_tokens = 4, 10
    memmap_path = tmp_path / "act.dat"
    arr = np.memmap(str(memmap_path), dtype=np.float32, mode="w+", shape=(total_tokens, hidden_size))
    arr[:] = np.random.randn(total_tokens, hidden_size).astype(np.float32)
    arr.flush()
    del arr
    index_entries = [
        {"split": "train", "start": 0, "length": 4},
        {"split": "val", "start": 4, "length": 2},
        {"split": "train", "start": 6, "length": 4},
    ]
    activations = np.memmap(str(memmap_path), dtype=np.float32, mode="r", shape=(total_tokens, hidden_size))
    n_train, mean, M2 = _rebuild_welford_from_memmap(activations, index_entries, 10, hidden_size)
    del activations
    assert n_train == 8
    assert mean.shape == (hidden_size,) and M2.shape == (hidden_size,)


def test_progress_with_last_chunk_in_doc_round_trip(tmp_path):
    """Progress with last_doc_index and last_chunk_in_doc round-trips for Pass 2 resume."""
    _save_progress(tmp_path, {
        "pass": 2,
        "write_offset": 100,
        "last_doc_index": 3,
        "last_chunk_in_doc": 2,
    })
    p = _load_progress(tmp_path)
    assert p["pass"] == 2 and p["write_offset"] == 100
    assert p["last_doc_index"] == 3 and p["last_chunk_in_doc"] == 2


@pytest.mark.slow
def test_extract_with_chunk_size_and_batch_size(tmp_path, monkeypatch):
    """Extraction runs with chunk_size and batch_size and produces valid activations."""
    from src.extract_activations import extract_activations
    from src.utils.io import load_json

    data_dir = tmp_path / "data"
    (data_dir / "cat_a").mkdir(parents=True)
    (data_dir / "cat_b").mkdir(parents=True)
    (data_dir / "cat_a" / "1.csv").write_text("prompt\n" + "\n".join(["a" + str(i) for i in range(4)]) + "\n")
    (data_dir / "cat_b" / "2.csv").write_text("prompt\n" + "\n".join(["b" + str(i) for i in range(4)]) + "\n")

    out_dir = tmp_path / "out"
    config = {
        "seed": 42,
        "data": {
            "data_dir": str(data_dir),
            "categories": ["cat_a", "cat_b"],
            "label_map": {"cat_a": 0, "cat_b": 1},
            "split_by_file": False,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
        },
        "model": {"base_model_name": "distilgpt2"},
        "extraction": {
            "max_seq_len": 64,
            "stride": 32,
            "chunk_size": 5,
            "batch_size": 2,
            "activation_dtype": "float32",
            "memmap_dir": str(out_dir),
            "limit_docs": 8,
        },
    }
    monkeypatch.setattr("src.extract_activations.get_device", lambda: __import__("torch").device("cpu"))
    extract_activations(config, "all", 0, for_pretrain=False)

    meta = load_json(out_dir / "layer_0" / "meta.json")
    index = load_json(out_dir / "layer_0" / "index.json")
    assert meta["total_tokens"] > 0
    assert meta["hidden_size"] > 0
    assert len(index) >= 1
    assert (out_dir / "layer_0" / "activations.dat").exists()
    assert not (out_dir / "layer_0" / "progress.json").exists()
