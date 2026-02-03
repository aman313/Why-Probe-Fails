"""
Layer search: run linear probe on each candidate layer; pick best by val metric.
Usage: python -m src.layer_search --config configs/default.yaml
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.extract_activations import extract_activations
from src.utils.dtypes import resolve_numpy_dtype
from src.utils.io import ensure_dir, load_json, load_yaml, save_json
from src.utils.seed import set_seed


def load_pooled_features_per_doc(
    memmap_path: Path,
    index_path: Path,
    meta_path: Path,
    mean_path: Path | None,
    std_path: Path | None,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load activations from memmap + index; pool per doc (mean over tokens); return (X, y).
    X: (n_docs, d), y: (n_docs,). Only entries with split=split.
    """
    index = load_json(index_path)
    meta = load_json(meta_path)
    total_tokens = meta["total_tokens"]
    hidden_size = meta["hidden_size"]
    dtype = resolve_numpy_dtype(meta.get("dtype", "float32"))
    arr = np.memmap(str(memmap_path), dtype=dtype, mode="r", shape=(total_tokens, hidden_size))

    # Group by doc_id for this split
    by_doc: dict[int, list[tuple[int, int]]] = {}
    labels: dict[int, int] = {}
    for e in index:
        if e["split"] != split:
            continue
        doc_id = e["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = []
            labels[doc_id] = e["label"]
        by_doc[doc_id].append((e["start"], e["length"]))

    doc_ids = sorted(by_doc.keys())
    if not doc_ids:
        return np.zeros((0, hidden_size), dtype=np.float32), np.array([], dtype=np.int64)
    vectors = []
    y_list = []
    for doc_id in doc_ids:
        chunks = by_doc[doc_id]
        parts = [np.array(arr[s : s + L]) for s, L in chunks]
        pooled = np.concatenate(parts, axis=0).mean(axis=0)
        vectors.append(pooled)
        y_list.append(labels[doc_id])
    X = np.stack(vectors)
    y = np.array(y_list)
    if mean_path is not None and mean_path.exists() and std_path is not None and std_path.exists():
        mean = np.load(mean_path)
        std = np.load(std_path)
        std = np.where(std < 1e-8, 1.0, std)
        X = (X - mean) / std
    return X, y


def run_probe_one_layer(
    layer_dir: Path,
    metric: str,
) -> float:
    """Run linear probe on train, return val metric (macro_f1 or accuracy)."""
    index_path = layer_dir / "index.json"
    meta_path = layer_dir / "meta.json"
    mean_path = layer_dir / "train_mean.npy"
    std_path = layer_dir / "train_std.npy"
    memmap_path = layer_dir / "activations.dat"
    if not memmap_path.exists() or not index_path.exists():
        return float("-inf")
    X_train, y_train = load_pooled_features_per_doc(
        memmap_path, index_path, meta_path, mean_path, std_path, "train"
    )
    X_val, y_val = load_pooled_features_per_doc(
        memmap_path, index_path, meta_path, mean_path, std_path, "val"
    )
    if len(X_train) == 0 or len(X_val) == 0:
        return float("-inf")
    if len(np.unique(y_train)) < 2:
        return 0.0
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    if metric == "macro_f1":
        return float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    return float((y_pred == y_val).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer search: pick best layer by linear probe val metric")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run_extraction", action="store_true", help="Run extraction for each layer if memmap missing")
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    ext_cfg = config.get("extraction", {})
    ls_cfg = config.get("layer_search", {})
    model_cfg = config.get("model", {})
    memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))
    layers = ls_cfg.get("layer_search_layers", [0, 1, 2, 3, 4, 5])
    metric = ls_cfg.get("layer_search_metric", "macro_f1")
    out_path = Path(ls_cfg.get("layer_search_output", "outputs/best_layer.json"))

    scores: dict[int, float] = {}
    for layer_idx in layers:
        layer_dir = memmap_dir / f"layer_{layer_idx}"
        if args.run_extraction and (not layer_dir.exists() or not (layer_dir / "index.json").exists()):
            print(f"[layer_search] Extracting layer {layer_idx} ...")
            extract_activations(config, "all", layer_idx)
        score = run_probe_one_layer(layer_dir, metric)
        scores[layer_idx] = score
        print(f"  layer {layer_idx}: {metric}={score:.4f}")

    best_layer = max(scores, key=scores.get)
    ensure_dir(out_path.parent)
    save_json(
        {"layer_index": best_layer, "metric": metric, "scores": scores},
        out_path,
    )
    print(f"[layer_search] Best layer: {best_layer} (saved to {out_path})")


if __name__ == "__main__":
    main()
