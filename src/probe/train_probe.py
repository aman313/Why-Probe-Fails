"""
Train classifier probe on pooled activations (raw) or fine-tune ActFormer probe.
Usage: python -m src.probe.train_probe --config configs/default.yaml [--probe_type raw_linear|raw_mlp|actformer_finetuned]
Legacy: --use_actformer true runs frozen ActFormer + linear (actformer_linear); false = raw_linear.
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.actformer.model import ActFormer, load_actformer_with_head
from src.probe.data import DocLevelProbeDataset, collate_doc_level_probe
from src.probe.model import build_probe
from src.run_dir import ensure_unique_output_dir, resolve_actformer_checkpoint_dir
from src.utils.device import get_device
from src.utils.dtypes import resolve_numpy_dtype
from src.utils.io import load_json, load_yaml, save_json
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb, load_dotenv_for_wandb

# Reuse layer_search's pooled-feature loading for raw activations
from src.layer_search import load_pooled_features_per_doc


def get_pooled_features_raw(
    layer_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled raw activations per doc for one split. Returns (X, y)."""
    return load_pooled_features_per_doc(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split,
    )


def get_pooled_features_actformer(
    layer_dir: Path,
    split: str,
    actformer_ckpt: Path,
    config: dict,
    pooling: str = "mean",
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run activations through ActFormer, pool hidden states per doc. Returns (X, y)."""
    device = device or get_device()
    meta = load_json(layer_dir / "meta.json")
    index = load_json(layer_dir / "index.json")
    af_cfg = config.get("actformer", {})
    actformer = ActFormer(
        d_in=meta["hidden_size"],
        d_model=af_cfg.get("d_model", 256),
        n_layers=af_cfg.get("n_layers", 2),
        n_heads=af_cfg.get("n_heads", 4),
        ff_mult=af_cfg.get("ff_mult", 4),
        dropout=0.0,
        loss_type=af_cfg.get("loss_type", "mse"),
        causal=af_cfg.get("causal", True),
    )
    ckpt = torch.load(actformer_ckpt, map_location="cpu", weights_only=True)
    actformer.load_state_dict(ckpt["model"])
    actformer = actformer.to(device).eval()
    dtype = resolve_numpy_dtype(meta.get("dtype", "float32"))
    arr = np.memmap(
        str(layer_dir / "activations.dat"),
        dtype=dtype,
        mode="r",
        shape=(meta["total_tokens"], meta["hidden_size"]),
    )
    mean = np.load(layer_dir / "train_mean.npy")
    std = np.load(layer_dir / "train_std.npy")
    std = np.where(std < 1e-8, 1.0, std)
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
    vectors = []
    y_list = []
    with torch.no_grad():
        for doc_id in doc_ids:
            chunks = by_doc[doc_id]
            parts = [np.array(arr[s : s + L]) for s, L in chunks]
            seq = np.concatenate(parts, axis=0).astype(np.float32)
            seq = ((seq - mean) / std).astype(np.float32)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
            _, _, hidden = actformer(x, mask)
            hidden = hidden[0]
            if pooling == "last":
                vec = hidden[-1].cpu().numpy()
            else:
                vec = hidden.mean(dim=0).cpu().numpy()
            vectors.append(vec)
            y_list.append(labels[doc_id])
    return np.stack(vectors), np.array(y_list)


def train_actformer_finetuned(
    layer_dir: Path,
    config: dict,
    actformer_ckpt: Path,
    n_classes: int,
    out_dir: Path,
) -> dict[str, float]:
    """Fine-tune ActFormer with classification head on doc-level (seq, label). Train on train split only; return test metrics."""
    set_seed(config.get("seed", 42))
    af_cfg = config.get("actformer", {})
    finetune_cfg = config.get("comparison", {}).get("actformer_finetune", config.get("probe", {}).get("actformer_finetune", {}))
    if not finetune_cfg:
        finetune_cfg = config.get("probe", {}).get("probe_train", {})
    meta = load_json(layer_dir / "meta.json")
    d_in = meta["hidden_size"]
    max_len = finetune_cfg.get("max_len")
    train_ds = DocLevelProbeDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split="train",
        max_len=max_len,
    )
    if len(train_ds) == 0:
        print("[probe] No train data for actformer_finetuned. Exiting.")
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    device = get_device()
    use_lora = bool(finetune_cfg.get("use_lora", False))
    model = load_actformer_with_head(
        checkpoint_path=str(actformer_ckpt),
        d_in=d_in,
        d_model=af_cfg.get("d_model", 256),
        n_layers=af_cfg.get("n_layers", 2),
        n_heads=af_cfg.get("n_heads", 4),
        n_classes=n_classes,
        ff_mult=af_cfg.get("ff_mult", 4),
        dropout=af_cfg.get("dropout", 0.1),
        loss_type=af_cfg.get("loss_type", "mse"),
        causal=af_cfg.get("causal", True),
        pool=finetune_cfg.get("pooling", "mean"),
        freeze_body=finetune_cfg.get("freeze_body", False),
        use_lora=use_lora,
        lora_r=int(finetune_cfg.get("lora_r", 8)),
        lora_alpha=int(finetune_cfg.get("lora_alpha", 16)),
        lora_dropout=float(finetune_cfg.get("lora_dropout", 0.05)),
        lora_target_modules=finetune_cfg.get("lora_target_modules"),
    )
    model = model.to(device)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(finetune_cfg.get("lr", 1e-4)),
        weight_decay=float(finetune_cfg.get("l2", finetune_cfg.get("wd", 0.01))),
    )
    batch_size = int(finetune_cfg.get("batch_size", 16))
    epochs = int(finetune_cfg.get("epochs", 10))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_doc_level_probe,
        num_workers=0,
    )
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, mask, yb in tqdm(train_loader, desc=f"Epoch {epoch}", leave=True):
            xb, mask, yb = xb.to(device), mask.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb, mask=mask)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), finetune_cfg.get("grad_clip", 1.0))
            opt.step()
            train_loss_sum += loss.item()
            n_batches += 1
        try:
            import wandb
            if wandb.run is not None:
                mean_loss = train_loss_sum / max(n_batches, 1)
                wandb.log({"train_loss": mean_loss}, step=epoch)
        except Exception:
            pass
    if use_lora:
        model.actformer = model.actformer.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "actformer_checkpoint": str(actformer_ckpt)}, out_dir / "actformer_probe.pt")
    # Evaluate on val and test
    val_ds = DocLevelProbeDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split="val",
        max_len=max_len,
    )
    test_ds = DocLevelProbeDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split="test",
        max_len=max_len,
    )
    model.eval()
    def eval_split(ds, desc="Eval"):
        if len(ds) == 0:
            return np.array([]), np.array([]), np.array([])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_doc_level_probe, num_workers=0)
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for xb, mask, yb in tqdm(loader, desc=desc, leave=False):
                xb, mask = xb.to(device), mask.to(device)
                logits = model(xb, mask=mask)
                preds.append(logits.argmax(dim=1).cpu().numpy())
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels.append(yb.numpy())
        return np.concatenate(preds), np.vstack(probs), np.concatenate(labels)
    y_pred_test, y_prob_test, y_test = eval_split(test_ds, desc="Test")
    if len(y_test) == 0:
        metrics = {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    else:
        metrics = compute_metrics(y_test, y_pred_test, y_prob_test, ["accuracy", "macro_f1", "auroc"])
    try:
        import wandb
        if wandb.run is not None:
            y_pred_val, y_prob_val, y_val = eval_split(val_ds, desc="Val")
            if len(y_val) > 0:
                val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val, ["accuracy", "macro_f1", "auroc"])
                wandb.log({"val/accuracy": val_metrics["accuracy"], "val/macro_f1": val_metrics["macro_f1"], "val/auroc": val_metrics["auroc"]})
            wandb.log({"test/accuracy": metrics["accuracy"], "test/macro_f1": metrics["macro_f1"], "test/auroc": metrics["auroc"]})
    except Exception:
        pass
    save_json(metrics, out_dir / "probe_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--probe_type", type=str, default="raw_linear", choices=["raw_linear", "raw_mlp", "actformer_finetuned"])
    parser.add_argument("--use_actformer", type=str, default="", choices=["true", "false", ""], help="Legacy: true=actformer_linear, false=raw_linear")
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    ext_cfg = config.get("extraction", {})
    model_cfg = config.get("model", {})
    probe_cfg = config.get("probe", {})
    memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))
    layer_index = model_cfg.get("layer_index", 0)
    best_layer_path = Path(config.get("layer_search", {}).get("layer_search_output", "outputs/best_layer.json"))
    if best_layer_path.exists():
        best = load_json(best_layer_path)
        layer_index = int(best["layer_index"]) if isinstance(best["layer_index"], int) else best["layer_index"]
    layer_dir = memmap_dir / f"layer_{layer_index}"
    if args.use_actformer:
        probe_type = "actformer_finetuned" if args.use_actformer == "true" else "raw_linear"
    else:
        probe_type = args.probe_type
    pooling = probe_cfg.get("pooling", "mean")
    probe_model_type = probe_cfg.get("probe_model", "linear")
    train_cfg = probe_cfg.get("probe_train", {})
    label_map = config.get("data", {}).get("label_map", {})
    n_classes = len(set(label_map.values())) if label_map else 2
    out_dir = Path(config.get("comparison", {}).get("output_dir", "outputs/probe_comparison"))
    out_dir = ensure_unique_output_dir(out_dir)

    load_dotenv_for_wandb(config)
    wandb_cfg = config.get("wandb", {})
    init_wandb(
        project=wandb_cfg.get("probe_project", "probe-train"),
        name=f"probe_{probe_type}",
        config={
            "probe_type": probe_type,
            "layer_index": layer_index,
            "n_classes": n_classes,
            "lr": train_cfg.get("lr"),
            "epochs": train_cfg.get("epochs"),
            "batch_size": train_cfg.get("batch_size"),
        },
        entity=wandb_cfg.get("entity"),
    )

    if probe_type == "actformer_finetuned":
        pretrain_out = Path(config.get("pretrain", {}).get("output_dir", "outputs/actformer"))
        ckpt_dir = resolve_actformer_checkpoint_dir(pretrain_out)
        actformer_ckpt = ckpt_dir / "best.pt"
        if not actformer_ckpt.exists():
            actformer_ckpt = Path(config.get("pretrain", {}).get("output_dir", "outputs/actformer_tiny")) / "best.pt"
        if not actformer_ckpt.exists():
            print(f"[probe] ActFormer checkpoint not found: {actformer_ckpt}. Run pretrain first.")
            return
        metrics = train_actformer_finetuned(layer_dir, config, actformer_ckpt, n_classes, out_dir)
        print(f"[probe] actformer_finetuned metrics: {metrics} saved to {out_dir}")
        return

    # Raw baselines: pooled features
    X_train, y_train = get_pooled_features_raw(layer_dir, "train")
    X_val, y_val = get_pooled_features_raw(layer_dir, "val")
    X_test, y_test = get_pooled_features_raw(layer_dir, "test")
    d_in = X_train.shape[1]

    if len(X_train) == 0:
        print("[probe] No train data. Exiting.")
        return
    if len(np.unique(y_train)) < 2:
        print("[probe] Only one class in train; probe will predict that class only.")

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    X_test_t = torch.from_numpy(X_test).float()
    if probe_model_type == "transformer":
        tp_cfg = probe_cfg.get("transformer_probe", {})
        probe = build_probe(
            "transformer",
            d_in,
            n_classes,
            d_model=tp_cfg.get("d_model", 256),
            n_layers=tp_cfg.get("n_layers", 2),
            n_heads=tp_cfg.get("n_heads", 4),
            ff_mult=tp_cfg.get("ff_mult", 4),
            dropout=tp_cfg.get("dropout", 0.1),
            pool=tp_cfg.get("pool", "mean"),
        )
        X_train_t = X_train_t.unsqueeze(1)
        X_val_t = X_val_t.unsqueeze(1)
        X_test_t = X_test_t.unsqueeze(1)
    else:
        probe = build_probe("mlp" if probe_type == "raw_mlp" else "linear", d_in, n_classes)
    device = get_device()
    probe = probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("l2", 0.01)))
    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 32))
    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train_t), device=X_train_t.device)
        for i in tqdm(range(0, len(X_train_t), batch_size), desc=f"Epoch {epoch}", leave=True):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx].to(device)
            yb = torch.from_numpy(np.atleast_1d(y_train[idx.cpu().numpy()])).long().to(device)
            opt.zero_grad()
            logits = probe(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
        if len(X_val_t) > 0:
            probe.eval()
            with torch.no_grad():
                logits = probe(X_val_t.to(device))
                pred = logits.argmax(dim=1).cpu().numpy()
                prob = torch.softmax(logits, dim=1).cpu().numpy()
            val_metrics = compute_metrics(y_val, pred, prob, ["accuracy", "macro_f1", "auroc"])
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"val/accuracy": val_metrics["accuracy"], "val/macro_f1": val_metrics["macro_f1"], "val/auroc": val_metrics["auroc"]}, step=epoch)
            except Exception:
                pass

    probe.eval()
    with torch.no_grad():
        logits_test = probe(X_test_t.to(device))
        y_pred = logits_test.argmax(dim=1).cpu().numpy()
        y_prob = torch.softmax(logits_test, dim=1).cpu().numpy()
    if len(y_test) == 0:
        metrics = {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    else:
        metrics = compute_metrics(y_test, y_pred, y_prob, ["accuracy", "macro_f1", "auroc"])
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"test/accuracy": metrics["accuracy"], "test/macro_f1": metrics["macro_f1"], "test/auroc": metrics["auroc"]})
    except Exception:
        pass
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / "probe_metrics.json")
    torch.save(probe.state_dict(), out_dir / "probe.pt")
    print(f"[probe] Metrics: {metrics} saved to {out_dir}")


if __name__ == "__main__":
    main()
