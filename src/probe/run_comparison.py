"""
Probe comparison: train each probe type on ID, evaluate on ID test and each OOD set.
Supports raw_linear, raw_mlp, actformer_finetuned.
Usage: python -m src.probe.run_comparison --config configs/default.yaml
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.probe.train_probe import (
    get_pooled_features_raw,
    train_actformer_finetuned,
)
from src.probe.data import DocLevelProbeDataset, collate_doc_level_probe
from src.probe.model import build_probe
from src.actformer.model import load_actformer_with_head
from src.run_dir import ensure_unique_output_dir, resolve_actformer_checkpoint_dir
from src.utils.device import get_device
from src.utils.io import load_json, load_yaml, save_json
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb, load_dotenv_for_wandb


def _eval_actformer_finetuned_on_split(
    layer_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    batch_size: int,
    max_len: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run fine-tuned ActFormerWithHead on a split; return (y_pred, y_prob, y_true)."""
    ds = DocLevelProbeDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split=split,
        max_len=max_len,
    )
    if len(ds) == 0:
        return np.array([]), np.array([]), np.array([])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_doc_level_probe, num_workers=0)
    preds, probs, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, mask, yb in loader:
            xb, mask = xb.to(device), mask.to(device)
            logits = model(xb, mask=mask)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_prob = np.vstack(probs)
    y_true = np.concatenate(labels)
    return y_pred, y_prob, y_true


def run_probe_comparison(config: dict[str, Any]) -> dict[str, Any]:
    """Train each probe type on ID train; evaluate on ID test and each OOD. Return results dict."""
    set_seed(config.get("seed", 42))
    comp = config.get("comparison", {})
    activations = comp.get("activations", {})
    id_cfg = activations.get("id", {})
    ood_list = activations.get("ood", [])
    probe_types = comp.get("probe_types", ["raw_linear", "actformer_finetuned"])
    ckpt_path_from_config = Path(comp.get("actformer_checkpoint", "outputs/actformer/best.pt"))
    base_dir = ckpt_path_from_config.parent
    ckpt_dir = resolve_actformer_checkpoint_dir(base_dir)
    actformer_ckpt = ckpt_dir / "best.pt"
    pooling = comp.get("pooling", "mean")
    probe_train = comp.get("probe_train", {})
    metrics_list = comp.get("metrics", ["accuracy", "macro_f1", "auroc"])
    out_dir = Path(comp.get("output_dir", "outputs/probe_comparison"))
    out_dir = ensure_unique_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_memmap = Path(id_cfg.get("memmap_dir", "outputs/activations/id"))
    id_index = Path(id_cfg.get("index_path", id_memmap / "index.json"))
    layer_dir = id_memmap
    if not (layer_dir / "meta.json").exists():
        layer_index = config.get("model", {}).get("layer_index", 0)
        best_path = Path(config.get("layer_search", {}).get("layer_search_output", "outputs/best_layer.json"))
        if best_path.exists():
            layer_index = load_json(best_path).get("layer_index", layer_index)
        candidate = id_memmap / f"layer_{layer_index}"
        if (candidate / "meta.json").exists():
            layer_dir = candidate
        else:
            layer_dir = id_memmap.parent / f"layer_{layer_index}" if id_memmap.name != f"layer_{layer_index}" else id_memmap
    if not (layer_dir / "meta.json").exists():
        raise FileNotFoundError(f"ID activations not found: {layer_dir}")

    results: dict[str, Any] = {}
    n_classes = 2
    raw_clf = None
    raw_scaler = None
    probe = None
    device = get_device()
    af_cfg = config.get("actformer", {})
    finetune_cfg = comp.get("actformer_finetune", {})

    for probe_type in probe_types:
        if probe_type == "actformer_finetuned":
            if not actformer_ckpt.exists():
                results[probe_type] = {"id": {m: float("nan") for m in metrics_list}, "ood": {}}
                continue
            actformer_probe_path = out_dir / "actformer_probe.pt"
            if not actformer_probe_path.exists():
                train_actformer_finetuned(layer_dir, config, actformer_ckpt, n_classes, out_dir)
            ckpt = torch.load(actformer_probe_path, map_location="cpu", weights_only=True)
            pretrained_ckpt_path = ckpt.get("actformer_checkpoint", str(actformer_ckpt))
            meta = load_json(layer_dir / "meta.json")
            model = load_actformer_with_head(
                checkpoint_path=pretrained_ckpt_path,
                d_in=meta["hidden_size"],
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
            )
            model.load_state_dict(ckpt["model"], strict=True)
            model = model.to(device).eval()
            batch_size = int(finetune_cfg.get("batch_size", 16))
            max_len = finetune_cfg.get("max_len")
            y_pred, y_prob, y_id_test = _eval_actformer_finetuned_on_split(
                layer_dir, model, device, "test", batch_size, max_len
            )
            if len(y_id_test) == 0:
                id_metrics = {m: float("nan") for m in metrics_list}
            else:
                id_metrics = compute_metrics(y_id_test, y_pred, y_prob, metrics_list)
            ood_metrics = {}
            for ood_cfg in ood_list:
                name = ood_cfg.get("name", "ood")
                ood_dir = Path(str(ood_cfg.get("memmap_dir", ood_cfg.get("index_path", "."))).replace("/index.json", ""))
                if not (ood_dir / "meta.json").exists():
                    ood_metrics[name] = {m: float("nan") for m in metrics_list}
                    continue
                y_pred_o, y_prob_o, y_ood = _eval_actformer_finetuned_on_split(
                    ood_dir, model, device, "test", batch_size, max_len
                )
                if len(y_ood) == 0:
                    ood_metrics[name] = {m: float("nan") for m in metrics_list}
                else:
                    ood_metrics[name] = compute_metrics(y_ood, y_pred_o, y_prob_o, metrics_list)
            results[probe_type] = {"id": id_metrics, "ood": ood_metrics}
            continue

        parts = probe_type.split("_")
        if len(parts) >= 2:
            feature_source = parts[0]
            classifier = "_".join(parts[1:])
        else:
            feature_source = "raw"
            classifier = "linear"
        use_actformer = feature_source == "actformer"

        if use_actformer and not actformer_ckpt.exists():
            results[probe_type] = {"id": {m: float("nan") for m in metrics_list}, "ood": {}}
            continue

        if use_actformer:
            from src.probe.train_probe import get_pooled_features_actformer
            X_train, y_train = get_pooled_features_actformer(layer_dir, "train", actformer_ckpt, config, pooling)
            X_val, y_val = get_pooled_features_actformer(layer_dir, "val", actformer_ckpt, config, pooling)
            X_id_test, y_id_test = get_pooled_features_actformer(layer_dir, "test", actformer_ckpt, config, pooling)
        else:
            X_train, y_train = get_pooled_features_raw(layer_dir, "train")
            X_val, y_val = get_pooled_features_raw(layer_dir, "val")
            X_id_test, y_id_test = get_pooled_features_raw(layer_dir, "test")

        if len(X_train) == 0:
            results[probe_type] = {"id": {m: float("nan") for m in metrics_list}, "ood": {}}
            continue

        d_in = X_train.shape[1]
        raw_clf = None
        raw_scaler = None
        if classifier == "linear" and not use_actformer:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            if len(np.unique(y_train)) < 2:
                id_metrics = {m: float("nan") for m in metrics_list}
                raw_clf = None
            else:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                raw_clf = LogisticRegression(max_iter=500, random_state=42)
                raw_clf.fit(X_train_s, y_train)
                raw_scaler = scaler
                if len(X_id_test) > 0:
                    X_id_test_s = raw_scaler.transform(X_id_test)
                    y_pred = raw_clf.predict(X_id_test_s)
                    y_prob = raw_clf.predict_proba(X_id_test_s)[:, 1] if n_classes == 2 else raw_clf.predict_proba(X_id_test_s)
                    id_metrics = compute_metrics(y_id_test, y_pred, y_prob, metrics_list)
                else:
                    id_metrics = {m: float("nan") for m in metrics_list}
        else:
            probe_cfg = config.get("probe", {})
            tp_cfg = probe_cfg.get("transformer_probe", {})
            if classifier == "transformer":
                probe = build_probe("transformer", d_in, n_classes, d_model=tp_cfg.get("d_model", 256),
                    n_layers=tp_cfg.get("n_layers", 2), n_heads=tp_cfg.get("n_heads", 4),
                    ff_mult=tp_cfg.get("ff_mult", 4), dropout=tp_cfg.get("dropout", 0.1), pool=tp_cfg.get("pool", "mean"))
                X_train_t = torch.from_numpy(X_train).float().unsqueeze(1)
                X_id_test_t = torch.from_numpy(X_id_test).float().unsqueeze(1)
            else:
                probe = build_probe("mlp" if classifier == "mlp" else "linear", d_in, n_classes)
                X_train_t = torch.from_numpy(X_train).float()
                X_id_test_t = torch.from_numpy(X_id_test).float()
            probe = probe.to(device)
            opt = torch.optim.AdamW(probe.parameters(), lr=float(probe_train.get("lr", 1e-3)), weight_decay=float(probe_train.get("l2", 0.01)))
            for _ in range(int(probe_train.get("epochs", 50))):
                probe.train()
                perm = torch.randperm(len(X_train_t))
                for i in range(0, len(X_train_t), int(probe_train.get("batch_size", 32))):
                    idx = perm[i:i+int(probe_train.get("batch_size", 32))]
                    if len(idx) == 0:
                        continue
                    xb = X_train_t[idx].to(device)
                    yb = torch.from_numpy(y_train[idx.cpu().numpy()]).long().to(device)
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(probe(xb), yb)
                    loss.backward()
                    opt.step()
            probe.eval()
            with torch.no_grad():
                logits = probe(X_id_test_t.to(device))
                y_pred = logits.argmax(dim=1).cpu().numpy()
                y_prob = torch.softmax(logits, dim=1).cpu().numpy()
            id_metrics = compute_metrics(y_id_test, y_pred, y_prob, metrics_list)

        ood_metrics: dict[str, dict[str, float]] = {}
        for ood_cfg in ood_list:
            name = ood_cfg.get("name", "ood")
            ood_dir = Path(str(ood_cfg.get("memmap_dir", ood_cfg.get("index_path", "."))).replace("/index.json", ""))
            if not (ood_dir / "meta.json").exists():
                ood_metrics[name] = {m: float("nan") for m in metrics_list}
                continue
            if use_actformer:
                X_ood, y_ood = get_pooled_features_actformer(ood_dir, "test", actformer_ckpt, config, pooling)
            else:
                X_ood, y_ood = get_pooled_features_raw(ood_dir, "test")
            if len(X_ood) == 0:
                ood_metrics[name] = {m: float("nan") for m in metrics_list}
                continue
            if classifier == "linear" and not use_actformer:
                if raw_clf is not None:
                    X_ood_s = raw_scaler.transform(X_ood)
                    y_pred = raw_clf.predict(X_ood_s)
                    y_prob = raw_clf.predict_proba(X_ood_s)[:, 1] if n_classes == 2 else raw_clf.predict_proba(X_ood_s)
                    ood_metrics[name] = compute_metrics(y_ood, y_pred, y_prob, metrics_list)
                else:
                    ood_metrics[name] = {m: float("nan") for m in metrics_list}
            else:
                X_ood_t = torch.from_numpy(X_ood).float().unsqueeze(1) if classifier == "transformer" else torch.from_numpy(X_ood).float()
                with torch.no_grad():
                    logits = probe(X_ood_t.to(device))
                    y_pred = logits.argmax(dim=1).cpu().numpy()
                    y_prob = torch.softmax(logits, dim=1).cpu().numpy()
                ood_metrics[name] = compute_metrics(y_ood, y_pred, y_prob, metrics_list)
        results[probe_type] = {"id": id_metrics, "ood": ood_metrics}

    save_json(results, out_dir / "comparison_metrics.json")
    load_dotenv_for_wandb(config)
    wandb_cfg = config.get("wandb", {})
    if init_wandb(
        project=wandb_cfg.get("comparison_project", "probe-comparison"),
        name="comparison",
        config={"probe_types": probe_types, "output_dir": str(out_dir)},
        entity=wandb_cfg.get("entity"),
    ):
        try:
            import wandb
            for pt, data in results.items():
                for metric, val in data.get("id", {}).items():
                    wandb.log({f"id/{pt}/{metric}": val})
                for ood_name, ood_data in data.get("ood", {}).items():
                    for metric, val in ood_data.items():
                        wandb.log({f"ood/{pt}/{ood_name}/{metric}": val})
        except Exception:
            pass
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    results = run_probe_comparison(config)
    print("[comparison] Results:", results)
    print(f"[comparison] Saved to {config.get('comparison', {}).get('output_dir', 'outputs/probe_comparison')}")


if __name__ == "__main__":
    main()
