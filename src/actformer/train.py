"""
ActFormer pretraining: next-step activation prediction.
Usage: python -m src.actformer.train --config configs/default.yaml
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.actformer.data import ActivationSequenceDataset, collate_activation_sequences
from src.actformer.model import ActFormer
from src.utils.device import get_device
from src.utils.io import ensure_dir, load_json, load_yaml
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    ext_cfg = config.get("extraction", {})
    model_cfg = config.get("model", {})
    af_cfg = config.get("actformer", {})
    train_cfg = config.get("pretrain", {})
    memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))
    out_dir = Path(train_cfg.get("output_dir", "outputs/actformer"))
    ensure_dir(out_dir)
    layer_index = model_cfg.get("layer_index", 0)
    best_path = Path(config.get("layer_search", {}).get("layer_search_output", "outputs/best_layer.json"))
    if best_path.exists():
        best = load_json(best_path)
        layer_index = int(best["layer_index"])
    layer_dir = memmap_dir / f"layer_{layer_index}"
    if not layer_dir.exists():
        raise FileNotFoundError(f"Run extraction first for layer {layer_index}: {layer_dir}")

    hidden_size = load_yaml(layer_dir / "meta.json")["hidden_size"]
    d_model = af_cfg.get("d_model", 256)
    n_layers = af_cfg.get("n_layers", 2)
    n_heads = af_cfg.get("n_heads", 4)
    ff_mult = af_cfg.get("ff_mult", 4)
    dropout = af_cfg.get("dropout", 0.1)
    loss_type = af_cfg.get("loss_type", "mse")
    causal = af_cfg.get("causal", False)
    min_sub = af_cfg.get("min_subseq_len", 2)
    max_sub = af_cfg.get("max_subseq_len", 256)
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-4))
    max_steps = train_cfg.get("max_steps")
    epochs = int(train_cfg.get("epochs", 3))

    train_ds = ActivationSequenceDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split="train",
        min_subseq_len=min_sub,
        max_subseq_len=max_sub,
        use_subsequence_sampling=True,
        seed=42,
    )
    val_ds = ActivationSequenceDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split="val",
        min_subseq_len=min_sub,
        max_subseq_len=max_sub,
        use_subsequence_sampling=False,
        seed=42,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_activation_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_activation_sequences,
        num_workers=0,
    )
    device = get_device()
    model = ActFormer(
        d_in=hidden_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_mult=ff_mult,
        dropout=dropout,
        loss_type=loss_type,
        causal=causal,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.get("wd", 0.01))
    best_val = float("inf")
    step = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb, mask in train_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            opt.zero_grad()
            mu, log_sigma, _ = model(xb, mask)
            loss = model.loss(mu, log_sigma, yb, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            opt.step()
            step += 1
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb, mask in val_loader:
                xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
                mu, log_sigma, _ = model(xb, mask)
                v = model.loss(mu, log_sigma, yb, mask)
                val_loss += v.item() * mask.sum().item()
                n_val += mask.sum().item()
        val_loss = val_loss / max(n_val, 1)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "step": step}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "step": step}, out_dir / "last.pt")
        print(f"Epoch {epoch} val_loss={val_loss:.4f} best={best_val:.4f}")
    print(f"[actformer] Saved to {out_dir}")


if __name__ == "__main__":
    main()
