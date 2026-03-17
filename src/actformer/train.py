"""
ActFormer pretraining: next-step activation prediction.
Usage: python -m src.actformer.train --config configs/default.yaml [--run_extraction]
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.actformer.data import ActivationSequenceDataset, collate_activation_sequences, make_mlm_collator
from src.actformer.model import ActFormer
from src.run_dir import ensure_unique_output_dir
from src.utils.device import get_device
from src.utils.io import load_json, load_yaml
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb, load_dotenv_for_wandb

REQUIRED_PRETRAIN_FILES = ["activations.dat", "index.json", "meta.json", "train_mean.npy", "train_std.npy"]


def _check_pretrain_extraction(
    layer_dir: Path,
    layer_index: int,
    config_path: str,
    use_pretrain_memmap: bool,
) -> None:
    """Raise FileNotFoundError if layer_dir or required files missing; message suggests extraction command."""
    missing = [f for f in REQUIRED_PRETRAIN_FILES if not (layer_dir / f).exists()]
    if not layer_dir.exists() or missing:
        if use_pretrain_memmap:
            data_source = "pretrain dataset"
            cmd = f"python -m src.extract_activations --config {config_path} --for_pretrain --layer_index {layer_index} --split all"
        else:
            data_source = "activations"
            cmd = f"python -m src.extract_activations --config {config_path} --layer_index {layer_index} --split all"
        msg = (
            f"Extraction for the {data_source} at best layer {layer_index} is required.\n"
            f"Expected path: {layer_dir}\n"
        )
        if missing:
            msg += f"Missing files: {missing}\n"
        msg += f"Run: {cmd}"
        raise FileNotFoundError(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--run_extraction",
        action="store_true",
        help="If pretrain layer dir is missing, run extraction (for_pretrain when pretrain.memmap_dir is set) then train",
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_yaml(config_path)
    set_seed(config.get("seed", 42))
    ext_cfg = config.get("extraction", {})
    pretrain_cfg = config.get("pretrain", {})
    model_cfg = config.get("model", {})
    af_cfg = config.get("actformer", {})
    train_cfg = config.get("pretrain", {})
    use_pretrain_memmap = bool(pretrain_cfg.get("memmap_dir"))
    if use_pretrain_memmap:
        memmap_dir = Path(pretrain_cfg["memmap_dir"])
    else:
        memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))
    out_dir = Path(train_cfg.get("output_dir", "outputs/actformer"))
    out_dir = ensure_unique_output_dir(out_dir, config=config)
    layer_index = model_cfg.get("layer_index", 0)
    best_path = Path(config.get("layer_search", {}).get("layer_search_output", "outputs/best_layer.json"))
    if best_path.exists():
        best = load_json(best_path)
        layer_index = int(best["layer_index"])
    layer_dir = memmap_dir / f"layer_{layer_index}"

    missing_files = [f for f in REQUIRED_PRETRAIN_FILES if not (layer_dir / f).exists()]
    if not layer_dir.exists() or missing_files:
        if args.run_extraction:
            from src.extract_activations import extract_activations
            print(f"[actformer] Running extraction for {'pretrain' if use_pretrain_memmap else 'activations'} at layer {layer_index} ...")
            extract_activations(config, "all", layer_index, for_pretrain=use_pretrain_memmap)
            missing_after = [f for f in REQUIRED_PRETRAIN_FILES if not (layer_dir / f).exists()]
            if not layer_dir.exists() or missing_after:
                _check_pretrain_extraction(layer_dir, layer_index, config_path, use_pretrain_memmap)
        else:
            _check_pretrain_extraction(layer_dir, layer_index, config_path, use_pretrain_memmap)

    hidden_size = load_yaml(layer_dir / "meta.json")["hidden_size"]
    d_model = af_cfg.get("d_model", 256)
    n_layers = af_cfg.get("n_layers", 2)
    n_heads = af_cfg.get("n_heads", 4)
    ff_mult = af_cfg.get("ff_mult", 4)
    dropout = af_cfg.get("dropout", 0.1)
    loss_type = af_cfg.get("loss_type", "mse")
    causal = af_cfg.get("causal", True)
    objective = af_cfg.get("objective", "next_token")
    mlm_mask_ratio = float(af_cfg.get("mlm_mask_ratio", 0.15))
    min_sub = af_cfg.get("min_subseq_len", 2)
    max_sub = af_cfg.get("max_subseq_len", 256)
    batch_size = int(train_cfg.get("batch_size", 32))
    lr = float(train_cfg.get("lr", 1e-4))
    max_steps = train_cfg.get("max_steps")
    epochs = int(train_cfg.get("epochs", 3))

    load_dotenv_for_wandb(config)
    wandb_cfg = config.get("wandb", {})
    init_wandb(
        project=wandb_cfg.get("pretrain_project", "actformer-pretrain"),
        name=wandb_cfg.get("name"),
        config={
            "layer_index": layer_index,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "objective": objective,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "causal": causal,
        },
        entity=wandb_cfg.get("entity"),
    )

    print(f"[actformer] objective={objective}, causal={causal}")
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
        objective=objective,
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
        objective=objective,
        seed=42,
    )
    if objective == "mlm":
        train_collate = make_mlm_collator(mask_ratio=mlm_mask_ratio)
        val_collate = make_mlm_collator(mask_ratio=mlm_mask_ratio)
    else:
        train_collate = collate_activation_sequences
        val_collate = collate_activation_sequences
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate,
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
        train_loss_sum = 0.0
        train_n = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=True):
            if objective == "mlm":
                xb, padding_mask, mlm_mask = batch
                xb, padding_mask, mlm_mask = xb.to(device), padding_mask.to(device), mlm_mask.to(device)
                opt.zero_grad()
                mu, log_sigma, _ = model(xb, padding_mask, mlm_mask=mlm_mask)
                loss_mask = mlm_mask & padding_mask
                loss = model.loss(mu, log_sigma, xb, loss_mask)
                train_n += loss_mask.sum().item()
            else:
                xb, yb, mask = batch
                xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
                opt.zero_grad()
                mu, log_sigma, _ = model(xb, mask)
                loss = model.loss(mu, log_sigma, yb, mask)
                train_n += mask.sum().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            opt.step()
            train_loss_sum += loss.item() * (loss_mask.sum().item() if objective == "mlm" else mask.sum().item())
            step += 1
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"train_loss_batch": loss.item()}, step=step)
            except Exception:
                pass
            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break
        mean_train_loss = train_loss_sum / max(train_n, 1)
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                if objective == "mlm":
                    xb, padding_mask, mlm_mask = batch
                    xb, padding_mask, mlm_mask = xb.to(device), padding_mask.to(device), mlm_mask.to(device)
                    mu, log_sigma, _ = model(xb, padding_mask, mlm_mask=mlm_mask)
                    loss_mask = mlm_mask & padding_mask
                    v = model.loss(mu, log_sigma, xb, loss_mask)
                    val_loss += v.item() * loss_mask.sum().item()
                    n_val += loss_mask.sum().item()
                else:
                    xb, yb, mask = batch
                    xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
                    mu, log_sigma, _ = model(xb, mask)
                    v = model.loss(mu, log_sigma, yb, mask)
                    val_loss += v.item() * mask.sum().item()
                    n_val += mask.sum().item()
        val_loss = val_loss / max(n_val, 1)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"train_loss": mean_train_loss, "val_loss": val_loss}, step=step)
        except Exception:
            pass
        actformer_config = {
            "d_in": hidden_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ff_mult": ff_mult,
            "loss_type": loss_type,
            "causal": causal,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "step": step, "actformer_config": actformer_config},
                out_dir / "best.pt",
            )
        torch.save(
            {"model": model.state_dict(), "step": step, "actformer_config": actformer_config},
            out_dir / "last.pt",
        )
        print(f"Epoch {epoch} val_loss={val_loss:.4f} best={best_val:.4f}")
    print(f"[actformer] Saved to {out_dir}")


if __name__ == "__main__":
    main()
