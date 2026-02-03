"""
ActFormer evaluation: MSE, cosine similarity, NLL on test split.
"""
from pathlib import Path

import torch

from src.actformer.data import ActivationSequenceDataset, collate_activation_sequences
from src.actformer.model import ActFormer
from src.utils.io import load_json, load_yaml


def evaluate(
    config: dict,
    checkpoint_path: Path,
) -> dict[str, float]:
    """Load model and compute test metrics. Returns dict with mse, cosine_sim, nll (if gaussian)."""
    ext_cfg = config.get("extraction", {})
    model_cfg = config.get("model", {})
    af_cfg = config.get("actformer", {})
    memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))
    layer_index = model_cfg.get("layer_index", 0)
    layer_dir = memmap_dir / f"layer_{layer_index}"
    meta = load_json(layer_dir / "meta.json")
    hidden_size = meta["hidden_size"]
    model = ActFormer(
        d_in=hidden_size,
        d_model=af_cfg.get("d_model", 256),
        n_layers=af_cfg.get("n_layers", 2),
        n_heads=af_cfg.get("n_heads", 4),
        ff_mult=af_cfg.get("ff_mult", 4),
        dropout=0.0,
        loss_type=af_cfg.get("loss_type", "mse"),
        causal=af_cfg.get("causal", False),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    test_ds = ActivationSequenceDataset(
        layer_dir / "activations.dat",
        layer_dir / "index.json",
        layer_dir / "meta.json",
        layer_dir / "train_mean.npy",
        layer_dir / "train_std.npy",
        split="test",
        use_subsequence_sampling=False,
    )
    if len(test_ds) == 0:
        return {"mse": float("nan"), "cosine_sim": float("nan")}
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, collate_fn=collate_activation_sequences
    )
    mse_sum = 0.0
    cos_sum = 0.0
    nll_sum = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb, mask in loader:
            mu, log_sigma, _ = model(xb, mask)
            diff = (yb - mu).masked_fill(~mask.unsqueeze(-1), 0)
            mse_sum += (diff ** 2).sum().item()
            mu_flat = mu.masked_fill(~mask.unsqueeze(-1), 0)
            y_flat = yb.masked_fill(~mask.unsqueeze(-1), 0)
            cos = torch.nn.functional.cosine_similarity(
                mu_flat.view(-1, mu.shape[-1]),
                y_flat.view(-1, yb.shape[-1]),
                dim=1,
            ).sum()
            cos_sum += cos.item()
            n += mask.sum().item()
            if log_sigma is not None:
                nll_sum += model.loss(mu, log_sigma, yb, mask).item() * mask.sum().item()
    mse = mse_sum / max(n, 1)
    cosine_sim = cos_sum / max(n, 1)
    out = {"mse": mse, "cosine_sim": cosine_sim}
    if af_cfg.get("loss_type") == "gaussian_nll":
        out["nll"] = nll_sum / max(n, 1)
    return out
