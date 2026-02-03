"""
Probe models: linear, MLP, and transformer-from-scratch classifiers.
The transformer probe is built from scratch (no pretrained weights).
"""
from typing import Literal

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Single linear layer: (B, d) -> (B, n_classes)."""

    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) or (B, T, d) -> if T, mean-pool over T
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.linear(x)


class MLPProbe(nn.Module):
    """Two-layer MLP: (B, d) -> (B, n_classes)."""

    def __init__(self, d_in: int, n_classes: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


class TransformerProbe(nn.Module):
    """
    Small transformer from scratch: (B, T, d_in) -> (B, n_classes).
    Uses learned positional embeddings, then N encoder layers, then pool + linear.
    """

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 512,
        pool: Literal["mean", "last"] = "mean",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, d_in)
        mask: (B, T) bool, True = valid token (attend). If None, all positions valid.
        """
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        if mask is not None:
            # TransformerEncoder expects key_padding_mask: (B, T) True = ignore
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.drop(x)
        if self.pool == "mean":
            if key_padding_mask is not None:
                x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                x = x.sum(dim=1) / lengths
            else:
                x = x.mean(dim=1)
        else:
            # last valid position per batch
            if key_padding_mask is not None:
                last_idx = mask.sum(dim=1) - 1
                x = x[torch.arange(B, device=x.device), last_idx]
            else:
                x = x[:, -1]
        return self.head(x)


def build_probe(
    probe_type: Literal["linear", "mlp", "transformer"],
    d_in: int,
    n_classes: int,
    **kwargs,
) -> nn.Module:
    """Build probe by type. kwargs passed to MLP/Transformer constructors."""
    if probe_type == "linear":
        return LinearProbe(d_in, n_classes)
    if probe_type == "mlp":
        return MLPProbe(d_in, n_classes, **kwargs)
    if probe_type == "transformer":
        return TransformerProbe(d_in, n_classes, **kwargs)
    raise ValueError(f"Unknown probe_type: {probe_type}")
