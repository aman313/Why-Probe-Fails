"""
ActFormer: small causal transformer that predicts next-step activation from previous activations.
Output head: MSE or Gaussian NLL. Exposes last_hidden_state for probing.
"""
import math
from typing import Literal

import torch
import torch.nn as nn


class ActFormer(nn.Module):
    """
    Input: (B, T, d). Project to d_model, causal transformer, output head to d (mu and optionally log_sigma).
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 2048,
        loss_type: Literal["mse", "gaussian_nll"] = "mse",
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.loss_type = loss_type
        self.causal = causal
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(d_in) * 0.02)
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
        self.head_mu = nn.Linear(d_model, d_in)
        if loss_type == "gaussian_nll":
            self.head_log_sigma = nn.Linear(d_model, d_in)
        else:
            self.head_log_sigma = None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        mlm_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        x: (B, T, d_in). mask: (B, T) True = valid. mlm_mask: (B, T) True = masked for MLM.
        Returns: mu (B, T, d_in), log_sigma (B, T, d_in) or None, last_hidden_state (B, T, d_model).
        """
        B, T, _ = x.shape
        if mlm_mask is not None:
            x = x.clone()
            x[mlm_mask] = self.mask_token
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        attn_mask = None
        if self.causal:
            # True = mask out (disallow attention). Upper-triangular mask blocks future positions.
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        if mask is not None:
            key_padding_mask = ~mask
        else:
            key_padding_mask = None
        hidden = self.transformer(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        last_hidden_state = self.drop(hidden)
        mu = self.head_mu(last_hidden_state)
        if self.head_log_sigma is not None:
            log_sigma = self.head_log_sigma(last_hidden_state)
            log_sigma = log_sigma.clamp(-3.0, 3.0)
            return mu, log_sigma, last_hidden_state
        return mu, None, last_hidden_state

    def loss(
        self,
        mu: torch.Tensor,
        log_sigma: torch.Tensor | None,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token loss; mean over valid positions."""
        if log_sigma is not None:
            sigma = torch.exp(log_sigma)
            nll = 0.5 * (((y - mu) / sigma) ** 2 + 2 * log_sigma + math.log(2 * math.pi))
        else:
            nll = (y - mu) ** 2
        nll = nll.mean(dim=-1)
        nll = nll.masked_fill(~mask, 0.0)
        return nll.sum() / mask.sum().clamp(min=1)


class ActFormerWithHead(nn.Module):
    """
    Pretrained ActFormer + pool + classification head for probe fine-tuning.
    Forward: (B, T, d_in) -> logits (B, n_classes). No mlm_mask; use for classification only.
    """

    def __init__(
        self,
        actformer: ActFormer,
        n_classes: int,
        pool: Literal["mean", "last"] = "mean",
        freeze_body: bool = False,
    ) -> None:
        super().__init__()
        self.actformer = actformer
        self.pool = pool
        self.head = nn.Linear(actformer.d_model, n_classes)
        if freeze_body:
            for p in self.actformer.parameters():
                p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, d_in). mask: (B, T) True = valid. No mlm_mask (classification only).
        Returns: logits (B, n_classes).
        """
        _, _, last_hidden = self.actformer(x, mask=mask, mlm_mask=None)
        B, T, D = last_hidden.shape
        if self.pool == "mean":
            if mask is not None:
                last_hidden = last_hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = last_hidden.sum(dim=1) / lengths
            else:
                pooled = last_hidden.mean(dim=1)
        else:
            if mask is not None:
                last_idx = mask.sum(dim=1) - 1
                pooled = last_hidden[torch.arange(B, device=last_hidden.device), last_idx]
            else:
                pooled = last_hidden[:, -1]
        return self.head(pooled)


def load_actformer_with_head(
    checkpoint_path: str | None,
    d_in: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_classes: int,
    ff_mult: int = 4,
    dropout: float = 0.1,
    loss_type: Literal["mse", "gaussian_nll"] = "mse",
    causal: bool = True,
    pool: Literal["mean", "last"] = "mean",
    freeze_body: bool = False,
) -> ActFormerWithHead:
    """Build ActFormerWithHead; load pretrained ActFormer weights from checkpoint if path given."""
    actformer = ActFormer(
        d_in=d_in,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_mult=ff_mult,
        dropout=dropout,
        loss_type=loss_type,
        causal=causal,
    )
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        actformer.load_state_dict(state, strict=True)
    return ActFormerWithHead(actformer, n_classes=n_classes, pool=pool, freeze_body=freeze_body)
