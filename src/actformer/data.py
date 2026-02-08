"""
ActFormer dataset: load activation sequences from memmap; next-step targets; optional subsequence sampling.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import load_json
from src.utils.dtypes import resolve_numpy_dtype


class ActivationSequenceDataset(Dataset):
    """
    Returns (x, y) for next-step prediction: x = positions 0..T-1, y = positions 1..T.
    Optional subsequence sampling: random contiguous slice of length L (min_subseq_len <= L <= max_subseq_len).
    """

    def __init__(
        self,
        memmap_path: Path,
        index_path: Path,
        meta_path: Path,
        mean_path: Path | None,
        std_path: Path | None,
        split: str,
        min_subseq_len: int = 2,
        max_subseq_len: int = 256,
        use_subsequence_sampling: bool = True,
        objective: str = "next_token",
        seed: int | None = None,
    ) -> None:
        self.split = split
        self.min_subseq_len = min_subseq_len
        self.max_subseq_len = max_subseq_len
        self.use_subsequence_sampling = use_subsequence_sampling
        self.objective = objective
        self.rng = np.random.default_rng(seed)
        meta = load_json(meta_path)
        self.total_tokens = meta["total_tokens"]
        self.hidden_size = meta["hidden_size"]
        dtype = resolve_numpy_dtype(meta.get("dtype", "float32"))
        self.arr = np.memmap(
            str(memmap_path),
            dtype=dtype,
            mode="r",
            shape=(self.total_tokens, self.hidden_size),
        )
        mean = np.load(mean_path) if mean_path and Path(mean_path).exists() else np.zeros(self.hidden_size)
        std = np.load(std_path) if std_path and Path(std_path).exists() else np.ones(self.hidden_size)
        std = np.where(std < 1e-8, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        index = load_json(index_path)
        self.entries = [e for e in index if e["split"] == split]

    def __len__(self) -> int:
        return len(self.entries)

    def _get_sequence(self, entry: dict[str, Any]) -> np.ndarray:
        s, L = entry["start"], entry["length"]
        seq = np.array(self.arr[s : s + L], dtype=np.float32)
        seq = (seq - self.mean) / self.std
        return seq

    def __getitem__(self, i: int) -> tuple[torch.Tensor, ...]:
        entry = self.entries[i]
        seq = self._get_sequence(entry)
        T = seq.shape[0]
        if self.use_subsequence_sampling and T >= self.min_subseq_len:
            min_len = self.min_subseq_len if self.objective == "next_token" else max(1, self.min_subseq_len)
            L = min(
                self.rng.integers(min_len, min(T, self.max_subseq_len) + 1),
                T,
            )
            if self.objective == "next_token" and L < 2:
                L = 2
            max_start = T - L
            start = self.rng.integers(0, max_start + 1) if max_start >= 0 else 0
            seq = seq[start : start + L]
            T = seq.shape[0]
        if self.objective == "mlm":
            if T < 1:
                seq = np.concatenate([seq, seq], axis=0)
            return (torch.from_numpy(seq),)
        # next_token: shift by one
        if T < 2:
            seq = np.concatenate([seq, seq], axis=0)
            T = 2
        x = seq[:-1]
        y = seq[1:]
        return torch.from_numpy(x), torch.from_numpy(y)


def collate_activation_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad to max length in batch; return (x, y, mask).
    x, y: (B, T_max, d), mask: (B, T_max) True = valid.
    """
    xs, ys = zip(*batch)
    max_len = max(z.shape[0] for z in xs)
    d = xs[0].shape[1]
    B = len(xs)
    x_pad = torch.full((B, max_len, d), pad_value, dtype=xs[0].dtype)
    y_pad = torch.full((B, max_len, d), pad_value, dtype=ys[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        x_pad[i, :L] = x
        y_pad[i, :L] = y
        mask[i, :L] = True
    return x_pad, y_pad, mask


def make_mlm_collator(
    mask_ratio: float = 0.15,
    pad_value: float = 0.0,
):
    """
    Factory for MLM collator. Returns (x, padding_mask, mlm_mask).
    x: (B, T_max, d) original sequences (model replaces masked positions internally).
    padding_mask: (B, T_max) True = valid token.
    mlm_mask: (B, T_max) True = masked for prediction.
    """

    def collate(
        batch: list[tuple[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = [item[0] for item in batch]
        max_len = max(s.shape[0] for s in seqs)
        d = seqs[0].shape[1]
        B = len(seqs)
        x = torch.full((B, max_len, d), pad_value, dtype=seqs[0].dtype)
        padding_mask = torch.zeros(B, max_len, dtype=torch.bool)
        mlm_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, seq in enumerate(seqs):
            L = seq.shape[0]
            x[i, :L] = seq
            padding_mask[i, :L] = True
            n_mask = max(1, int(L * mask_ratio))
            indices = torch.randperm(L)[:n_mask]
            mlm_mask[i, indices] = True
        return x, padding_mask, mlm_mask

    return collate
