"""
Doc-level probe dataset: (sequence, label) per document from memmap + index.
For ActFormer fine-tuning: one sample per doc, sequence = concatenated chunks, normalized.
"""
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.io import load_json
from src.utils.dtypes import resolve_numpy_dtype


class DocLevelProbeDataset(Dataset):
    """
    One sample per document: (seq, label). Sequence = concatenation of all chunks for that doc,
    normalized with train mean/std, optionally truncated to max_len.
    """

    def __init__(
        self,
        memmap_path: Path,
        index_path: Path,
        meta_path: Path,
        mean_path: Path | None,
        std_path: Path | None,
        split: str,
        max_len: int | None = None,
    ) -> None:
        self.split = split
        self.max_len = max_len
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
        self.doc_ids = sorted(by_doc.keys())
        self.chunks = by_doc
        self.labels = labels

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        doc_id = self.doc_ids[i]
        chunks = self.chunks[doc_id]
        parts = [np.array(self.arr[s : s + L], dtype=np.float32) for s, L in chunks]
        seq = np.concatenate(parts, axis=0)
        seq = (seq - self.mean) / self.std
        if self.max_len is not None and seq.shape[0] > self.max_len:
            seq = seq[: self.max_len]
        if seq.shape[0] < 1:
            seq = np.zeros((1, self.hidden_size), dtype=np.float32)
        label = self.labels[doc_id]
        return torch.from_numpy(seq), label


def collate_doc_level_probe(
    batch: list[tuple[torch.Tensor, int]],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad sequences to batch max length. Return (x, mask, labels).
    x: (B, T_max, d), mask: (B, T_max) True = valid, labels: (B,).
    """
    seqs, labels = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    d = seqs[0].shape[1]
    B = len(seqs)
    x = torch.full((B, max_len, d), pad_value, dtype=seqs[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        L = seq.shape[0]
        x[i, :L] = seq
        mask[i, :L] = True
    return x, mask, torch.tensor(labels, dtype=torch.long)
