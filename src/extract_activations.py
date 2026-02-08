"""
Extract per-token activations from a frozen LM layer; save to memmap + index.
Usage: python -m src.extract_activations --config configs/default.yaml [--split train|val|test|all] [--layer_index L] [--for_pretrain]
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.utils.data_loading import load_and_split
from src.utils.device import get_device
from src.utils.io import ensure_dir, load_json, load_yaml, save_json
from src.utils.seed import set_seed


def get_chunk_ranges(
    seq_len: int,
    max_seq_len: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return list of (start, length) for chunks. No cross-doc mixing."""
    if seq_len <= max_seq_len:
        return [(0, seq_len)] if seq_len > 0 else []
    ranges = []
    start = 0
    while start < seq_len:
        length = min(max_seq_len, seq_len - start)
        ranges.append((start, length))
        if start + length >= seq_len:
            break
        start += stride
    return ranges


def _welford_update(
    n: int,
    mean: np.ndarray,
    M2: np.ndarray,
    chunk: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Update running mean and M2 (Welford) with a (L, D) chunk. Returns (n, mean, M2)."""
    for i in range(chunk.shape[0]):
        n += 1
        x = chunk[i].astype(np.float64)
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + delta * (x - mean)
    return n, mean, M2


def extract_activations(
    config: dict[str, Any],
    split: str,
    layer_index: int,
    for_pretrain: bool = False,
) -> None:
    """Run extraction for the given split and layer; write memmap + index."""
    set_seed(config.get("seed", 42))
    model_cfg = config.get("model", {})
    ext_cfg = config.get("extraction", {})
    top_data = config.get("data", {})

    if for_pretrain:
        pretrain_cfg = config.get("pretrain", {})
        pretrain_memmap = pretrain_cfg.get("memmap_dir")
        pretrain_data = pretrain_cfg.get("data")
        if pretrain_data is not None and pretrain_memmap:
            data_cfg = {**top_data, **pretrain_data}
            memmap_dir = Path(pretrain_memmap)
        else:
            data_cfg = top_data
            memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))
    else:
        data_cfg = top_data
        memmap_dir = Path(ext_cfg.get("memmap_dir", "outputs/activations"))

    data_dir = data_cfg.get("data_dir", "data/RS1")
    categories = data_cfg.get("categories", ["benign", "malicious"])
    label_map = data_cfg.get("label_map", {"benign": 0, "malicious": 1})
    split_by_file = data_cfg.get("split_by_file", False)
    train_ratio = data_cfg.get("train_ratio", 0.7)
    val_ratio = data_cfg.get("val_ratio", 0.15)
    test_ratio = data_cfg.get("test_ratio", 0.15)
    limit_docs = data_cfg.get("limit_docs") or ext_cfg.get("limit_docs")

    base_model_name = model_cfg.get("base_model_name", "distilgpt2")
    tap_point = model_cfg.get("tap_point", "resid_post")
    max_seq_len = ext_cfg.get("max_seq_len", 256)
    stride = ext_cfg.get("stride", 128)
    batch_size = ext_cfg.get("batch_size_extract", 8)
    dtype_str = ext_cfg.get("activation_dtype", "float32")
    chunk_docs = ext_cfg.get("chunk_docs") or 0
    chunk_docs = int(chunk_docs) if chunk_docs else 0

    dtype = getattr(np, dtype_str, np.float32)
    if dtype_str == "float16":
        dtype = np.float16
    elif dtype_str == "bfloat16":
        dtype = np.float32  # numpy doesn't have bfloat16; store as float32

    # Load data and split
    full_df, train_df, val_df, test_df = load_and_split(
        data_dir=data_dir,
        categories=categories,
        label_map=label_map,
        split_by_file=split_by_file,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=config.get("seed", 42),
        limit_docs=limit_docs,
    )
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    if split != "all":
        docs_df = split_dfs[split]
    else:
        docs_df = full_df

    if len(docs_df) == 0:
        print(f"[extract] No documents for split={split}. Exiting.")
        return

    # Output dir: memmap_dir / layer_{layer_index} (and split suffix if not all)
    out_dir = ensure_dir(memmap_dir / f"layer_{layer_index}")
    if split != "all":
        out_dir = ensure_dir(out_dir / split)

    cache_dir = model_cfg.get("cache_dir")
    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    device = get_device()
    device_map = "auto" if device.type == "cuda" else None

    if chunk_docs > 0:
        # ----- Two-pass chunked extraction -----
        print(f"[extract] Chunked extraction (chunk_docs={chunk_docs})")
        print(f"[extract] Loading tokenizer {base_model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, padding_side="left", **kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        hf_config = AutoConfig.from_pretrained(base_model_name, **kwargs)
        hidden_size = hf_config.hidden_size

        # Pass 1: tokenize only, build index and chunk boundaries
        print("[extract] Pass 1: tokenize and build index ...")
        index_entries: list[dict[str, Any]] = []
        chunk_boundaries: list[tuple[int, int]] = []
        global_start = 0
        n_docs = len(docs_df)
        for chunk_start in range(0, n_docs, chunk_docs):
            chunk_end = min(chunk_start + chunk_docs, n_docs)
            chunk_df = docs_df.iloc[chunk_start:chunk_end]
            chunk_n_entries_before = len(index_entries)
            for _, row in tqdm(
                chunk_df.iterrows(),
                total=len(chunk_df),
                desc="Pass1 tokenize",
                leave=False,
            ):
                enc = tokenizer(
                    row["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len * 4,
                )
                seq_len = enc["input_ids"].shape[1]
                ranges = get_chunk_ranges(seq_len, max_seq_len, stride)
                doc_id = int(row["doc_id"])
                label = int(row["label"])
                text = row["text"]
                doc_split = split
                if split == "all":
                    if doc_id in train_df["doc_id"].values:
                        doc_split = "train"
                    elif doc_id in val_df["doc_id"].values:
                        doc_split = "val"
                    else:
                        doc_split = "test"
                for seq_idx, (_, length) in enumerate(ranges):
                    index_entries.append({
                        "doc_id": doc_id,
                        "seq_id": seq_idx,
                        "start": int(global_start),
                        "length": int(length),
                        "label": label,
                        "split": doc_split,
                        "text_ref": text[:100] + "..." if len(text) > 100 else text,
                    })
                    global_start += length
            chunk_boundaries.append((chunk_n_entries_before, len(index_entries)))
        total_tokens = global_start
        print(f"[extract] Total tokens: {total_tokens}, hidden_size: {hidden_size}")

        memmap_path = out_dir / "activations.dat"
        activations = np.memmap(
            str(memmap_path),
            dtype=dtype,
            mode="w+",
            shape=(total_tokens, hidden_size),
        )

        # Pass 2: load model, fill memmap, Welford for train mean/std
        print("[extract] Pass 2: extract activations ...")
        print(f"[extract] Loading {base_model_name} ...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=device_map,
            output_hidden_states=True,
            **kwargs,
        ).eval()
        if device_map is None:
            model = model.to(device)
        else:
            device = next(model.parameters()).device

        n_train = 0
        welford_mean = np.zeros(hidden_size, dtype=np.float64)
        welford_M2 = np.zeros(hidden_size, dtype=np.float64)

        for chunk_idx in range(len(chunk_boundaries)):
            start_idx, end_idx = chunk_boundaries[chunk_idx]
            write_off = index_entries[start_idx]["start"]
            chunk_start_docs = chunk_idx * chunk_docs
            chunk_end_docs = min(chunk_start_docs + chunk_docs, n_docs)
            chunk_df = docs_df.iloc[chunk_start_docs:chunk_end_docs]
            entry_idx = start_idx
            for _, row in tqdm(
                chunk_df.iterrows(),
                total=len(chunk_df),
                desc=f"Pass2 chunk {chunk_idx + 1}/{len(chunk_boundaries)}",
                leave=False,
            ):
                enc = tokenizer(
                    row["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len * 4,
                )
                seq_len = enc["input_ids"].shape[1]
                ranges = get_chunk_ranges(seq_len, max_seq_len, stride)
                input_ids_full = enc["input_ids"].to(device)
                for (chunk_start_off, chunk_len) in ranges:
                    input_ids = input_ids_full[:, chunk_start_off : chunk_start_off + chunk_len]
                    if input_ids.shape[1] < chunk_len:
                        pad = torch.full(
                            (1, chunk_len - input_ids.shape[1]),
                            tokenizer.pad_token_id or 0,
                            device=device,
                            dtype=input_ids.dtype,
                        )
                        input_ids = torch.cat([input_ids, pad], dim=1)
                    attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()
                    with torch.no_grad():
                        out = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                        )
                    arr = out.hidden_states[layer_index][0].cpu().float().numpy()
                    length = arr.shape[0]
                    activations[write_off : write_off + length] = arr
                    entry = index_entries[entry_idx]
                    if entry["split"] == "train":
                        n_train, welford_mean, welford_M2 = _welford_update(
                            n_train, welford_mean, welford_M2, arr
                        )
                    entry_idx += 1
                    write_off += length

        activations.flush()
        del activations

        save_json(index_entries, out_dir / "index.json")
        save_json(
            {"total_tokens": total_tokens, "hidden_size": hidden_size, "dtype": dtype_str},
            out_dir / "meta.json",
        )
        if n_train > 0:
            std = np.sqrt(welford_M2 / n_train)
            std = np.where(std < 1e-8, 1.0, std)
            np.save(out_dir / "train_mean.npy", welford_mean.astype(np.float64))
            np.save(out_dir / "train_std.npy", std.astype(np.float64))
            print(f"[extract] Saved train mean/std shape {welford_mean.shape} (Welford, n_train={n_train})")
        print(f"[extract] Saved {len(index_entries)} sequences to {out_dir}")
    else:
        # ----- Single-pass (original) -----
        print(f"[extract] Loading {base_model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, padding_side="left", **kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=device_map,
            output_hidden_states=True,
            **kwargs,
        ).eval()
        if device_map is None:
            model = model.to(device)
        else:
            device = next(model.parameters()).device
        hidden_size = model.config.hidden_size

        # First pass: tokenize each doc and get chunk ranges
        print("[extract] Tokenizing and computing chunk ranges ...")
        doc_encodings: list[dict] = []
        doc_chunk_ranges: list[list[tuple[int, int]]] = []
        for _, row in tqdm(docs_df.iterrows(), total=len(docs_df), desc="Tokenize"):
            enc = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len * 4,
            )
            seq_len = enc["input_ids"].shape[1]
            ranges = get_chunk_ranges(seq_len, max_seq_len, stride)
            doc_encodings.append(enc)
            doc_chunk_ranges.append(ranges)
        total_tokens = sum(
            length for ranges in doc_chunk_ranges for _, length in ranges
        )
        print(f"[extract] Total tokens: {total_tokens}, hidden_size: {hidden_size}")

        memmap_path = out_dir / "activations.dat"
        activations = np.memmap(
            str(memmap_path),
            dtype=dtype,
            mode="w+",
            shape=(total_tokens, hidden_size),
        )

        index_entries = []
        global_start = 0
        train_states: list[np.ndarray] = []

        for doc_idx, (_, row) in enumerate(tqdm(docs_df.iterrows(), total=len(docs_df), desc="Extract")):
            doc_id = int(row["doc_id"])
            label = int(row["label"])
            text = row["text"]
            doc_split = split
            if split == "all":
                if doc_id in train_df["doc_id"].values:
                    doc_split = "train"
                elif doc_id in val_df["doc_id"].values:
                    doc_split = "val"
                else:
                    doc_split = "test"
            enc = doc_encodings[doc_idx]
            input_ids_full = enc["input_ids"].to(device)
            ranges = doc_chunk_ranges[doc_idx]
            for seq_idx, (chunk_start, chunk_len) in enumerate(ranges):
                input_ids = input_ids_full[:, chunk_start : chunk_start + chunk_len]
                if input_ids.shape[1] < chunk_len:
                    pad = torch.full(
                        (1, chunk_len - input_ids.shape[1]),
                        tokenizer.pad_token_id or 0,
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    input_ids = torch.cat([input_ids, pad], dim=1)
                attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()

                with torch.no_grad():
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                layer_states = out.hidden_states[layer_index]
                arr = layer_states[0].cpu().float().numpy()
                length = arr.shape[0]
                activations[global_start : global_start + length] = arr
                if doc_split == "train":
                    train_states.append(arr.copy())
                index_entries.append({
                    "doc_id": doc_id,
                    "seq_id": seq_idx,
                    "start": int(global_start),
                    "length": int(length),
                    "label": label,
                    "split": doc_split,
                    "text_ref": text[:100] + "..." if len(text) > 100 else text,
                })
                global_start += length

        activations.flush()
        del activations

        save_json(index_entries, out_dir / "index.json")
        save_json(
            {"total_tokens": total_tokens, "hidden_size": hidden_size, "dtype": dtype_str},
            out_dir / "meta.json",
        )
        if train_states:
            train_arr = np.concatenate(train_states, axis=0)
            mean = np.mean(train_arr, axis=0).astype(np.float64)
            std = np.std(train_arr, axis=0).astype(np.float64)
            std[std < 1e-8] = 1.0
            np.save(out_dir / "train_mean.npy", mean)
            np.save(out_dir / "train_std.npy", std)
            print(f"[extract] Saved train mean/std shape {mean.shape}")

        print(f"[extract] Saved {len(index_entries)} sequences to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activations to memmap")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--layer_index", type=int, default=None, help="Override config layer_index")
    parser.add_argument(
        "--for_pretrain",
        action="store_true",
        help="Use pretrain.data and pretrain.memmap_dir; requires both to be set in config",
    )
    parser.add_argument(
        "--chunk_docs",
        type=int,
        default=None,
        help="Override extraction.chunk_docs for chunked extraction (documents per chunk)",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.chunk_docs is not None:
        config.setdefault("extraction", {})["chunk_docs"] = args.chunk_docs
    if args.layer_index is not None:
        config.setdefault("model", {})["layer_index"] = args.layer_index
        layer_index = args.layer_index
    else:
        layer_index = config.get("model", {}).get("layer_index", 0)
        if args.for_pretrain:
            best_path = Path(
                config.get("layer_search", {}).get("layer_search_output", "outputs/best_layer.json")
            )
            if best_path.exists():
                best = load_json(best_path)
                layer_index = int(best["layer_index"])
    extract_activations(config, args.split, layer_index, for_pretrain=args.for_pretrain)


if __name__ == "__main__":
    main()
