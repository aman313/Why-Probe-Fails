"""
Extract per-token activations from a frozen LM layer; save to memmap + index.
Uses incremental input (streaming docs), incremental write (flush per doc), and resume via progress.json.
Usage: python -m src.extract_activations --config configs/default.yaml [--split train|val|test|all] [--layer_index L] [--for_pretrain] [--verbose]
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.utils.data_loading import (
    build_doc_id_to_split,
    iter_docs,
    load_docs_metadata,
    split_docs,
)
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


def _entry_idx_at_write_offset(index_entries: list[dict[str, Any]], write_offset: int) -> int:
    """Return the number of entries fully written (entry.start + entry.length <= write_offset)."""
    idx = 0
    for e in index_entries:
        if e["start"] + e["length"] <= write_offset:
            idx += 1
        else:
            break
    return idx


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


def _load_progress(out_dir: Path) -> dict[str, Any] | None:
    """Load progress.json if it exists; else None."""
    p = out_dir / "progress.json"
    if not p.exists():
        return None
    return load_json(p)


def _save_progress(out_dir: Path, data: dict[str, Any]) -> None:
    save_json(data, out_dir / "progress.json")


def _rebuild_welford_from_memmap(
    activations: np.memmap,
    index_entries: list[dict[str, Any]],
    write_offset: int,
    hidden_size: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Recompute Welford state for train entries with start+length <= write_offset."""
    n_train = 0
    mean = np.zeros(hidden_size, dtype=np.float64)
    M2 = np.zeros(hidden_size, dtype=np.float64)
    for e in index_entries:
        if e["split"] != "train":
            continue
        start, length = e["start"], e["length"]
        if start + length > write_offset:
            continue
        chunk = np.asarray(activations[start : start + length], dtype=np.float64)
        n_train, mean, M2 = _welford_update(n_train, mean, M2, chunk)
    return n_train, mean, M2


def _process_buffer(
    buffer: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]],
    batch_size: int,
    model: torch.nn.Module,
    layer_index: int,
    activations: np.memmap,
    write_offset: int,
    n_train: int,
    welford_mean: np.ndarray,
    welford_M2: np.ndarray,
    device: torch.device,
    dtype: np.dtype,
    tokenizer: Any,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Process buffer in batches: pad, forward, write to memmap, update Welford. Returns (write_offset, n_train, welford_mean, welford_M2)."""
    pad_token_id = tokenizer.pad_token_id or getattr(tokenizer, "eos_token_id", 0) or 0
    for i in range(0, len(buffer), batch_size):
        batch_items = buffer[i : i + batch_size]
        max_len = max(item[0].shape[1] for item in batch_items)
        input_ids_list = []
        attention_mask_list = []
        lengths = []
        for input_ids, attention_mask, entry in batch_items:
            L = input_ids.shape[1]
            lengths.append(L)
            if L < max_len:
                pad = torch.full(
                    (1, max_len - L), pad_token_id, device=device, dtype=input_ids.dtype
                )
                input_ids = torch.cat([input_ids, pad], dim=1)
                mask_pad = torch.zeros(1, max_len - L, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        input_ids_b = torch.cat(input_ids_list, dim=0)
        attention_mask_b = torch.cat(attention_mask_list, dim=0)
        with torch.no_grad():
            out = model(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                output_hidden_states=True,
            )
        hidden = out.hidden_states[layer_index]
        for b, (_, _, entry) in enumerate(batch_items):
            L = lengths[b]
            arr = hidden[b, :L].cpu().float().numpy().astype(dtype)
            start = entry["start"]
            activations[start : start + L] = arr
            if entry["split"] == "train":
                n_train, welford_mean, welford_M2 = _welford_update(
                    n_train, welford_mean, welford_M2, arr
                )
            write_offset = start + L
    return write_offset, n_train, welford_mean, welford_M2


def extract_activations(
    config: dict[str, Any],
    split: str,
    layer_index: int,
    for_pretrain: bool = False,
    verbose: bool = False,
) -> None:
    """Run two-pass extraction with streaming input, incremental write, and resume."""
    start_time = time.perf_counter()
    set_seed(config.get("seed", 42))
    model_cfg = config.get("model", {})
    ext_cfg = config.get("extraction", {})
    verbose = verbose or ext_cfg.get("verbose", False)
    top_data = config.get("data", {})
    log_prefix = f"[extract] [layer_{layer_index}] [{split}]"

    def vlog(msg: str) -> None:
        if verbose:
            print(f"{log_prefix} {msg}")

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
    max_seq_len = ext_cfg.get("max_seq_len", 256)
    stride = ext_cfg.get("stride", 128)
    chunk_size = int(ext_cfg.get("chunk_size", 100_000))
    batch_size = int(ext_cfg.get("batch_size") or ext_cfg.get("batch_size_extract", 5000))
    dtype_str = ext_cfg.get("activation_dtype", "float32")

    dtype = getattr(np, dtype_str, np.float32)
    if dtype_str == "float16":
        dtype = np.float16
    elif dtype_str == "bfloat16":
        dtype = np.float32  # numpy doesn't have bfloat16; store as float32

    # Metadata only (small in memory)
    meta_df = load_docs_metadata(
        data_dir=data_dir,
        categories=categories,
        label_map=label_map,
        limit_docs=limit_docs,
    )
    vlog(f"Data loading: loaded metadata from {data_dir}, total docs={len(meta_df)}, limit_docs={limit_docs}")
    train_df, val_df, test_df = split_docs(
        meta_df, split_by_file, train_ratio, val_ratio, test_ratio, seed=config.get("seed", 42)
    )
    vlog(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}
    doc_id_to_split = build_doc_id_to_split(train_df, val_df, test_df)
    vlog(f"doc_id_to_split: {len(doc_id_to_split)} entries")

    if split == "all":
        docs_meta = meta_df
    else:
        docs_meta = split_dfs[split]
    n_docs = len(docs_meta)
    if n_docs == 0:
        print(f"[extract] No documents for split={split}. Exiting.")
        return

    out_dir = ensure_dir(memmap_dir / f"layer_{layer_index}")
    if split != "all":
        out_dir = ensure_dir(out_dir / split)

    print(f"[extract] Data source: {Path(data_dir).resolve()} (for_pretrain={for_pretrain})")
    print(f"[extract] Output dir: {out_dir.resolve()}")
    print(f"[extract] Docs in split: {n_docs}")

    cache_dir = model_cfg.get("cache_dir")
    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    device = get_device()
    device_map = "auto" if device.type == "cuda" else None

    progress = _load_progress(out_dir)
    index_path = out_dir / "index.json"
    index_jsonl_path = out_dir / "index.jsonl"
    memmap_path = out_dir / "activations.dat"

    # ----- Pass 1: tokenize stream, build index (index.jsonl), total_tokens -----
    pass1_start_doc = 0
    index_entries: list[dict[str, Any]] = []
    total_tokens = 0

    if progress is not None and progress.get("pass") == 1:
        pass1_start_doc = int(progress["last_doc_index"]) + 1
        total_tokens = int(progress.get("total_tokens", 0))
        if index_jsonl_path.exists():
            with open(index_jsonl_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries = json.loads(line)
                    index_entries.extend(entries)
        print(f"[extract] Resuming Pass 1 from doc_index {pass1_start_doc}")
        vlog(f"Resuming Pass 1: index_entries loaded={len(index_entries)}, total_tokens so far={total_tokens}")

    pass1_doc_step = max(1, (n_docs - pass1_start_doc) // 10) if (n_docs - pass1_start_doc) > 0 else 1
    print("[extract] Pass 1: tokenize and build index ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, padding_side="left", **kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_config = AutoConfig.from_pretrained(base_model_name, **kwargs)
    hidden_size = hf_config.hidden_size

    jsonl_mode = "a" if pass1_start_doc > 0 else "w"
    with open(index_jsonl_path, jsonl_mode) as jsonl_file:
        for doc_index, doc_id, text, label, doc_split in tqdm(
            iter_docs(
                data_dir,
                categories,
                label_map,
                doc_id_to_split,
                limit_docs=limit_docs,
                start_doc_index=pass1_start_doc,
                split=split,
            ),
            total=n_docs - pass1_start_doc if pass1_start_doc < n_docs else 0,
            desc="Pass 1 tokenize",
        ):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len * 4,
            )
            seq_len = enc["input_ids"].shape[1]
            ranges = get_chunk_ranges(seq_len, max_seq_len, stride)
            doc_entries = []
            for seq_idx, (_, length) in enumerate(ranges):
                doc_entries.append({
                    "doc_id": doc_id,
                    "seq_id": seq_idx,
                    "start": int(total_tokens),
                    "length": int(length),
                    "label": label,
                    "split": doc_split,
                    "text_ref": text[:100] + "..." if len(text) > 100 else text,
                })
                total_tokens += length
            index_entries.extend(doc_entries)
            jsonl_file.write(json.dumps(doc_entries) + "\n")
            jsonl_file.flush()
            _save_progress(out_dir, {"pass": 1, "last_doc_index": doc_index, "total_tokens": total_tokens})
            if verbose and (doc_index - pass1_start_doc) % pass1_doc_step == 0:
                vlog(f"Pass 1 progress: doc_index={doc_index}, total_tokens={total_tokens}, index_entries={len(index_entries)}")

    print(f"[extract] Total tokens: {total_tokens}, hidden_size: {hidden_size}")
    vlog(f"Pass 1 complete. Index entries: {len(index_entries)}, total_tokens: {total_tokens}")
    save_json(index_entries, index_path)
    if index_jsonl_path.exists():
        index_jsonl_path.unlink()

    # ----- Pass 2: load model, fill memmap in chunk_size buffers, batch_size forwards, Welford -----
    pass2_start_doc = 0
    write_offset = 0
    skip_first_n_chunks = 0
    n_train = 0
    welford_mean = np.zeros(hidden_size, dtype=np.float64)
    welford_M2 = np.zeros(hidden_size, dtype=np.float64)

    if progress is not None and progress.get("pass") == 2:
        write_offset = int(progress["write_offset"])
        if write_offset >= total_tokens:
            pass2_start_doc = n_docs
            activations = np.memmap(str(memmap_path), dtype=dtype, mode="r", shape=(total_tokens, hidden_size))
            n_train, welford_mean, welford_M2 = _rebuild_welford_from_memmap(
                activations, index_entries, total_tokens, hidden_size
            )
            del activations
            print("[extract] Pass 2 already complete (write_offset==total_tokens), finalizing ...")
            vlog("Skipping Pass 2 loop, going to finalize.")
        else:
            pass2_start_doc = int(progress["last_doc_index"])
            skip_first_n_chunks = int(progress.get("last_chunk_in_doc", -1)) + 1
            activations = np.memmap(str(memmap_path), dtype=dtype, mode="r+", shape=(total_tokens, hidden_size))
            n_train, welford_mean, welford_M2 = _rebuild_welford_from_memmap(
                activations, index_entries, write_offset, hidden_size
            )
            del activations
            print(f"[extract] Resuming Pass 2 from doc_index {pass2_start_doc}, skip_first_n_chunks={skip_first_n_chunks}, write_offset={write_offset}")
            vlog(f"Welford rebuilt from memmap: n_train={n_train}")

    if pass2_start_doc < n_docs:
        print("[extract] Pass 2: extract activations ...")
        print(f"[extract] chunk_size={chunk_size}, batch_size={batch_size}")
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

        activations = np.memmap(
            str(memmap_path),
            dtype=dtype,
            mode="r+" if memmap_path.exists() else "w+",
            shape=(total_tokens, hidden_size),
        )

        entry_idx = _entry_idx_at_write_offset(index_entries, write_offset)
        buffer: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any], int, int]] = []
        vlog(f"Pass 2 starting: entry_idx={entry_idx}, write_offset={write_offset}, total_tokens={total_tokens}")

        for doc_index, doc_id, text, label, doc_split in tqdm(
            iter_docs(
                data_dir, categories, label_map, doc_id_to_split,
                limit_docs=limit_docs, start_doc_index=pass2_start_doc, split=split,
            ),
            total=n_docs - pass2_start_doc if pass2_start_doc < n_docs else 0,
            desc="Pass 2 extract",
        ):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len * 4,
            )
            seq_len = enc["input_ids"].shape[1]
            ranges = get_chunk_ranges(seq_len, max_seq_len, stride)
            input_ids_full = enc["input_ids"].to(device)
            for chunk_in_doc, (chunk_start_off, chunk_len) in enumerate(ranges):
                if skip_first_n_chunks > 0:
                    skip_first_n_chunks -= 1
                    entry_idx += 1
                    continue
                if entry_idx >= len(index_entries):
                    break
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
                entry = index_entries[entry_idx]
                buffer.append((input_ids, attention_mask, entry, doc_index, chunk_in_doc))
                entry_idx += 1

                if len(buffer) >= chunk_size:
                    write_offset, n_train, welford_mean, welford_M2 = _process_buffer(
                        [(b[0], b[1], b[2]) for b in buffer],
                        batch_size, model, layer_index, activations,
                        write_offset, n_train, welford_mean, welford_M2,
                        device, dtype, tokenizer,
                    )
                    activations.flush()
                    last_doc_index = buffer[-1][3]
                    last_chunk_in_doc = buffer[-1][4]
                    _save_progress(out_dir, {
                        "pass": 2,
                        "write_offset": write_offset,
                        "last_doc_index": last_doc_index,
                        "last_chunk_in_doc": last_chunk_in_doc,
                    })
                    pct = 100.0 * write_offset / total_tokens if total_tokens else 0
                    vlog(f"Buffer flushed: write_offset={write_offset} ({pct:.1f}% of total_tokens), entries written={entry_idx}")
                    buffer.clear()

            if entry_idx >= len(index_entries):
                vlog(f"Reached end of index (entry_idx={entry_idx}), stopping doc loop")
                break

        if buffer:
            write_offset, n_train, welford_mean, welford_M2 = _process_buffer(
                [(b[0], b[1], b[2]) for b in buffer],
                batch_size, model, layer_index, activations,
                write_offset, n_train, welford_mean, welford_M2,
                device, dtype, tokenizer,
            )
            activations.flush()
            vlog(f"Processing final buffer: {len(buffer)} chunks, write_offset={write_offset}")
            last_doc_index = buffer[-1][3]
            last_chunk_in_doc = buffer[-1][4]
            _save_progress(out_dir, {
                "pass": 2,
                "write_offset": write_offset,
                "last_doc_index": last_doc_index,
                "last_chunk_in_doc": last_chunk_in_doc,
            })

        vlog(f"Pass 2 complete. Tokens written: {write_offset}/{total_tokens}")
        del activations

    if (out_dir / "progress.json").exists():
        (out_dir / "progress.json").unlink()
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
    elapsed = time.perf_counter() - start_time
    if verbose and elapsed > 0:
        tokens_per_sec = total_tokens / elapsed
        print(f"{log_prefix} Elapsed: {elapsed:.1f}s, throughput: {tokens_per_sec:.0f} tokens/s")


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
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging (data load, progress, timing)")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.verbose:
        config.setdefault("extraction", {})["verbose"] = True
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
    extract_activations(config, args.split, layer_index, for_pretrain=args.for_pretrain, verbose=args.verbose)


if __name__ == "__main__":
    main()
