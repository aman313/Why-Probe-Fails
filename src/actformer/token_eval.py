"""Token-level evaluation for ActFormer replacement.

Replaced path uses first-token-only lower stack: run embedding + layers 0..(best_layer-1)
only on the first token to get boundary_0; then ActFormer autoregressively extends the
boundary sequence, and only the upper stack (layers best_layer..end) runs on that
boundary to produce next-token logits. Baseline is full free-running with argmax.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable
try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from src.actformer.model import ActFormer
from src.run_dir import resolve_actformer_checkpoint_dir
from src.utils.data_loading import build_doc_id_to_split, iter_docs, load_docs_metadata, split_docs
from src.utils.device import get_device
from src.utils.io import ensure_dir, load_json, load_yaml, save_json


def compute_topk_jaccard_similarity(topk_a: torch.Tensor, topk_b: torch.Tensor) -> torch.Tensor:
    """Return per-position Jaccard similarity for top-k ids tensors (B,T,K)."""
    if topk_a.shape != topk_b.shape:
        raise ValueError(f"shape mismatch: {tuple(topk_a.shape)} vs {tuple(topk_b.shape)}")
    k = topk_a.shape[-1]
    if k <= 0:
        raise ValueError("k must be positive")
    inter = (topk_a.unsqueeze(-1) == topk_b.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float()
    union = (2 * k) - inter
    return inter / union.clamp(min=1.0)


def position_accuracy_curve(matches: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (correct_per_pos, total_per_pos)."""
    if matches.shape != valid_mask.shape:
        raise ValueError("matches and valid_mask must have same shape")
    return (matches & valid_mask).sum(dim=0).long(), valid_mask.sum(dim=0).long()


def _resolve_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("unsupported architecture for layer replacement")


def _find_eval_layer_index(config: dict[str, Any], num_layers: int) -> int:
    idx = int(config.get("model", {}).get("layer_index", 0))
    best_path = Path(config.get("layer_search", {}).get("layer_search_output", "outputs/best_layer.json"))
    if best_path.exists():
        idx = int(load_json(best_path).get("layer_index", idx))
    return max(0, min(idx, num_layers - 1))


def _load_eval_docs(config: dict[str, Any], split: str, max_docs: int | None) -> list[dict[str, Any]]:
    data_cfg = config.get("data", {})
    meta = load_docs_metadata(
        data_dir=data_cfg.get("data_dir", "data/RS1"),
        categories=data_cfg.get("categories", ["benign", "malicious"]),
        label_map=data_cfg.get("label_map", {"benign": 0, "malicious": 1}),
        limit_docs=data_cfg.get("limit_docs"),
    )
    train_df, val_df, test_df = split_docs(
        meta,
        data_cfg.get("split_by_file", False),
        data_cfg.get("train_ratio", 0.7),
        data_cfg.get("val_ratio", 0.15),
        data_cfg.get("test_ratio", 0.15),
        seed=config.get("seed", 42),
    )
    doc_id_to_split = build_doc_id_to_split(train_df, val_df, test_df)
    docs: list[dict[str, Any]] = []
    for _, doc_id, text, label, doc_split in iter_docs(
        data_cfg.get("data_dir", "data/RS1"),
        data_cfg.get("categories", ["benign", "malicious"]),
        data_cfg.get("label_map", {"benign": 0, "malicious": 1}),
        doc_id_to_split,
        limit_docs=data_cfg.get("limit_docs"),
        split=split,
    ):
        docs.append({"doc_id": int(doc_id), "label": int(label), "split": str(doc_split), "text": text})
        if max_docs is not None and len(docs) >= max_docs:
            break
    return docs


def _load_normalization_stats(config: dict[str, Any], layer_index: int, hidden_size: int) -> tuple[np.ndarray, np.ndarray]:
    eval_cfg = config.get("token_eval", {})
    ext_cfg = config.get("extraction", {})
    pretrain_cfg = config.get("pretrain", {})
    stats_memmap_dir = eval_cfg.get("normalization_memmap_dir") or pretrain_cfg.get("memmap_dir") or ext_cfg.get(
        "memmap_dir", "outputs/activations"
    )
    layer_dir = Path(stats_memmap_dir) / f"layer_{layer_index}"
    mean_path = layer_dir / "train_mean.npy"
    std_path = layer_dir / "train_std.npy"
    if mean_path.exists() and std_path.exists():
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
    else:
        mean = np.zeros(hidden_size, dtype=np.float32)
        std = np.ones(hidden_size, dtype=np.float32)
    return mean, np.where(std < 1e-8, 1.0, std)


def _load_actformer(config: dict[str, Any], hidden_size: int, device: torch.device) -> tuple[ActFormer, Path]:
    eval_cfg = config.get("token_eval", {})
    af_cfg = config.get("actformer", {})
    checkpoint = eval_cfg.get("actformer_checkpoint") or config.get("comparison", {}).get("actformer_checkpoint")
    if checkpoint is None:
        checkpoint = resolve_actformer_checkpoint_dir(
            Path(config.get("pretrain", {}).get("output_dir", "outputs/actformer"))
        ) / "best.pt"
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = resolve_actformer_checkpoint_dir(ckpt_path) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ActFormer checkpoint not found: {ckpt_path}")
    model = ActFormer(
        d_in=hidden_size,
        d_model=af_cfg.get("d_model", 256),
        n_layers=af_cfg.get("n_layers", 2),
        n_heads=af_cfg.get("n_heads", 4),
        ff_mult=af_cfg.get("ff_mult", 4),
        dropout=0.0,
        loss_type=af_cfg.get("loss_type", "mse"),
        causal=af_cfg.get("causal", True),
    ).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    return model, ckpt_path


def _build_replacement_hidden(
    actformer: ActFormer,
    boundary_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if boundary_hidden.shape[1] < 2:
        return boundary_hidden
    x = (boundary_hidden[:, :-1, :] - mean) / std
    mask = attention_mask[:, :-1].bool()
    mu_norm, _, _ = actformer(x=x, mask=mask)
    mu = mu_norm * std + mean
    out = boundary_hidden.clone()
    out[:, 1:, :] = mu
    return out


def _forward_with_layer_input_replacement(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_idx: int,
    replacement_hidden: torch.Tensor,
) -> torch.Tensor:
    layer = _resolve_layers(model)[target_layer_idx]

    def _hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if len(args) > 0 and torch.is_tensor(args[0]):
            new_args = (replacement_hidden.to(device=args[0].device, dtype=args[0].dtype),) + tuple(args[1:])
            return new_args, kwargs
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = replacement_hidden.to(
                device=kwargs["hidden_states"].device,
                dtype=kwargs["hidden_states"].dtype,
            )
        return args, kwargs

    h = layer.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    finally:
        h.remove()
    return out.logits


def _forward_from_layer(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    from_layer_idx: int,
) -> torch.Tensor:
    """Run only layers [from_layer_idx:] + final norm + lm_head on hidden_states.

    NOTE: This bypasses the model's internal mask preparation (e.g. create_causal_mask /
    _prepare_4d_causal_attention_mask) and does NOT compute rotary position embeddings
    (RoPE). It works correctly for GPT-2-style models but will produce wrong results for
    Llama/Qwen architectures that use RoPE and 4D causal masks. For those models use
    _forward_with_layer_input_replacement instead.
    """
    layers = _resolve_layers(model)
    if from_layer_idx >= len(layers):
        raise ValueError(f"from_layer_idx {from_layer_idx} >= num_layers {len(layers)}")
    h = hidden_states
    for layer in layers[from_layer_idx:]:
        out = layer(h, attention_mask=attention_mask, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out
    # Final norm: GPT-2 has transformer.ln_f, Llama/Mistral have model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        h = model.transformer.ln_f(h)
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        h = model.model.norm(h)
    # Else: no final norm (e.g. minimal test fake); use h as-is
    # LM head
    lm_head = getattr(model, "get_output_embeddings", lambda: None)()
    if lm_head is None and hasattr(model, "lm_head"):
        lm_head = model.lm_head
    if lm_head is None:
        raise ValueError("unsupported architecture: no lm_head found")
    logits = lm_head(h)
    return logits


def _safe_decode(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        return str(token_id)


def evaluate_token_replacement(config: dict[str, Any]) -> dict[str, Any]:
    if AutoTokenizer is None or AutoModelForCausalLM is None or AutoConfig is None:
        raise ImportError("transformers is required for token evaluation")
    eval_cfg = config.get("token_eval", {})
    model_cfg = config.get("model", {})
    split = eval_cfg.get("split", "test")
    max_docs = eval_cfg.get("max_docs")
    max_seq_len = int(eval_cfg.get("max_seq_len", 256))
    batch_size = int(eval_cfg.get("batch_size", 4))
    topk_values = sorted({int(k) for k in eval_cfg.get("topk_values", [5, 10]) if int(k) > 0})
    save_examples = int(eval_cfg.get("save_examples", 200))
    output_dir = ensure_dir(Path(eval_cfg.get("output_dir", "outputs/actformer_token_eval")))

    docs = _load_eval_docs(config, split=split, max_docs=max_docs)
    if not docs:
        raise ValueError(f"No documents found for split={split}")

    hf_kwargs: dict[str, Any] = {"trust_remote_code": True}
    cache_dir = model_cfg.get("cache_dir")
    if cache_dir:
        hf_kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    base_model_name = model_cfg.get("base_model_name", "distilgpt2")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **hf_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        output_hidden_states=True,
        **hf_kwargs,
    ).to(device)
    model.eval()

    num_layers = len(_resolve_layers(model))
    eval_layer_idx = _find_eval_layer_index(config, num_layers)
    depth = eval_layer_idx + 1  # single enforced depth: replace subnetwork up to best_layer-1
    if depth > num_layers or depth < 1:
        raise ValueError(f"best_layer {eval_layer_idx} yields depth {depth} outside [1, {num_layers}]")
    depths = [depth]

    hidden_size = AutoConfig.from_pretrained(base_model_name, **hf_kwargs).hidden_size
    actformer, ckpt_path = _load_actformer(config, hidden_size=hidden_size, device=device)

    mean_np, std_np = _load_normalization_stats(config, layer_index=eval_layer_idx, hidden_size=hidden_size)
    mean_t = torch.from_numpy(mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    std_t = torch.from_numpy(std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)

    max_steps = int(eval_cfg.get("max_steps") or max_seq_len)
    if max_steps < 2:
        max_steps = 2

    vocab_size = int(model.config.vocab_size)
    topk_values = [min(k, vocab_size) for k in topk_values]

    stats: dict[str, Any] = {
        "match_correct": 0,
        "match_total": 0,
        "pos_correct": [],
        "pos_total": [],
        "jaccard_sum": {k: 0.0 for k in topk_values},
        "jaccard_count": {k: 0 for k in topk_values},
        "errors": [],
        "pairs": Counter(),
    }

    for start in tqdm(range(0, len(docs), batch_size), desc="token_eval"):
        batch_docs = docs[start : start + batch_size]
        enc = tokenizer(
            [d["text"] for d in batch_docs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        seq_len = input_ids.size(1)
        steps_this_batch = min(max_steps, max(0, seq_len - 1))
        if steps_this_batch < 1:
            continue

        prefix_base = input_ids[:, :1].clone()
        prefix_rep = input_ids[:, :1].clone()
        s = stats

        # First-token-only: run lower stack once on first token to get boundary at position 0.
        with torch.no_grad():
            out_first = model(
                input_ids=input_ids[:, :1],
                attention_mask=attention_mask[:, :1],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        boundary_sequence = out_first.hidden_states[depth - 1][:, 0:1, :].clone()  # (B, 1, H)

        for step in range(1, steps_this_batch + 1):
            attn = attention_mask[:, :step].to(device)
            with torch.no_grad():
                out_base = model(
                    input_ids=prefix_base,
                    attention_mask=attn,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                logits_base = out_base.logits[:, -1, :]
                next_base = logits_base.argmax(dim=-1)

                # Replaced path: extend boundary by one position via ActFormer, then upper stack only.
                # boundary_sequence is (B, t, H) with t = step at loop start (positions 0..step-1).
                boundary_padded = torch.cat(
                    [boundary_sequence, boundary_sequence[:, -1:, :]], dim=1
                )  # (B, step+1, H)
                attn_ext = attention_mask[:, : step + 1].to(device)
                replaced_full = _build_replacement_hidden(
                    actformer, boundary_padded, attn_ext, mean_t, std_t
                )
                boundary_t = replaced_full[:, -1:, :]
                boundary_sequence = torch.cat([boundary_sequence, boundary_t], dim=1)  # (B, step+1, H)
                # Use hook approach so the model handles RoPE + causal mask prep correctly for
                # all architectures (Qwen, Llama, GPT-2, etc.). Extend prefix_rep with a
                # placeholder token so its length matches boundary_sequence (step+1); the hook
                # replaces hidden states at the boundary layer before any upper-stack computation.
                prefix_for_upper = torch.cat([prefix_rep, prefix_rep[:, -1:]], dim=1)
                replaced_logits_full = _forward_with_layer_input_replacement(
                    model, prefix_for_upper, attn_ext, depth - 1, boundary_sequence
                )
                logits_rep = replaced_logits_full[:, -1, :]
                next_rep = logits_rep.argmax(dim=-1)

            valid_step = attention_mask[:, step].bool() if step < attention_mask.size(1) else attn[:, -1].bool()
            matches = (next_base == next_rep) & valid_step

            s["match_correct"] += int(matches.sum().item())
            s["match_total"] += int(valid_step.sum().item())

            pos_idx = step - 1
            while len(s["pos_correct"]) <= pos_idx:
                s["pos_correct"].append(0)
                s["pos_total"].append(0)
            s["pos_correct"][pos_idx] += int((matches & valid_step).sum().item())
            s["pos_total"][pos_idx] += int(valid_step.sum().item())

            max_k = topk_values[-1]
            topk_base = torch.topk(logits_base, k=max_k, dim=-1).indices.unsqueeze(1)
            topk_rep = torch.topk(logits_rep, k=max_k, dim=-1).indices.unsqueeze(1)
            jac_step = compute_topk_jaccard_similarity(topk_base, topk_rep).squeeze(1)
            for k in topk_values:
                topk_b = torch.topk(logits_base, k=k, dim=-1).indices.unsqueeze(1)
                topk_r = torch.topk(logits_rep, k=k, dim=-1).indices.unsqueeze(1)
                jk = compute_topk_jaccard_similarity(topk_b, topk_r).squeeze(1)
                s["jaccard_sum"][k] += float((jk * valid_step.float()).sum().item())
                s["jaccard_count"][k] += int(valid_step.sum().item())

            mismatch_mask = (~(next_base == next_rep)) & valid_step
            jac_max = jac_step
            for b in range(mismatch_mask.size(0)):
                if not mismatch_mask[b].item():
                    continue
                base_id = int(next_base[b].item())
                rep_id = int(next_rep[b].item())
                s["pairs"][(base_id, rep_id)] += 1
                if len(s["errors"]) >= save_examples:
                    continue
                ctx_left = max(0, step - 5)
                ctx_right = step
                ctx_ids = prefix_base[b, ctx_left:ctx_right]
                s["errors"].append(
                    {
                        "replacement_depth": depth,
                        "doc_id": int(batch_docs[b]["doc_id"]),
                        "position": pos_idx,
                        "gold_token_id": base_id,
                        "baseline_token_id": base_id,
                        "replaced_token_id": rep_id,
                        "gold_token": _safe_decode(tokenizer, base_id),
                        "baseline_token": _safe_decode(tokenizer, base_id),
                        "replaced_token": _safe_decode(tokenizer, rep_id),
                        "topk_jaccard_similarity": float(jac_max[b].item()),
                        "baseline_topk_tokens": [_safe_decode(tokenizer, x) for x in topk_base[b, 0].tolist()],
                        "replaced_topk_tokens": [_safe_decode(tokenizer, x) for x in topk_rep[b, 0].tolist()],
                        "context_window": tokenizer.decode(
                            ctx_ids.tolist(),
                            clean_up_tokenization_spaces=False,
                        ),
                    }
                )

            prefix_base = torch.cat([prefix_base, next_base.unsqueeze(1)], dim=1)
            prefix_rep = torch.cat([prefix_rep, next_rep.unsqueeze(1)], dim=1)

    summary_depths: dict[str, Any] = {}
    position_depths: dict[str, Any] = {}
    all_errors: list[dict[str, Any]] = []
    examples = ["# ActFormer Token Evaluation Examples", ""]

    for depth in depths:
        s = stats  # single depth: stats is flat
        total = max(s["match_total"], 1)
        acc = s["match_correct"] / total
        row = {
            "replacement_depth": depth,
            "token_match_accuracy": acc,
            "mismatch_rate": 1.0 - acc,
            "total_positions": s["match_total"],
            "topk_jaccard_similarity": {},
            "topk_jaccard_distance": {},
            "top_disagreement_pairs": [
                {"baseline_token_id": int(a), "replaced_token_id": int(b), "count": int(c)}
                for (a, b), c in s["pairs"].most_common(20)
            ],
        }
        for k in topk_values:
            denom = max(s["jaccard_count"][k], 1)
            sim = s["jaccard_sum"][k] / denom
            row["topk_jaccard_similarity"][str(k)] = sim
            row["topk_jaccard_distance"][str(k)] = 1.0 - sim
        summary_depths[str(depth)] = row

        pos_curve: list[dict[str, Any]] = []
        for i, (c, n) in enumerate(zip(s["pos_correct"], s["pos_total"])):
            if n > 0:
                pos_curve.append({"position": i, "accuracy": c / n, "count": n})
        position_depths[str(depth)] = pos_curve

        all_errors.extend(s["errors"])
        examples.append(f"## Replacement Depth {depth}")
        if not s["errors"]:
            examples.append("- No mismatches captured.")
            examples.append("")
            continue
        for ex in s["errors"][: min(20, len(s["errors"]))]:
            examples.append(
                f"- doc_id={ex['doc_id']} pos={ex['position']} "
                f"gold={ex['gold_token']!r} baseline={ex['baseline_token']!r} "
                f"replaced={ex['replaced_token']!r} jaccard={ex['topk_jaccard_similarity']:.3f}"
            )
            examples.append(f"  - context: {ex['context_window']!r}")
        examples.append("")

    summary = {
        "base_model_name": base_model_name,
        "actformer_checkpoint": str(ckpt_path),
        "split": split,
        "num_docs": len(docs),
        "max_seq_len": max_seq_len,
        "replacement_depths": depths,
        "topk_values": topk_values,
        "layer_index_for_stats": eval_layer_idx,
        "best_layer": eval_layer_idx,
        "evaluation_mode": "free_running_argmax",
        "depth_metrics": summary_depths,
    }
    save_json(summary, output_dir / "summary.json")
    save_json(position_depths, output_dir / "position_metrics.json")
    with open(output_dir / "errors.jsonl", "w") as f:
        for row in all_errors:
            f.write(json.dumps(row) + "\n")
    (output_dir / "examples.md").write_text("\n".join(examples) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="ActFormer token-level replacement evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--actformer_run_dir",
        type=str,
        default=None,
        help="Path to an ActFormer pretrain run dir (expects best.pt; uses config.yaml there if present)",
    )
    args = parser.parse_args()
    if args.actformer_run_dir:
        run_dir = Path(args.actformer_run_dir).resolve()
        run_cfg = run_dir / "config.yaml"
        config = load_yaml(run_cfg) if run_cfg.exists() else load_yaml(args.config)
        config.setdefault("token_eval", {})["actformer_checkpoint"] = str(run_dir / "best.pt")
    else:
        config = load_yaml(args.config)
    out = evaluate_token_replacement(config)
    out_path = Path(config.get("token_eval", {}).get("output_dir", "outputs/actformer_token_eval")) / "summary.json"
    print(f"[token_eval] done: docs={out['num_docs']} depths={out['replacement_depths']} summary={out_path}")


if __name__ == "__main__":
    main()
