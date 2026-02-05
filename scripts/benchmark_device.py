"""
Benchmark CPU vs MPS inference speed. Runs a small number of LM forward passes
on CPU and (if available) MPS and reports wall-clock time and speedup.

Usage (from repo root): PYTHONPATH=. python scripts/benchmark_device.py [--config configs/tiny.yaml] [--n_warmup 2] [--n_iters 20]
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.io import load_yaml


def _mps_sync() -> None:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def run_forward_passes(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    seq_len: int = 32,
    batch_size: int = 2,
    n_warmup: int = 2,
    n_iters: int = 20,
) -> float:
    """Run n_warmup + n_iters forward passes, return mean time per pass in seconds (excluding warmup)."""
    model.eval()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    # Fixed small input
    input_ids = torch.full(
        (batch_size, seq_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    # Ensure some non-pad tokens so we do real work
    input_ids[:, 0] = tokenizer.eos_token_id or 1
    attention_mask = (input_ids != pad_id).long()

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if device.type == "mps":
            _mps_sync()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if device.type == "mps":
            _mps_sync()
        elif device.type == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / n_iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU vs MPS inference")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml")
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_iters", type=int, default=20)
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    model_cfg = config.get("model", {})
    base_model_name = model_cfg.get("base_model_name", "distilgpt2")
    cache_dir = model_cfg.get("cache_dir")
    kwargs = {"trust_remote_code": True}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser().resolve())

    print(f"[benchmark] Model: {base_model_name}")
    print(f"[benchmark] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load once on CPU (we will move to device per run)
    print(f"[benchmark] Loading model (on CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=None,
        output_hidden_states=True,
        **kwargs,
    ).eval()

    seq_len = 32
    batch_size = 2
    n_warmup = args.n_warmup
    n_iters = args.n_iters

    # CPU (model already on CPU)
    print(f"[benchmark] Running CPU: {n_warmup} warmup, {n_iters} iters ...")
    cpu_time = run_forward_passes(
        model, tokenizer, torch.device("cpu"), seq_len=seq_len, batch_size=batch_size,
        n_warmup=n_warmup, n_iters=n_iters,
    )
    print(f"  CPU: {cpu_time*1000:.2f} ms / forward")

    mps_available = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )

    if mps_available:
        model = model.to("mps")
        print(f"[benchmark] Running MPS: {n_warmup} warmup, {n_iters} iters ...")
        mps_time = run_forward_passes(
            model, tokenizer, torch.device("mps"), seq_len=seq_len, batch_size=batch_size,
            n_warmup=n_warmup, n_iters=n_iters,
        )
        print(f"  MPS: {mps_time*1000:.2f} ms / forward")
        speedup = cpu_time / mps_time
        print(f"[benchmark] MPS speedup over CPU: {speedup:.2f}x")
        if speedup < 1.0:
            print(f"[benchmark] Note: MPS is slower than CPU for this workload (common for small models/batches).")
    else:
        print("[benchmark] MPS not available, skipping MPS run.")
        print("  (Use a Mac with Apple Silicon and PyTorch built with MPS to compare.)")


if __name__ == "__main__":
    main()
