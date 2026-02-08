"""
Prepare TinyStories for the activation pipeline: download from HuggingFace and write
CSVs under data/tinystories/text/ with a 'prompt' column (compatible with data_loading).

Usage:
  python scripts/prepare_tinystories.py [--limit N] [--out_dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TINYSTORIES_HF = "roneneldan/TinyStories"
DEFAULT_OUT_DIR = "data/tinystories/text"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories CSV for extraction")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of stories to write (default: all)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for CSVs (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_tinystories] Loading {TINYSTORIES_HF} ...")
    ds = load_dataset(TINYSTORIES_HF, split="train")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[prepare_tinystories] Limited to {len(ds)} stories")
    else:
        print(f"[prepare_tinystories] Using {len(ds)} stories")

    # Build DataFrame with 'prompt' column for data_loading (HF has 'text')
    df = ds.to_pandas()[["text"]].rename(columns={"text": "prompt"})
    df = df.dropna(subset=["prompt"])
    df["prompt"] = df["prompt"].astype(str).str.strip()
    df = df[df["prompt"].str.len() > 0]
    out_path = out_dir / "tinystories.csv"
    df.to_csv(out_path, index=False)
    print(f"[prepare_tinystories] Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
