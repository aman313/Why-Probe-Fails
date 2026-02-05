"""
Run-scoped output directory: create outputs/run_<timestamp>/, rewrite config so all
output paths live under the run dir, and optionally write run_meta.json.
Data paths (data.data_dir, OOD data dirs) are left unchanged.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, load_yaml, save_json, save_yaml

RUN_DIR_PLACEHOLDER_LAYER = "BEST"
OUTPUTS_RUN_PREFIX = "outputs/run_"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _rewrite_path(value: str | Any, run_dir: Path, placeholder_layer: str | int) -> str | Any:
    """Replace placeholder layer in path if value is a string."""
    if not isinstance(value, str):
        return value
    s = value.replace(f"layer_{RUN_DIR_PLACEHOLDER_LAYER}", f"layer_{placeholder_layer}")
    return s


def _set_nested(cfg: dict[str, Any], key_path: list[str], value: Any) -> None:
    """Set cfg[key0][key1][...] = value, creating nested dicts as needed."""
    d = cfg
    for k in key_path[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[key_path[-1]] = value


def _get_nested(cfg: dict[str, Any], key_path: list[str], default: Any = None) -> Any:
    d = cfg
    for k in key_path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def rewrite_config_for_run_dir(
    config: dict[str, Any],
    run_dir: Path,
    *,
    placeholder_layer: str = RUN_DIR_PLACEHOLDER_LAYER,
) -> dict[str, Any]:
    """
    Deep-copy config and rewrite only output-related paths to live under run_dir.
    Data paths (data.data_dir, etc.) are left unchanged.
    Uses placeholder_layer (e.g. "BEST") where the best layer index is not yet known.
    """
    run_dir = Path(run_dir).resolve()
    out = copy.deepcopy(config)

    # extraction.memmap_dir -> run_dir/activations
    if "extraction" in out and "memmap_dir" in out["extraction"]:
        out["extraction"]["memmap_dir"] = str(run_dir / "activations")

    # layer_search.layer_search_output -> run_dir/best_layer.json
    if "layer_search" in out and "layer_search_output" in out["layer_search"]:
        out["layer_search"]["layer_search_output"] = str(run_dir / "best_layer.json")

    # pretrain.output_dir -> run_dir/actformer
    if "pretrain" in out and "output_dir" in out["pretrain"]:
        out["pretrain"]["output_dir"] = str(run_dir / "actformer")

    # comparison.activations.id.memmap_dir -> run_dir/activations
    comp = out.get("comparison", {}).get("activations", {})
    id_cfg = comp.get("id", {})
    if id_cfg is not None:
        if "comparison" not in out:
            out["comparison"] = {}
        if "activations" not in out["comparison"]:
            out["comparison"]["activations"] = {}
        if "id" not in out["comparison"]["activations"]:
            out["comparison"]["activations"]["id"] = {}
        out["comparison"]["activations"]["id"]["memmap_dir"] = str(run_dir / "activations")
        # index_path: run_dir/activations/layer_BEST/index.json (placeholder)
        out["comparison"]["activations"]["id"]["index_path"] = str(
            run_dir / "activations" / f"layer_{placeholder_layer}" / "index.json"
        )

    # comparison.activations.ood[].memmap_dir -> run_dir/activations_ood/<name>/layer_BEST
    ood_list = _get_nested(out, ["comparison", "activations", "ood"])
    if ood_list and isinstance(ood_list, list):
        new_ood = []
        for entry in ood_list:
            if not isinstance(entry, dict):
                new_ood.append(entry)
                continue
            name = entry.get("name", "ood")
            new_ood.append({
                **entry,
                "memmap_dir": str(run_dir / "activations_ood" / name / f"layer_{placeholder_layer}"),
            })
        _set_nested(out, ["comparison", "activations", "ood"], new_ood)

    # comparison.actformer_checkpoint -> run_dir/actformer/best.pt
    if "comparison" in out and "actformer_checkpoint" in out["comparison"]:
        out["comparison"]["actformer_checkpoint"] = str(run_dir / "actformer" / "best.pt")

    # comparison.output_dir -> run_dir/probe_comparison
    if "comparison" in out and "output_dir" in out["comparison"]:
        out["comparison"]["output_dir"] = str(run_dir / "probe_comparison")

    return out


def update_run_config_after_layer_search(
    run_dir: Path,
    best_layer_index: int,
) -> None:
    """
    Update run_dir/config.yaml: replace placeholder layer (BEST) with actual best_layer_index
    in comparison.activations.id.index_path and comparison.activations.ood[].memmap_dir.
    """
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return
    config = load_yaml(config_path)

    # Update id.index_path
    id_cfg = _get_nested(config, ["comparison", "activations", "id"])
    if id_cfg and isinstance(id_cfg, dict):
        idx_path = id_cfg.get("index_path")
        if idx_path and RUN_DIR_PLACEHOLDER_LAYER in str(idx_path):
            id_cfg["index_path"] = _rewrite_path(idx_path, run_dir, best_layer_index)
        # id.memmap_dir stays run_dir/activations (no placeholder)

    # Update ood[].memmap_dir
    ood_list = _get_nested(config, ["comparison", "activations", "ood"])
    if ood_list and isinstance(ood_list, list):
        new_ood = []
        for entry in ood_list:
            if not isinstance(entry, dict):
                new_ood.append(entry)
                continue
            memmap_dir = entry.get("memmap_dir", "")
            if memmap_dir and RUN_DIR_PLACEHOLDER_LAYER in str(memmap_dir):
                entry = {**entry, "memmap_dir": _rewrite_path(memmap_dir, run_dir, best_layer_index)}
            new_ood.append(entry)
        _set_nested(config, ["comparison", "activations", "ood"], new_ood)

    save_yaml(config, config_path)


def create_run_dir(
    user_config_path: str | Path,
    outputs_root: str | Path = "outputs",
    *,
    write_run_meta: bool = True,
) -> tuple[Path, Path]:
    """
    Create a new run directory under outputs_root/run_<timestamp>/,
    write the rewritten config to run_dir/config.yaml, optionally run_meta.json.
    Returns (run_dir_path, run_config_path).
    """
    outputs_root = Path(outputs_root).resolve()
    ensure_dir(outputs_root)
    ts = _timestamp()
    run_dir = outputs_root / f"run_{ts}"
    ensure_dir(run_dir)

    config = load_yaml(user_config_path)
    run_config = rewrite_config_for_run_dir(config, run_dir)
    config_path = run_dir / "config.yaml"
    save_yaml(run_config, config_path)

    if write_run_meta:
        meta: dict[str, Any] = {
            "timestamp": ts,
            "run_dir": str(run_dir),
            "original_config": str(Path(user_config_path).resolve()),
        }
        try:
            meta["git_sha"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(user_config_path).resolve().parent,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except Exception:
            pass
        save_json(meta, run_dir / "run_meta.json")

    return run_dir, config_path


def is_run_dir_config(config_path: str | Path) -> bool:
    """Return True if config_path is inside outputs/run_* (reproduction mode)."""
    p = Path(config_path).resolve()
    parts = p.parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts) and parts[idx + 1].startswith("run_"):
            return True
    return False


def get_run_dir_from_config_path(config_path: str | Path) -> Path | None:
    """If config_path is under outputs/run_*, return that run dir; else None."""
    p = Path(config_path).resolve()
    for parent in p.parents:
        if parent.name.startswith("run_") and parent.parent.name == "outputs":
            return parent
    return None


def run_ood_extraction(
    run_dir: Path,
    best_layer_index: int,
    ood_entries: list[dict[str, Any]],
    ood_config_dir: str | Path = "configs",
) -> None:
    """
    For each OOD entry (from comparison.activations.ood), load configs/ood_<name>.yaml,
    override extraction.memmap_dir to run_dir/activations_ood/<name>/layer_<best>,
    run extract_activations, then copy ID train mean/std into each OOD layer dir.
    """
    from src.extract_activations import extract_activations

    run_dir = Path(run_dir).resolve()
    ood_config_dir = Path(ood_config_dir).resolve()
    id_layer_dir = run_dir / "activations" / f"layer_{best_layer_index}"
    id_mean = id_layer_dir / "train_mean.npy"
    id_std = id_layer_dir / "train_std.npy"

    for entry in ood_entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "ood")
        ood_config_path = ood_config_dir / f"ood_{name}.yaml"
        if not ood_config_path.exists():
            continue
        config = load_yaml(ood_config_path)
        ood_layer_dir = run_dir / "activations_ood" / name / f"layer_{best_layer_index}"
        ensure_dir(ood_layer_dir)
        config["extraction"] = config.get("extraction", {})
        config["extraction"]["memmap_dir"] = str(ood_layer_dir.parent)
        extract_activations(config, "all", best_layer_index)
        if id_mean.exists() and id_std.exists():
            shutil.copy2(id_mean, ood_layer_dir / "train_mean.npy")
            shutil.copy2(id_std, ood_layer_dir / "train_std.npy")
