"""Unit tests for run_dir helpers: ensure_unique_output_dir, resolve_actformer_checkpoint_dir."""
import re
import tempfile
from pathlib import Path

import pytest

from src.run_dir import (
    ensure_unique_output_dir,
    is_under_run_dir,
    resolve_actformer_checkpoint_dir,
)
from src.utils.io import load_yaml


def test_ensure_unique_output_dir_creates_run_timestamp_subdir():
    """ensure_unique_output_dir('outputs/actformer') returns outputs/actformer/run_<timestamp> and creates it."""
    with tempfile.TemporaryDirectory() as tmp:
        base = (Path(tmp) / "outputs" / "actformer").resolve()
        result = ensure_unique_output_dir(base)
        assert result.is_dir()
        assert result.parent.resolve() == base.resolve()
        assert re.match(r"run_\d{8}_\d{6}", result.name), f"Expected run_<timestamp>, got {result.name}"
        assert result.resolve() == (base / result.name).resolve()


def test_ensure_unique_output_dir_writes_config_when_provided():
    """When config is passed and a new run_<timestamp> is created, config.yaml is written with that config."""
    with tempfile.TemporaryDirectory() as tmp:
        base = (Path(tmp) / "outputs" / "actformer").resolve()
        config = {"seed": 42, "pretrain": {"lr": 1e-4}}
        result = ensure_unique_output_dir(base, config=config)
        assert result.is_dir()
        config_path = result / "config.yaml"
        assert config_path.exists()
        loaded = load_yaml(config_path)
        assert loaded == config


def test_ensure_unique_output_dir_no_config_file_when_config_none():
    """When config is None, no config.yaml is created (backward compatibility)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = (Path(tmp) / "outputs" / "actformer").resolve()
        result = ensure_unique_output_dir(base, config=None)
        assert result.is_dir()
        assert not (result / "config.yaml").exists()


def test_ensure_unique_output_dir_under_run_dir_unchanged():
    """ensure_unique_output_dir('outputs/run_123/actformer') returns the same path (no extra run_*)."""
    with tempfile.TemporaryDirectory() as tmp:
        run_actformer = (Path(tmp) / "outputs" / "run_123" / "actformer").resolve()
        run_actformer.mkdir(parents=True)
        result = ensure_unique_output_dir(run_actformer)
        assert result.resolve() == run_actformer.resolve()
        assert result.is_dir()
        assert "run_" in result.parent.name  # parent is run_123


def test_resolve_actformer_checkpoint_dir_prefers_base():
    """If base_dir/best.pt exists, resolve_actformer_checkpoint_dir returns base_dir."""
    with tempfile.TemporaryDirectory() as tmp:
        base = (Path(tmp) / "outputs" / "actformer").resolve()
        base.mkdir(parents=True)
        (base / "best.pt").write_text("x")
        assert resolve_actformer_checkpoint_dir(base).resolve() == base.resolve()


def test_resolve_actformer_checkpoint_dir_uses_latest_run_subdir():
    """resolve_actformer_checkpoint_dir with base that has run_*/best.pt returns the latest run dir."""
    with tempfile.TemporaryDirectory() as tmp:
        base = (Path(tmp) / "outputs" / "actformer").resolve()
        base.mkdir(parents=True)
        r1 = base / "run_20250101_120000"
        r2 = base / "run_20250102_120000"
        r1.mkdir()
        r2.mkdir()
        (r1 / "best.pt").write_text("a")
        (r2 / "best.pt").write_text("b")
        result = resolve_actformer_checkpoint_dir(base)
        assert result.resolve() == r2.resolve()


def test_is_under_run_dir():
    """is_under_run_dir is True for paths under outputs/run_*, False otherwise."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        under_run = root / "outputs" / "run_123" / "actformer"
        under_run.mkdir(parents=True)
        not_under = root / "outputs" / "actformer"
        not_under.mkdir(parents=True)
        assert is_under_run_dir(under_run) is True
        assert is_under_run_dir(not_under) is False
        assert is_under_run_dir(under_run / "best.pt") is True
