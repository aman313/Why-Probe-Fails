"""I/O helpers: load/save YAML, JSON; ensure directory exists."""
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist. Return path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    """Save a dict (or list) to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    """Save to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
