"""
Weights & Biases helpers: load .env (WANDB_API_KEY), init wandb when available.
"""
import os
from pathlib import Path
from typing import Any

_dotenv_loaded = False


def _repo_root() -> Path:
    """Repo root: two levels up from src/utils (utils -> src -> repo)."""
    return Path(__file__).resolve().parent.parent.parent


def load_dotenv_for_wandb(config: dict | None = None) -> None:
    """Load .env from repo root so WANDB_API_KEY is available. Idempotent."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv
        env_path = _repo_root() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
        _dotenv_loaded = True
    except Exception:
        _dotenv_loaded = True


def is_wandb_available() -> bool:
    """Return True if WANDB_API_KEY is set (after load_dotenv_for_wandb)."""
    load_dotenv_for_wandb()
    return bool(os.environ.get("WANDB_API_KEY"))


def init_wandb(
    project: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    entity: str | None = None,
) -> bool:
    """If W&B is available, init run and update config. Return True if initialized."""
    if not is_wandb_available():
        return False
    try:
        import wandb
        kwargs = {"project": project, "name": name}
        if entity:
            kwargs["entity"] = entity
        wandb.init(**kwargs)
        if config:
            wandb.config.update(config, allow_val_change=True)
        return True
    except Exception:
        return False
