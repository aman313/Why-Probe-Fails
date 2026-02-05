"""Reproducibility: set seeds for PyTorch, numpy, and Python random."""
import random

import numpy as np
import torch

from .device import get_device


def set_seed(seed: int) -> None:
    """Set seed for PyTorch, numpy, and random. Call at start of each run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if get_device().type == "mps" and getattr(torch.mps, "manual_seed", None) is not None:
        torch.mps.manual_seed(seed)