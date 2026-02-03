"""Utilities for mapping config dtype strings to numpy dtypes."""
from __future__ import annotations

import numpy as np


def resolve_numpy_dtype(dtype_str: str) -> np.dtype:
    """
    Map dtype strings to numpy dtypes. Handles bfloat16 by mapping to float32
    (numpy has no native bfloat16).
    """
    s = str(dtype_str).lower()
    if s in {"float16", "fp16", "half"}:
        return np.float16
    if s in {"bfloat16", "bf16"}:
        return np.float32
    if s in {"float64", "fp64", "double"}:
        return np.float64
    return np.float32
