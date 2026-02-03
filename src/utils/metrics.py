"""Classification metrics: accuracy, macro F1, AUROC."""
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    metrics_list: list[str],
) -> dict[str, float]:
    """
    Compute requested classification metrics.
    y_true, y_pred: (N,) integer labels.
    y_prob: (N,) or (N, n_classes) probabilities for positive / each class (for AUROC).
    metrics_list: e.g. ["accuracy", "macro_f1", "auroc"].
    """
    out: dict[str, float] = {}
    if "accuracy" in metrics_list:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "macro_f1" in metrics_list:
        out["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
    if "auroc" in metrics_list and y_prob is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            try:
                out["auroc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                out["auroc"] = float("nan")
        else:
            try:
                out["auroc"] = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except (ValueError, TypeError):
                out["auroc"] = float("nan")
    return out
