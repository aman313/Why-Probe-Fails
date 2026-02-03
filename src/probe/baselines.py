"""
Baselines: raw activation probe (no ActFormer), optional PCA + linear.
"""
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import compute_metrics


def run_raw_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Linear probe on raw (pooled) features. Returns metrics on test."""
    if len(np.unique(y_train)) < 2 or len(X_train) == 0 or len(X_test) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else clf.predict_proba(X_test)
    return compute_metrics(y_test, y_pred, y_prob, ["accuracy", "macro_f1", "auroc"])


def run_pca_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_components: int = 64,
) -> dict[str, float]:
    """PCA on train, then linear probe on projected features."""
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
    n_components = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    if n_components < 1:
        return run_raw_probe(X_train, y_train, X_val, y_val, X_test, y_test)
    pca = PCA(n_components=n_components, random_state=42)
    X_train_p = pca.fit_transform(X_train)
    X_val_p = pca.transform(X_val)
    X_test_p = pca.transform(X_test)
    scaler = StandardScaler()
    X_train_p = scaler.fit_transform(X_train_p)
    X_test_p = scaler.transform(X_test_p)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train_p, y_train)
    y_pred = clf.predict(X_test_p)
    y_prob = clf.predict_proba(X_test_p)[:, 1] if len(np.unique(y_test)) == 2 else clf.predict_proba(X_test_p)
    return compute_metrics(y_test, y_pred, y_prob, ["accuracy", "macro_f1", "auroc"])
