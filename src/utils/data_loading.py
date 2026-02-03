"""Load CSVs from data_dir by category; build doc table and train/val/test splits."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def assert_disjoint_splits(
    train_ids: list[int] | set[int],
    val_ids: list[int] | set[int],
    test_ids: list[int] | set[int],
) -> None:
    """Ensure train/val/test document IDs are disjoint (no leakage)."""
    t, v, te = set(train_ids), set(val_ids), set(test_ids)
    assert t & v == set(), "Train and val doc_ids must be disjoint"
    assert t & te == set(), "Train and test doc_ids must be disjoint"
    assert v & te == set(), "Val and test doc_ids must be disjoint"


def load_docs_from_data_dir(
    data_dir: str | Path,
    categories: list[str],
    label_map: dict[str, int],
    limit_docs: int | None = None,
) -> pd.DataFrame:
    """
    Load all CSVs from data_dir/{category}/*.csv. Each row = one document.
    Returns DataFrame with columns: doc_id, text, label, source_file.
    """
    data_dir = Path(data_dir)
    rows: list[dict[str, Any]] = []
    doc_id = 0
    for category in categories:
        label = label_map.get(category, -1)
        cat_dir = data_dir / category
        if not cat_dir.exists():
            continue
        for csv_path in sorted(cat_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "prompt" not in df.columns:
                continue
            source_file = f"{category}/{csv_path.name}"
            for _, row in df.iterrows():
                text = row["prompt"]
                if pd.isna(text) or str(text).strip() == "":
                    continue
                rows.append(
                    {
                        "doc_id": doc_id,
                        "text": str(text).strip(),
                        "label": label,
                        "source_file": source_file,
                    }
                )
                doc_id += 1
                if limit_docs is not None and len(rows) >= limit_docs:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def split_docs(
    df: pd.DataFrame,
    split_by_file: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split documents into train/val/test. Returns (train_df, val_df, test_df).
    - If split_by_file: assign each source_file to one split (stratified by label).
    - Else: stratified split by doc_id (row-level).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = __import__("random").Random(seed)

    if split_by_file:
        files = df["source_file"].unique().tolist()
        labels_per_file = df.groupby("source_file")["label"].first()
        file_labels = [labels_per_file[f] for f in files]
        n = len(files)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train_files = {files[indices[i]] for i in range(n_train)}
        val_files = {files[indices[i]] for i in range(n_train, n_train + n_val)}
        test_files = {files[indices[i]] for i in range(n_train + n_val, n)}
        train_df = df[df["source_file"].isin(train_files)].copy()
        val_df = df[df["source_file"].isin(val_files)].copy()
        test_df = df[df["source_file"].isin(test_files)].copy()
    else:
        from sklearn.model_selection import train_test_split

        ids = df["doc_id"].values
        n = len(ids)
        if n < 3:
            # Minimal split: 1 train, 1 val, 0 test (or 1/1/0)
            id_train = ids[:1]
            id_val = ids[1:2] if n >= 2 else np.array([], dtype=ids.dtype)
            id_test = ids[2:] if n >= 3 else np.array([], dtype=ids.dtype)
        else:
            id_train, id_rest = train_test_split(
                ids,
                train_size=train_ratio,
                stratify=df["label"].values,
                random_state=seed,
            )
            rest_df = df[df["doc_id"].isin(id_rest)].copy()
            val_ratio_rest = val_ratio / (val_ratio + test_ratio)
            n_rest = len(rest_df)
            if n_rest < 2:
                id_val = id_rest
                id_test = np.array([], dtype=id_rest.dtype)
            else:
                id_val, id_test = train_test_split(
                    rest_df["doc_id"].values,
                    train_size=val_ratio_rest,
                    stratify=rest_df["label"].values,
                    random_state=seed,
                )
        train_df = df[df["doc_id"].isin(id_train)].copy()
        val_df = df[df["doc_id"].isin(id_val)].copy()
        test_df = df[df["doc_id"].isin(id_test)].copy()

    assert_disjoint_splits(
        train_df["doc_id"].tolist(),
        val_df["doc_id"].tolist(),
        test_df["doc_id"].tolist(),
    )
    return train_df, val_df, test_df


def load_and_split(
    data_dir: str | Path,
    categories: list[str],
    label_map: dict[str, int],
    split_by_file: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    limit_docs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load docs and split. Returns (full_df, train_df, val_df, test_df).
    """
    df = load_docs_from_data_dir(
        data_dir, categories, label_map, limit_docs=limit_docs
    )
    train_df, val_df, test_df = split_docs(
        df, split_by_file, train_ratio, val_ratio, test_ratio, seed=seed
    )
    return df, train_df, val_df, test_df
