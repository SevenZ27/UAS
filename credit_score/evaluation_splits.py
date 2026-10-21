"""Guard the independent grouping unit and the exact stage-one holdout."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .contract import CLASSES, FEATURE_NAMES


def validate_partition(groups, train, test):
    train, test = np.asarray(train), np.asarray(test)
    if train.ndim != 1 or test.ndim != 1 or len(train) == 0 or len(test) == 0:
        raise ValueError("Empty or malformed partition")
    all_indices = np.concatenate([train, test])
    if (
        not np.issubdtype(all_indices.dtype, np.integer)
        or all_indices.min() < 0
        or all_indices.max() >= len(groups)
    ):
        raise ValueError("Invalid row index")
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Repeated or overlapping row indices")
    if set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]):
        raise ValueError("Customer leakage across partition")


def load_data_and_splits(data_path, artifact_dir, metadata):
    data_path, artifact_dir = Path(data_path), Path(artifact_dir)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["source_sha256"]:
        raise ValueError("Dataset differs from the reserved stage-one split")
    split_file = artifact_dir / "split-indices.npz"
    if (
        hashlib.sha256(split_file.read_bytes()).hexdigest()
        != metadata["split_indices_sha256"]
    ):
        raise ValueError("Reserved split indices changed")
    raw = pd.read_csv(data_path)
    if not {*FEATURE_NAMES, "Credit_Score", "Customer_ID", "Month"} <= set(raw):
        raise ValueError("Missing source columns")
    if raw.Customer_ID.isna().any() or raw.Credit_Score.isna().any():
        raise ValueError("Missing group/target")
    if set(raw.Credit_Score) != set(CLASSES):
        raise ValueError("Unexpected target classes")
    if raw.duplicated(["Customer_ID", "Month"]).any():
        raise ValueError("Duplicate customer-month grain")
    with np.load(split_file, allow_pickle=False) as split:
        train, holdout = split["train"], split["holdout"]
    validate_partition(raw.Customer_ID.to_numpy(), train, holdout)
    if len(train) + len(holdout) != len(raw):
        raise ValueError("Partition is not exhaustive")
    return raw, train, holdout


def development_folds(y, groups):
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y, groups
        )
    )
    counts = np.zeros(len(y), dtype=int)
    for train, val in folds:
        validate_partition(groups, train, val)
        if set(np.asarray(y)[train]) != set(CLASSES) or set(np.asarray(y)[val]) != set(
            CLASSES
        ):
            raise ValueError("A fold is missing an evaluation class")
        counts[val] += 1
    if not np.all(counts == 1):
        raise ValueError("OOF rows are not covered exactly once")
    return folds
