"""Build the stage-one integration model; holdout scoring is reserved for stage two."""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .contract import (
    CLASSES,
    CONTRACT_VERSION,
    FEATURE_NAMES,
    FEATURES,
    contract_document,
    validate_instances,
)
from .preprocessing import FeatureCleaner, make_pipeline


def build_bundle(data_path, output_dir):
    data_path, output_dir = Path(data_path), Path(output_dir)
    raw = pd.read_csv(data_path)
    required = {*FEATURE_NAMES, "Credit_Score", "Customer_ID", "Month"}
    if not required <= set(raw.columns):
        raise ValueError("Training source is missing required columns.")
    if (
        raw.Customer_ID.isna().any()
        or set(raw.Credit_Score.dropna()) != set(CLASSES)
        or raw.Credit_Score.isna().any()
    ):
        raise ValueError("Invalid groups or target classes in training source.")
    if raw.duplicated(["Customer_ID", "Month"]).any():
        raise ValueError(
            "Duplicate customer-month records require a documented resolution."
        )
    train_index, holdout_index = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            raw, groups=raw.Customer_ID
        )
    )
    train_groups = set(raw.iloc[train_index].Customer_ID)
    holdout_groups = set(raw.iloc[holdout_index].Customer_ID)
    if train_groups & holdout_groups:
        raise AssertionError("Customer groups overlap.")
    train = raw.iloc[train_index].loc[:, list(FEATURE_NAMES)]
    cleaned = FeatureCleaner().fit_transform(train)
    if cleaned.isna().all().any():
        raise ValueError("A training feature has no valid values.")
    pipeline = make_pipeline()
    pipeline.fit(train, raw.iloc[train_index].Credit_Score)
    transformed = pipeline[:-1].transform(train)
    ranges, defaults = {}, {}
    for i, feature in enumerate(FEATURES):
        lo, hi = float(np.min(transformed[:, i])), float(np.max(transformed[:, i]))
        if feature.name == "Payment_of_Min_Amount":
            lo, hi = 0.0, 1.0
        if feature.integer:
            lo, hi = float(np.ceil(lo)), float(np.floor(hi))
        default = float(np.clip(np.median(transformed[:, i]), lo, hi))
        if feature.integer:
            default = round(default)
        ranges[feature.name] = {"min": lo, "max": hi}
        defaults[feature.name] = default
    validate_instances([defaults])
    source_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("preprocessing.py"),
        Path(__file__).with_name("contract.py"),
    ]
    code_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths
    }
    model_version = (
        "stage1-"
        + hashlib.sha256(
            (source_sha + json.dumps(code_hashes, sort_keys=True)).encode()
        ).hexdigest()[:12]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, output_dir / "pipeline.joblib", compress=3)
    np.savez_compressed(
        output_dir / "split-indices.npz", train=train_index, holdout=holdout_index
    )
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "model_version": model_version,
        "stage": "integration build; no final evaluation or tuning",
        "built_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_sha,
        "model_sha256": hashlib.sha256(
            (output_dir / "pipeline.joblib").read_bytes()
        ).hexdigest(),
        "split_indices_sha256": hashlib.sha256(
            (output_dir / "split-indices.npz").read_bytes()
        ).hexdigest(),
        "source_code_sha256": code_hashes,
        "features": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ["scikit-learn", "numpy", "pandas", "joblib"]
        },
        "split": {
            "method": "GroupShuffleSplit",
            "seed": 42,
            "train_rows": len(train_index),
            "holdout_rows": len(holdout_index),
            "shared_customers": 0,
            "train_customers": len(train_groups),
            "holdout_customers": len(holdout_groups),
        },
        "training_missing_after_parsing": {
            key: int(value) for key, value in cleaned.isna().sum().items()
        },
        "supported_ranges": ranges,
        "defaults": defaults,
        "imputation_medians": dict(
            zip(FEATURE_NAMES, pipeline.named_steps["impute"].statistics_.tolist())
        ),
        "money_unit": "Source dataset units; currency not established",
        "limitations": [
            "Integration model; performance evaluation pending stage two.",
            "Dataset provenance, currency and target construction remain unverified.",
            "Supported ranges are training-derived demo limits, not lending eligibility rules.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    docs = Path(__file__).resolve().parent.parent / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "data-contract.json").write_text(
        json.dumps(contract_document(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the unchanged archived data_C.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    metadata = build_bundle(args.data, args.output)
    print(
        json.dumps(
            {
                "model_version": metadata["model_version"],
                "split": metadata["split"],
                "model_sha256": metadata["model_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
