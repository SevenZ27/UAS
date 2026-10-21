"""Verify frozen reports against saved predictions and independent sklearn metrics."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)

from credit_score.contract import CLASSES, FEATURE_NAMES, ContractError
from credit_score.evaluation_splits import development_folds, load_data_and_splits
from credit_score.service import load_service


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "docs/evaluation/stage-two"
    protocol = json.loads((data / "run-protocol.json").read_text())
    selection = json.loads((data / "selection.json").read_text())
    report = json.loads((data / "holdout-results.json").read_text())
    assert report["selection_sha256"] == digest(data / "selection.json")
    assert selection["run_protocol_sha256"] == digest(data / "run-protocol.json")
    assert datetime.fromisoformat(selection["frozen_at"]) < datetime.fromisoformat(
        report["scored_at"]
    )
    for name, value in protocol["source_code_sha256"].items():
        assert digest(root / name.replace("\\", "/")) == value, name
    assert digest(root / "docs/EVALUATION-PROTOCOL.md") == protocol["protocol_sha256"]
    metadata = json.loads((root / "artifacts/metadata.json").read_text())
    raw, dev, holdout = load_data_and_splits(
        root.parent
        / "SOURCES/SEMESTER 4/Model Deployment/Final Project/MD/UAS/lokal/data_C.csv",
        root / "artifacts",
        metadata,
    )
    with np.load(
        root / "qa/evaluation/stage-two/holdout-predictions.npz", allow_pickle=False
    ) as saved:
        y = saved["y_true"]
        pred = saved["y_pred"]
        prob = saved["probabilities"]
        accepted = saved["accepted"]
        indices = saved["row_indices"]
        baseline = saved["baseline"]
    np.testing.assert_array_equal(indices, holdout)
    np.testing.assert_array_equal(y, raw.iloc[holdout].Credit_Score.to_numpy())
    checks = {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "log_loss": log_loss(y, prob, labels=list(CLASSES)),
    }
    for key, value in checks.items():
        assert np.isclose(value, report["selected"][key]), key
    np.testing.assert_array_equal(
        confusion_matrix(y, pred, labels=list(CLASSES)), report["confusion_matrix"]
    )
    assert np.isclose(accuracy_score(y, baseline), report["prior_baseline"]["accuracy"])
    table = pd.read_csv(data / "cv-folds.csv")
    eligible = (
        table[table.eligible]
        .groupby("name")
        .agg({"macro_f1": "mean", "log_loss": "mean"})
        .reset_index()
    )
    selected = eligible.sort_values(
        ["macro_f1", "log_loss", "name"], ascending=[False, True, True]
    ).iloc[0]["name"]
    assert selected == selection["selected_candidate"]
    folds = development_folds(
        raw.iloc[dev].Credit_Score.to_numpy(), raw.iloc[dev].Customer_ID.to_numpy()
    )
    split = json.loads((data / "split-audit.json").read_text())
    for i, (train, val) in enumerate(folds):
        assert (
            hashlib.sha256(dev[train].tobytes()).hexdigest()
            == split["folds"][i]["train_indices_sha256"]
        )
        assert (
            hashlib.sha256(dev[val].tobytes()).hexdigest()
            == split["folds"][i]["validation_indices_sha256"]
        )
    service = load_service(root / "artifacts/stage-two")
    assert (
        digest(root / "artifacts/stage-two/pipeline.joblib") == report["model_sha256"]
    )
    X = raw.iloc[dev].loc[:, list(FEATURE_NAMES)]
    parsed = service.pipeline.named_steps["clean"].transform(X)
    np.testing.assert_allclose(
        parsed.median().to_numpy(), service.pipeline.named_steps["impute"].statistics_
    )
    raw_holdout = raw.iloc[holdout].loc[:, list(FEATURE_NAMES)]
    canonical = (
        service.pipeline.named_steps["clean"].transform(raw_holdout).astype(object)
    )
    canonical = canonical.where(canonical.notna(), None)
    accepted_rows = np.flatnonzero(accepted)[:100]
    payload = {
        "contract_version": "1.0",
        "instances": canonical.iloc[accepted_rows].to_dict("records"),
    }
    np.testing.assert_array_equal(
        service.predict(payload)["labels"], pred[accepted_rows]
    )
    for row in np.flatnonzero(~accepted)[:10]:
        try:
            service.predict(
                {
                    "contract_version": "1.0",
                    "instances": [canonical.iloc[row].to_dict()],
                }
            )
        except ContractError:
            pass
        else:
            raise AssertionError(
                "Rejected benchmark row was accepted by serving contract"
            )
    result = {
        "independent_metrics_match": True,
        "selection_frozen_before_final_scoring": True,
        "folds_and_holdout_unchanged": True,
        "source_and_protocol_hashes_match": True,
        "train_only_medians_match": True,
        "accepted_holdout_service_parity_cases": len(accepted_rows),
        "rejected_profile_cases": 10,
        "baseline_match": True,
        "model_version": report["model_version"],
        "source_report_sha256": digest(data / "holdout-results.json"),
    }
    (root / "docs/verification/evaluation-stage-two.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
