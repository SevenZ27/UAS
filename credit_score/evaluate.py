"""Locked development comparison followed by one untouched customer holdout."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .contract import (
    CLASSES,
    FEATURE_NAMES,
    FEATURES,
    validate_instances,
)
from .evaluation_metrics import (
    classification_metrics,
    clustered_intervals,
    confusion,
    fit_temperature,
    reliability,
    temperature_scale,
)
from .evaluation_models import candidate_pipeline, candidates, select_candidate
from .evaluation_splits import development_folds, load_data_and_splits
from .preprocessing import FeatureCleaner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf8",
    )


def numeric_scores(result):
    return {key: value for key, value in result.items() if key != "per_class"}


def aligned_probabilities(model, X):
    probabilities = model.predict_proba(X)
    order = [list(model.classes_).index(label) for label in CLASSES]
    return probabilities[:, order]


def support_profile(pipeline, X):
    parsed = pipeline.named_steps["clean"].transform(X)
    imputed = pipeline.named_steps["impute"].transform(parsed)
    transformed = pipeline.named_steps["clip"].transform(imputed)
    ranges, defaults = {}, {}
    for i, feature in enumerate(FEATURES):
        lo, hi = float(transformed[:, i].min()), float(transformed[:, i].max())
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
    return ranges, defaults


def serving_coverage(X, ranges):
    parsed = FeatureCleaner().transform(X)
    valid = np.ones(len(parsed), dtype=bool)
    range_failure = np.zeros(len(parsed), dtype=bool)
    for feature in FEATURES:
        values = parsed[feature.name].to_numpy()
        if not feature.nullable:
            valid &= np.isfinite(values)
        outside = np.isfinite(values) & (
            (values < ranges[feature.name]["min"])
            | (values > ranges[feature.name]["max"])
        )
        range_failure |= outside
        valid &= ~outside
    return valid, {
        "accepted": int(valid.sum()),
        "total": len(valid),
        "coverage": float(valid.mean()),
        "rows_outside_training_ranges": int(range_failure.sum()),
        "missing_required_age_after_parsing": int(parsed.Age.isna().sum()),
    }


def slice_metrics(X, y, pred):
    parsed = FeatureCleaner().transform(X)
    ages = parsed.Age.to_numpy()
    missing = parsed.isna().sum(axis=1).to_numpy()
    masks = {
        "age_under_25": ages < 25,
        "age_25_to_44": (ages >= 25) & (ages < 45),
        "age_45_plus": ages >= 45,
        "age_missing": np.isnan(ages),
        "missing_zero": missing == 0,
        "missing_1_to_2": (missing >= 1) & (missing <= 2),
        "missing_3_plus": missing >= 3,
    }
    return {
        key: {
            "rows": int(mask.sum()),
            "metrics": classification_metrics(y[mask], pred[mask])
            if mask.any()
            else None,
        }
        for key, mask in masks.items()
    }


def build_evaluated_bundle(
    pipeline, metadata, selection, report_path, X, model_path, output
):
    output.mkdir(parents=True, exist_ok=True)
    ranges, defaults = support_profile(pipeline, X)
    shutil.copy2(model_path, output / "pipeline.joblib")
    updated = dict(metadata)
    updated.update(
        model_version="stage2-" + sha(selection)[:12],
        stage="customer-grouped evaluated prototype",
        built_at=datetime.now(UTC).isoformat(),
        model_sha256=sha(model_path),
        supported_ranges=ranges,
        defaults=defaults,
        selected_candidate=json.loads(selection.read_text())["selected_candidate"],
        evaluation_report_sha256=sha(report_path),
        evaluation_report="docs/evaluation/stage-two/holdout-results.json",
        imputation_medians=dict(
            zip(FEATURE_NAMES, pipeline.named_steps["impute"].statistics_.tolist())
        ),
        limitations=[
            "Local grouped benchmark of dataset categories, not real lending validation.",
            "Provenance, currency, label construction and observation year remain unverified.",
            "Customer cluster intervals condition on the fixed model and benchmark sample.",
            "Strict app-domain coverage is lower than whole-dataset benchmark coverage.",
        ],
    )
    write_json(output / "metadata.json", updated)
    return updated


def run(data_path, artifact_dir, output, private, phase):
    output, private = Path(output), Path(private)
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    if (output / "holdout-results.json").exists():
        raise RuntimeError(
            "Final holdout was already scored; existing results are immutable. Use verification, not retuning."
        )
    metadata = json.loads((Path(artifact_dir) / "metadata.json").read_text())
    raw, development, holdout = load_data_and_splits(data_path, artifact_dir, metadata)
    X = raw.iloc[development].loc[:, list(FEATURE_NAMES)]
    y = raw.iloc[development].Credit_Score.to_numpy()
    groups = raw.iloc[development].Customer_ID.to_numpy()
    folds = development_folds(y, groups)
    root = Path(__file__).resolve().parents[1]
    code_files = [
        *sorted((root / "credit_score").glob("evaluation*.py")),
        Path(__file__),
        root / "credit_score/preprocessing.py",
        root / "credit_score/contract.py",
    ]
    source_hashes = {str(file.relative_to(root)): sha(file) for file in code_files}
    protocol = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_sha256": metadata["source_sha256"],
        "split_indices_sha256": metadata["split_indices_sha256"],
        "protocol_sha256": sha(root / "docs/EVALUATION-PROTOCOL.md"),
        "source_code_sha256": source_hashes,
        "primary_metric": "mean development fold macro-F1",
        "fold_seed": 42,
        "candidate_specs": [
            {
                **{key: value for key, value in vars(c).items() if key != "estimator"},
                "estimator_class": type(c.estimator).__name__,
                "estimator_params": c.estimator.get_params(),
            }
            for c in candidates()
        ],
        "runtime": {
            key: importlib.metadata.version(key)
            for key in ["scikit-learn", "numpy", "pandas", "joblib", "scipy"]
        },
    }
    protocol_path = output / "run-protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        for key in [
            "source_sha256",
            "split_indices_sha256",
            "protocol_sha256",
            "source_code_sha256",
        ]:
            if previous[key] != protocol[key]:
                raise RuntimeError("Frozen run inputs changed: " + key)
    else:
        write_json(protocol_path, protocol)
    split_report = {
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "development_customers": len(set(groups)),
        "holdout_customers": int(raw.iloc[holdout].Customer_ID.nunique()),
        "shared_customers": 0,
        "folds": [],
    }
    for i, (train, val) in enumerate(folds):
        split_report["folds"].append(
            {
                "fold": i,
                "train_rows": len(train),
                "validation_rows": len(val),
                "train_customers": len(set(groups[train])),
                "validation_customers": len(set(groups[val])),
                "shared_customers": 0,
                "train_class_counts": pd.Series(y[train]).value_counts().to_dict(),
                "validation_class_counts": pd.Series(y[val]).value_counts().to_dict(),
                "train_indices_sha256": hashlib.sha256(
                    development[train].tobytes()
                ).hexdigest(),
                "validation_indices_sha256": hashlib.sha256(
                    development[val].tobytes()
                ).hexdigest(),
            }
        )
    write_json(output / "split-audit.json", split_report)
    specs = candidates()
    rows = []
    summaries = []
    selection_path = output / "selection.json"
    if phase in ["all", "cv"]:
        if selection_path.exists():
            raise RuntimeError("Development selection is already frozen.")
        for candidate in specs:
            oof = np.zeros((len(X), 3))
            fold_rows = []
            for fold, (train, val) in enumerate(folds):
                started = time.perf_counter()
                model = candidate_pipeline(candidate)
                with (
                    warnings.catch_warnings(record=True) as caught,
                    threadpool_limits(limits=2),
                ):
                    warnings.simplefilter("always")
                    model.fit(X.iloc[train], y[train])
                    probabilities = aligned_probabilities(model, X.iloc[val])
                    prediction = model.predict(X.iloc[val])
                metrics = classification_metrics(y[val], prediction, probabilities)
                row = {
                    "name": candidate.name,
                    "eligible": candidate.eligible,
                    "fold": fold,
                    **numeric_scores(metrics),
                    "seconds": time.perf_counter() - started,
                    "warnings": [str(w.message) for w in caught],
                }
                rows.append(row)
                fold_rows.append(row)
                oof[val] = probabilities
                pd.DataFrame(
                    [
                        {k: v for k, v in item.items() if k != "warnings"}
                        for item in rows
                    ]
                ).to_csv(output / "cv-folds.csv", index=False)
                print(
                    f"{candidate.name} fold {fold + 1}/5 macro-F1={metrics['macro_f1']:.4f}",
                    flush=True,
                )
            pooled = classification_metrics(
                y, np.asarray(CLASSES)[oof.argmax(axis=1)], oof
            )
            summary = {
                "name": candidate.name,
                "eligible": candidate.eligible,
                "pooled_oof": pooled,
            }
            for metric in [
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "weighted_f1",
                "log_loss",
                "brier_multiclass",
            ]:
                values = [row[metric] for row in fold_rows]
                summary["mean_" + metric] = float(np.mean(values))
                summary["fold_sd_" + metric] = float(np.std(values, ddof=1))
            summaries.append(summary)
            np.savez_compressed(
                private / (candidate.name + "-oof.npz"), probabilities=oof
            )
        write_json(output / "cv-results.json", {"candidates": summaries, "folds": rows})
        chosen = select_candidate(summaries)
        probabilities = np.load(
            private / (chosen["name"] + "-oof.npz"), allow_pickle=False
        )["probabilities"]
        temperature = fit_temperature(y, probabilities)
        selection = {
            "frozen_at": datetime.now(UTC).isoformat(),
            "selected_candidate": chosen["name"],
            "selection_rule": "Highest mean fold macro-F1; lower mean log loss then name for ties",
            "selected_parameters": next(
                c for c in specs if c.name == chosen["name"]
            ).estimator.get_params(),
            "development_mean_macro_f1": chosen["mean_macro_f1"],
            "temperature": temperature,
            "oof_uncalibrated": classification_metrics(
                y, np.asarray(CLASSES)[probabilities.argmax(axis=1)], probabilities
            ),
            "oof_calibrated": classification_metrics(
                y,
                np.asarray(CLASSES)[probabilities.argmax(axis=1)],
                temperature_scale(probabilities, temperature),
            ),
            "run_protocol_sha256": sha(protocol_path),
            "holdout_used_for_selection": False,
        }
        write_json(selection_path, selection)
        print("FROZEN SELECTION: " + chosen["name"], flush=True)
        if phase == "cv":
            return
    if not selection_path.exists():
        raise RuntimeError("Run development comparison before final scoring.")
    selection = json.loads(selection_path.read_text())
    if selection["run_protocol_sha256"] != sha(protocol_path):
        raise RuntimeError("Protocol changed after selection")
    spec = next(c for c in specs if c.name == selection["selected_candidate"])
    selected = candidate_pipeline(spec)
    baseline = candidate_pipeline(specs[0])
    with threadpool_limits(limits=2):
        selected.fit(X, y)
        baseline.fit(X, y)
    model_path = private / "selected-pipeline.joblib"
    joblib.dump(selected, model_path, compress=3)
    # First scoring access to holdout outcomes occurs only after selection is frozen.
    Xh = raw.iloc[holdout].loc[:, list(FEATURE_NAMES)]
    yh = raw.iloc[holdout].Credit_Score.to_numpy()
    gh = raw.iloc[holdout].Customer_ID.to_numpy()
    with threadpool_limits(limits=2):
        predicted = selected.predict(Xh)
        probabilities = aligned_probabilities(selected, Xh)
        reference = baseline.predict(Xh)
        reference_prob = aligned_probabilities(baseline, Xh)
    calibrated = temperature_scale(probabilities, selection["temperature"])
    _, inverse, counts = np.unique(gh, return_inverse=True, return_counts=True)
    customer_weights = 1 / counts[inverse]
    intervals = clustered_intervals(yh, predicted, reference, gh)
    ranges, _ = support_profile(selected, X)
    accepted, coverage = serving_coverage(Xh, ranges)
    report = {
        "scored_at": datetime.now(UTC).isoformat(),
        "selection_sha256": sha(selection_path),
        "selected_candidate": spec.name,
        "model_version": "stage2-" + sha(selection_path)[:12],
        "model_sha256": sha(model_path),
        "holdout_rows": len(yh),
        "holdout_customers": len(set(gh)),
        "classes": list(CLASSES),
        "selected": classification_metrics(yh, predicted, probabilities),
        "prior_baseline": classification_metrics(yh, reference, reference_prob),
        "selected_customer_balanced": classification_metrics(
            yh, predicted, weights=customer_weights
        ),
        "confusion_matrix": confusion(yh, predicted).astype(int).tolist(),
        "bootstrap": intervals,
        "temperature": selection["temperature"],
        "calibrated_probability_metrics": classification_metrics(
            yh, predicted, calibrated
        ),
        "reliability_raw": reliability(yh, probabilities),
        "reliability_temperature": reliability(yh, calibrated),
        "serving_domain": {
            **coverage,
            "accepted_only_metrics": classification_metrics(
                yh[accepted], predicted[accepted]
            )
            if accepted.any()
            else None,
        },
        "exploratory_slices": slice_metrics(Xh, yh, predicted),
        "limits": [
            "Fixed dataset categories, not default or loan approval outcomes.",
            "Customer-held-out sample is not temporal or external validation.",
            "Cluster intervals exclude model-fitting, selection and population-shift uncertainty.",
            "OOF temperature fit is post-selection; calibration is diagnostic and not displayed in the serving API.",
            "Full raw-record benchmark includes profiles outside strict app-domain coverage.",
        ],
    }
    write_json(output / "holdout-results.json", report)
    pd.DataFrame(report["selected"]["per_class"]).T.to_csv(
        output / "holdout-per-class.csv", index_label="class"
    )
    pd.DataFrame(report["confusion_matrix"], index=CLASSES, columns=CLASSES).to_csv(
        output / "confusion-matrix.csv", index_label="actual"
    )
    np.savez_compressed(
        private / "holdout-predictions.npz",
        row_indices=holdout,
        y_true=yh.astype(str),
        y_pred=predicted.astype(str),
        baseline=reference.astype(str),
        probabilities=probabilities,
        calibrated=calibrated,
        customer_codes=inverse,
        accepted=accepted,
    )
    bundle = Path(artifact_dir) / "stage-two"
    build_evaluated_bundle(
        selected,
        metadata,
        selection_path,
        output / "holdout-results.json",
        X,
        model_path,
        bundle,
    )
    print(
        json.dumps(
            {
                "selected": spec.name,
                "holdout": numeric_scores(report["selected"]),
                "baseline": numeric_scores(report["prior_baseline"]),
                "coverage": coverage,
                "bootstrap": intervals,
                "bundle": str(bundle),
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/evaluation/stage-two")
    )
    parser.add_argument("--private", type=Path, default=Path("qa/evaluation/stage-two"))
    parser.add_argument("--phase", choices=["all", "cv", "final"], default="all")
    args = parser.parse_args()
    run(args.data, args.artifacts, args.output, args.private, args.phase)


if __name__ == "__main__":
    main()
