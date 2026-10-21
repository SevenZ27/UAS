"""Explicit classification estimands and customer-cluster uncertainty."""

import numpy as np
from scipy.optimize import minimize_scalar

from .contract import CLASSES


def encode(labels):
    lookup = {label: i for i, label in enumerate(CLASSES)}
    try:
        return np.asarray([lookup[x] for x in labels], dtype=int)
    except KeyError as exc:
        raise ValueError("Unknown evaluation class") from exc


def confusion(y_true, y_pred, weights=None):
    y, p = encode(y_true), encode(y_pred)
    if y.shape != p.shape:
        raise ValueError("Prediction lengths differ")
    matrix = np.zeros((3, 3), dtype=float)
    np.add.at(matrix, (y, p), 1.0 if weights is None else np.asarray(weights))
    return matrix


def scores_from_confusion(matrix):
    cm = np.asarray(matrix, dtype=float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    total = cm.sum()
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support > 0)
    precision = np.divide(
        cm.diagonal(), predicted, out=np.zeros(3), where=predicted > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(cm.trace() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "weighted_f1": float(f1 @ support / total) if total else 0.0,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": float(support[i]),
            }
            for i, label in enumerate(CLASSES)
        },
    }


def check_probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1)
    ):
        raise ValueError("Invalid class probabilities")
    return p


def probability_metrics(y_true, probabilities):
    p = check_probabilities(probabilities)
    y = encode(y_true)
    if len(y) != len(p):
        raise ValueError("Probability lengths differ")
    eps = np.finfo(float).eps
    safe = np.clip(p, eps, 1 - eps)
    safe /= safe.sum(axis=1, keepdims=True)
    return {
        "log_loss": float(-np.log(safe[np.arange(len(y)), y]).mean()),
        "brier_multiclass": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
    }


def classification_metrics(y_true, y_pred, probabilities=None, weights=None):
    result = scores_from_confusion(confusion(y_true, y_pred, weights))
    if probabilities is not None:
        result.update(probability_metrics(y_true, probabilities))
    return result


def temperature_scale(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    p = check_probabilities(probabilities)
    logits = np.log(np.clip(p, np.finfo(float).eps, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return result


def fit_temperature(y_true, probabilities):
    result = minimize_scalar(
        lambda log_t: probability_metrics(
            y_true, temperature_scale(probabilities, np.exp(log_t))
        )["log_loss"],
        bounds=(-2, 2),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError("Temperature fitting failed")
    return float(np.exp(result.x))


def reliability(y_true, probabilities, bins=10):
    p = check_probabilities(probabilities)
    y = encode(y_true)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    assignment = np.minimum((confidence * bins).astype(int), bins - 1)
    table = []
    ece = 0.0
    for b in range(bins):
        mask = assignment == b
        count = int(mask.sum())
        mean_conf = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(mean_conf - accuracy)
        table.append(
            {
                "lower": b / bins,
                "upper": (b + 1) / bins,
                "count": count,
                "confidence": mean_conf,
                "accuracy": accuracy,
            }
        )
    return {"ece_top_label": float(ece), "bins": table}


def grouped_confusions(y_true, y_pred, groups):
    _, indices = np.unique(groups, return_inverse=True)
    if len(indices) != len(y_true):
        raise ValueError("Group length mismatch")
    matrices = np.zeros((indices.max() + 1, 3, 3), dtype=float)
    np.add.at(matrices, (indices, encode(y_true), encode(y_pred)), 1.0)
    return matrices


def clustered_intervals(
    y_true, y_pred, baseline_pred, groups, replicates=2000, seed=20261021
):
    selected = grouped_confusions(y_true, y_pred, groups)
    baseline = grouped_confusions(y_true, baseline_pred, groups)
    n = len(selected)
    if n < 2:
        raise ValueError("At least two customers are needed for cluster uncertainty")
    rng = np.random.default_rng(seed)
    keys = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        *[f"recall_{x}" for x in CLASSES],
    ]
    samples = {key: [] for key in keys}
    differences = {key: [] for key in ["accuracy", "macro_f1"]}
    for _ in range(replicates):
        counts = rng.multinomial(n, np.full(n, 1 / n))
        scores = scores_from_confusion(np.einsum("i,ijk->jk", counts, selected))
        reference = scores_from_confusion(np.einsum("i,ijk->jk", counts, baseline))
        for key in keys:
            value = (
                scores["per_class"][key.removeprefix("recall_")]["recall"]
                if key.startswith("recall_")
                else scores[key]
            )
            samples[key].append(value)
        for key, values in differences.items():
            values.append(scores[key] - reference[key])
    interval = lambda values: [float(x) for x in np.quantile(values, [0.025, 0.975])]
    return {
        "method": "Customer cluster bootstrap, all observations retained; 95% percentile",
        "customers": n,
        "replicates": replicates,
        "seed": seed,
        "intervals": {key: interval(value) for key, value in samples.items()},
        "paired_difference_vs_prior": {
            key: interval(value) for key, value in differences.items()
        },
    }
