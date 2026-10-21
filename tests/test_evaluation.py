import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from credit_score.contract import CLASSES
from credit_score.evaluation_metrics import (
    classification_metrics,
    clustered_intervals,
    fit_temperature,
    reliability,
    temperature_scale,
)
from credit_score.evaluation_models import (
    candidate_pipeline,
    candidates,
    select_candidate,
)
from credit_score.evaluation_splits import (
    development_folds,
    validate_partition,
)


def test_scores_match_independent_sklearn_implementation():
    y = np.array(["Good", "Poor", "Standard", "Good", "Poor", "Standard"])
    p = np.array(["Good", "Standard", "Standard", "Poor", "Poor", "Standard"])
    probabilities = np.full((6, 3), 0.1)
    for i, label in enumerate(p):
        probabilities[i, CLASSES.index(label)] = 0.8
    result = classification_metrics(y, p, probabilities)
    assert result["accuracy"] == pytest.approx(accuracy_score(y, p))
    assert result["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert result["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, p))
    assert result["log_loss"] == pytest.approx(
        log_loss(y, probabilities, labels=list(CLASSES))
    )
    assert result["brier_multiclass"] == pytest.approx(
        np.mean(
            np.square(probabilities - np.eye(3)[[CLASSES.index(x) for x in y]]).sum(
                axis=1
            )
        )
    )


@pytest.mark.parametrize(
    "train,test", [([0, 1], [1, 2]), ([0, 0], [2]), ([0], [4]), ([], [1]), ([0.5], [2])]
)
def test_bad_row_partitions_are_rejected(train, test):
    with pytest.raises(ValueError):
        validate_partition(np.array(["a", "b", "c", "d"]), train, test)


def test_repeated_customer_rows_cannot_cross_partitions():
    with pytest.raises(ValueError, match="Customer leakage"):
        validate_partition(np.array(["a", "a", "b", "b"]), [0, 2], [1, 3])
    validate_partition(np.array(["a", "a", "b", "b"]), [0, 1], [2, 3])


def test_group_folds_cover_every_row_once():
    y = np.tile(CLASSES, 30)
    groups = np.repeat(np.arange(30), 3)
    folds = development_folds(y, groups)
    counts = np.zeros(len(y))
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
        counts[val] += 1
    assert np.all(counts == 1)


def test_sensitivity_model_cannot_win_selection():
    rows = [
        {
            "name": "sensitivity",
            "eligible": False,
            "mean_macro_f1": 1.0,
            "mean_log_loss": 0.0,
        },
        {"name": "a", "eligible": True, "mean_macro_f1": 0.5, "mean_log_loss": 1.0},
        {"name": "b", "eligible": True, "mean_macro_f1": 0.5, "mean_log_loss": 0.8},
    ]
    assert select_candidate(rows)["name"] == "b"


def test_fold_statistics_do_not_see_validation_extremes(profile):
    spec = next(x for x in candidates() if x.name == "random_forest")
    pipeline = candidate_pipeline(spec)
    train = pd.DataFrame(
        [
            {**profile, "Annual_Income": income}
            for income in [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        ]
    )
    pipeline.fit(train, np.tile(CLASSES, 2))
    statistics = pipeline.named_steps["impute"].statistics_.copy()
    assert statistics[1] == 350.0
    pipeline.predict(pd.DataFrame([{**profile, "Annual_Income": 1e12}]))
    np.testing.assert_array_equal(
        statistics, pipeline.named_steps["impute"].statistics_
    )
    assert not hasattr(spec.estimator, "classes_")


def test_cluster_intervals_retain_all_months_of_customer():
    y = np.array(["Good", "Poor", "Standard"] * 4)
    groups = np.repeat(np.arange(4), 3)
    perfect = y.copy()
    baseline = np.repeat("Standard", 12)
    result = clustered_intervals(y, perfect, baseline, groups, replicates=50, seed=7)
    assert result["customers"] == 4
    assert result["intervals"]["accuracy"] == [1.0, 1.0]
    assert result["intervals"]["macro_f1"] == [1.0, 1.0]
    assert result["paired_difference_vs_prior"]["accuracy"] == pytest.approx(
        [2 / 3, 2 / 3]
    )
    assert result == clustered_intervals(
        y, perfect, baseline, groups, replicates=50, seed=7
    )


def test_customer_balanced_weighting_differs_from_row_weighting():
    y = ["Good"] * 9 + ["Poor"]
    p = ["Good"] * 10
    row = classification_metrics(y, p)["accuracy"]
    customer = classification_metrics(y, p, weights=[1 / 9] * 9 + [1.0])["accuracy"]
    assert row == 0.9
    assert customer == pytest.approx(0.5)


def test_temperature_preserves_argmax_and_probability_sum():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    calibrated = temperature_scale(p, 1.7)
    np.testing.assert_array_equal(p.argmax(axis=1), calibrated.argmax(axis=1))
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    temperature = fit_temperature(np.array(CLASSES), p)
    assert np.exp(-2) <= temperature <= np.exp(2)


def test_reliability_bins_include_confidence_one_and_empty_bins():
    result = reliability(np.array(CLASSES), np.eye(3))
    assert result["ece_top_label"] == 0
    assert result["bins"][-1]["count"] == 3
    assert result["bins"][0]["confidence"] is None


def test_final_holdout_report_cannot_be_overwritten(tmp_path):
    from credit_score.evaluate import run

    output = tmp_path / "public"
    output.mkdir()
    (output / "holdout-results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already scored"):
        run(
            tmp_path / "absent.csv",
            tmp_path / "absent-artifacts",
            output,
            tmp_path / "private",
            "final",
        )
    assert (output / "holdout-results.json").read_text() == "{}"


def test_changed_source_data_is_rejected_before_split_loading(tmp_path):
    from credit_score.evaluation_splits import load_data_and_splits

    source = tmp_path / "source.csv"
    source.write_text("not,the,reserved,data")
    with pytest.raises(ValueError, match="Dataset differs"):
        load_data_and_splits(source, tmp_path, {"source_sha256": "incorrect"})


def test_changed_split_file_is_rejected(tmp_path):
    import hashlib

    from credit_score.evaluation_splits import load_data_and_splits

    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n")
    (tmp_path / "split-indices.npz").write_bytes(b"changed")
    metadata = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split_indices_sha256": "incorrect",
    }
    with pytest.raises(ValueError, match="split indices changed"):
        load_data_and_splits(source, tmp_path, metadata)
