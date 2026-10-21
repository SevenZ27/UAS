"""Prespecified candidates; sensitivity models never enter winner selection."""

from dataclasses import dataclass

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import make_pipeline


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    eligible: bool = True
    scale: bool = False
    clipping: bool = True
    indicators: bool = False


def candidates():
    return [
        Candidate("prior_baseline", DummyClassifier(strategy="prior")),
        Candidate(
            "logistic_regression", LogisticRegression(C=1, max_iter=3000), scale=True
        ),
        Candidate(
            "decision_tree",
            DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=42),
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
        ),
        Candidate(
            "random_forest_regularized",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=2,
            ),
        ),
        Candidate(
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.1,
                max_leaf_nodes=15,
                l2_regularization=1,
                early_stopping=False,
                random_state=42,
            ),
        ),
        Candidate(
            "sensitivity_no_clipping",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
            eligible=False,
            clipping=False,
        ),
        Candidate(
            "sensitivity_missing_indicators",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
            eligible=False,
            indicators=True,
        ),
    ]


def candidate_pipeline(candidate):
    pipeline = make_pipeline()
    pipeline.set_params(classifier=clone(candidate.estimator))
    if not candidate.clipping:
        pipeline.set_params(clip="passthrough")
    if candidate.indicators:
        pipeline.set_params(impute__add_indicator=True)
    if candidate.scale:
        pipeline = Pipeline(
            [*pipeline.steps[:-1], ("scale", StandardScaler()), pipeline.steps[-1]]
        )
    return pipeline


def select_candidate(rows):
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("No eligible candidates")
    return min(
        eligible,
        key=lambda row: (-row["mean_macro_f1"], row["mean_log_loss"], row["name"]),
    )
