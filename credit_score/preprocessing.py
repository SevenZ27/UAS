"""One serializable preprocessing pipeline, fitted only on training rows."""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .contract import FEATURE_NAMES, FEATURES


def numeric(value):
    if value is None or pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return np.nan
    text = str(value).strip().strip("_")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return np.nan
    number = float(text)
    return number if np.isfinite(number) else np.nan


def parse_history(value):
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*(\d+)\s+Years?\s+and\s+(\d+)\s+Months?\s*", value, re.IGNORECASE
        )
        if match:
            months = int(match[2])
            return int(match[1]) * 12 + months if months < 12 else np.nan
    return numeric(value)


def parse_payment(value):
    if isinstance(value, str) and value.strip().lower() in {"yes", "no"}:
        return {"yes": 1.0, "no": 0.0}[value.strip().lower()]
    return numeric(value)


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Stateless parsing is idempotent for canonical numeric inputs."""

    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = len(FEATURE_NAMES)
        self.feature_names_in_ = np.asarray(FEATURE_NAMES, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(FEATURE_NAMES):
            raise ValueError("Expected exactly the 18 named raw features.")
        result = pd.DataFrame(index=X.index)
        for feature in FEATURES:
            parser = (
                parse_history
                if feature.name == "Credit_History_Age"
                else parse_payment
                if feature.name == "Payment_of_Min_Amount"
                else numeric
            )
            values = X[feature.name].map(parser).astype(float)
            invalid = pd.Series(False, index=values.index)
            if feature.minimum is not None:
                invalid |= values < feature.minimum
            if feature.maximum is not None:
                invalid |= values > feature.maximum
            if feature.integer:
                invalid |= values.notna() & (values != np.floor(values))
            result[feature.name] = values.mask(invalid)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


class TrainIQRClipper(TransformerMixin, BaseEstimator):
    """Retain signed features; freeze IQR fences learned from the train matrix."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.quantile(values, [0.25, 0.75], axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - 1.5 * iqr, np.min(values, axis=0))
        self.upper_ = np.where(iqr > 0, q3 + 1.5 * iqr, np.max(values, axis=0))
        for i, feature in enumerate(FEATURES):
            if feature.minimum is not None:
                self.lower_[i] = max(self.lower_[i], feature.minimum)
            if feature.maximum is not None:
                self.upper_[i] = min(self.upper_[i], feature.maximum)
            if feature.integer:
                self.lower_[i] = np.ceil(self.lower_[i])
                self.upper_[i] = np.floor(self.upper_[i])
            if feature.name == "Payment_of_Min_Amount":
                self.lower_[i], self.upper_[i] = 0, 1
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def make_pipeline():
    return Pipeline(
        [
            ("clean", FeatureCleaner()),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("clip", TrainIQRClipper()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2),
            ),
        ]
    )
