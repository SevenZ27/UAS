import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from credit_score.contract import FEATURE_NAMES
from credit_score.preprocessing import (
    FeatureCleaner,
    numeric,
    parse_history,
    parse_payment,
)


@pytest.mark.parametrize(
    "value,expected", [("2 Years and 3 Months", 27), (27, 27), (27.0, 27), ("27", 27)]
)
def test_history_accepts_source_and_canonical_months(value, expected):
    assert parse_history(value) == expected


@pytest.mark.parametrize("value,expected", [("Yes", 1), ("No", 0), (1, 1), (0.0, 0)])
def test_binary_accepts_source_and_canonical_values(value, expected):
    assert parse_payment(value) == expected


@pytest.mark.parametrize(
    "value", ["_", "abc123", "12-34", "1e9999", "__import__('os')", None]
)
def test_malformed_source_does_not_become_a_number(value):
    assert np.isnan(numeric(value))


def test_cleaning_twice_preserves_history_payment_and_signed_fields(profile):
    profile.update(
        Credit_History_Age="2 Years and 3 Months",
        Payment_of_Min_Amount="Yes",
        Changed_Credit_Limit=-4.5,
        Delay_from_due_date=-2,
        Num_of_Delayed_Payment="-3",
    )
    cleaner = FeatureCleaner()
    first = cleaner.fit_transform(pd.DataFrame([profile]))
    second = cleaner.transform(first)
    assert_frame_equal(first, second)
    assert first.Credit_History_Age.iloc[0] == 27
    assert first.Payment_of_Min_Amount.iloc[0] == 1
    assert first.Changed_Credit_Limit.iloc[0] == -4.5
    assert first.Delay_from_due_date.iloc[0] == -2
    assert np.isnan(first.Num_of_Delayed_Payment.iloc[0])


def test_imputer_and_outlier_statistics_are_frozen_after_fit(service, profile):
    medians = service.pipeline.named_steps["impute"].statistics_.copy()
    fences = service.pipeline.named_steps["clip"].upper_.copy()
    profile.update(Annual_Income=1e12, Credit_History_Age=None)
    transformed = service.pipeline[:-1].transform(pd.DataFrame([profile]))
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(
        medians, service.pipeline.named_steps["impute"].statistics_
    )
    np.testing.assert_array_equal(fences, service.pipeline.named_steps["clip"].upper_)


def test_synthetic_train_only_median_not_affected_by_test(profile):
    from sklearn.impute import SimpleImputer

    train = pd.DataFrame(
        [{**profile, "Monthly_Inhand_Salary": x} for x in [100.0, 200.0, None]]
    )
    cleaner = FeatureCleaner()
    cleaned = cleaner.fit_transform(train)
    imputer = SimpleImputer(strategy="median").fit(cleaned)
    test = cleaner.transform(pd.DataFrame([{**profile, "Monthly_Inhand_Salary": 1e9}]))
    imputer.transform(test)
    assert imputer.statistics_[FEATURE_NAMES.index("Monthly_Inhand_Salary")] == 150.0


def test_wrong_feature_set_fails_instead_of_zero_filling(profile):
    del profile["Age"]
    with pytest.raises(ValueError):
        FeatureCleaner().transform(pd.DataFrame([profile]))
