import math

import pytest

from credit_score.contract import (
    ContractError,
    PredictionError,
    validate_instances,
    validate_request,
    validate_response,
)
from credit_score.service import request_for


@pytest.mark.parametrize(
    "value", [True, False, "10000", math.nan, math.inf, -math.inf, [], {}]
)
def test_rejects_ambiguous_or_nonfinite_numbers(profile, value):
    profile["Annual_Income"] = value
    with pytest.raises(ContractError):
        validate_instances([profile])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "fractional_count",
        "negative_count",
        "bad_binary",
        "high_ratio",
        "missing_age",
    ],
)
def test_invalid_profile_never_reaches_model(profile, mutation):
    if mutation == "missing":
        del profile["Annual_Income"]
    if mutation == "extra":
        profile["Customer_ID"] = "not-a-feature"
    if mutation == "fractional_count":
        profile["Num_of_Loan"] = 1.5
    if mutation == "negative_count":
        profile["Num_of_Delayed_Payment"] = -2
    if mutation == "bad_binary":
        profile["Payment_of_Min_Amount"] = 2
    if mutation == "high_ratio":
        profile["Credit_Utilization_Ratio"] = 101
    if mutation == "missing_age":
        profile["Age"] = None
    with pytest.raises(ContractError):
        validate_instances([profile])


def test_numeric_missing_is_explicit_and_feature_order_is_irrelevant(profile):
    profile["Monthly_Inhand_Salary"] = None
    reversed_profile = dict(reversed(list(profile.items())))
    assert validate_instances([reversed_profile])[0] == profile


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"instances": []},
        {"contract_version": "old", "instances": []},
        {"contract_version": "1.0", "instances": [[1, 2]]},
    ],
)
def test_invalid_request_envelope(payload):
    with pytest.raises(ContractError):
        validate_request(payload)


def test_batch_limits_and_extra_request_keys(profile):
    assert len(validate_instances([profile] * 100)) == 100
    for instances in [[], [profile] * 101]:
        with pytest.raises(ContractError):
            validate_instances(instances)
    payload = request_for(profile)
    payload["ignored"] = "forbidden"
    with pytest.raises(ContractError):
        validate_request(payload)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"labels": ["Good"]},
        {
            "status": "ok",
            "contract_version": "1.0",
            "model_version": "x",
            "warnings": [],
            "labels": [],
        },
        {
            "status": "ok",
            "contract_version": "1.0",
            "model_version": "x",
            "warnings": [],
            "labels": ["UNKNOWN"],
        },
        {
            "status": "ok",
            "contract_version": "1.0",
            "model_version": "x",
            "warnings": [],
            "labels": [None],
        },
        {
            "status": "ok",
            "contract_version": "1.0",
            "model_version": "",
            "warnings": [],
            "labels": ["Poor"],
        },
    ],
)
def test_bad_response_is_technical_error(response):
    with pytest.raises(PredictionError):
        validate_response(response, 1)
