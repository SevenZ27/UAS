import json

import pandas as pd
import pytest

import inference
from credit_score.contract import FEATURE_NAMES, ContractError
from credit_score.service import request_for
from credit_score.serving_policy import ExplorationService, load_exploration_service


def test_outside_observed_range_uses_original_pipeline_without_retraining(
    service, profile
):
    model = ExplorationService(service)
    row = {**profile, "Age": 70, "Annual_Income": 120000000}
    direct = service.pipeline.predict(
        pd.DataFrame([row], columns=FEATURE_NAMES)
    ).tolist()
    response = model.predict(request_for(row))
    assert response["labels"] == direct
    assert any("belum teruji" in w for w in response["warnings"])
    assert any(
        "Pendapatan tahunan" in w and "membatasi" in w for w in response["warnings"]
    )
    assert row["Annual_Income"] == 120000000


def test_in_range_prediction_is_identical_to_frozen_strict_service(service, profile):
    payload = request_for(profile)
    assert ExplorationService(service).predict(payload) == service.predict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Age", 121),
        ("Num_of_Loan", 1.5),
        ("Num_Bank_Accounts", -1),
        ("Annual_Income", float("inf")),
    ],
)
def test_warning_policy_does_not_relax_invalid_schema(service, profile, field, value):
    with pytest.raises(ContractError):
        ExplorationService(service).predict(
            {"contract_version": "1.0", "instances": [{**profile, field: value}]}
        )


def test_local_and_serialized_hooks_share_outside_range_warnings(service, profile):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    payload = request_for({**profile, "Age": 70, "Annual_Income": 120000000})
    expected = load_exploration_service(root / "artifacts").predict(payload)
    hooks = inference.model_fn(root / "artifacts")
    response, kind = inference.output_fn(
        inference.predict_fn(
            inference.input_fn(json.dumps(payload), "application/json"), hooks
        ),
        "application/json",
    )
    assert json.loads(response) == expected
    assert kind == "application/json"


def test_batch_warning_identifies_the_affected_profile(service, profile):
    response = ExplorationService(service).predict(
        {"contract_version": "1.0", "instances": [profile, {**profile, "Age": 70}]}
    )
    assert len(response["labels"]) == 2
    assert any("Profil 2" in w for w in response["warnings"])
    assert not any("Profil 1 ·" in w for w in response["warnings"])
