import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ReadTimeoutError

import inference
from credit_score.client import invoke_sagemaker
from credit_score.contract import (
    CLASSES,
    FEATURE_NAMES,
    FEATURES,
    ContractError,
    PredictionError,
)
from credit_score.service import load_service, request_for


def test_local_serialized_sagemaker_and_batch_outputs_agree(service, profile):
    rng = np.random.default_rng(42)
    profiles = [profile]
    for _ in range(99):
        row = {}
        for f in FEATURES:
            bounds = service.metadata["supported_ranges"][f.name]
            row[f.name] = (
                int(rng.integers(int(bounds["min"]), int(bounds["max"]) + 1))
                if f.integer
                else float(rng.uniform(bounds["min"], bounds["max"]))
            )
        profiles.append(row)
    payload = {"contract_version": "1.0", "instances": profiles}
    expected = service.predict(payload)
    wire = inference.input_fn(json.dumps(payload), "application/json")
    returned, content_type = inference.output_fn(
        inference.predict_fn(wire, service), "application/json"
    )
    assert json.loads(returned) == expected
    assert content_type == "application/json"
    assert set(expected["labels"]) <= set(CLASSES)
    direct = service.pipeline.predict(
        pd.DataFrame(profiles, columns=FEATURE_NAMES)
    ).tolist()
    assert direct == expected["labels"]
    assert (
        service.predict(request_for(dict(reversed(list(profile.items())))))["labels"]
        == expected["labels"][:1]
    )


def test_missing_salary_is_imputed_and_disclosed(service, profile):
    profile["Monthly_Inhand_Salary"] = None
    result = service.predict(request_for(profile))
    assert result["labels"][0] in CLASSES
    assert result["warnings"]


def test_large_legacy_money_defaults_are_rejected(service, profile):
    profile["Annual_Income"] = 120000000
    with pytest.raises(ContractError) as caught:
        service.predict(request_for(profile))
    assert any(x["field"] == "Annual_Income" for x in caught.value.issues)


def test_sagemaker_schema_error_and_model_error_are_never_labels(
    service, profile, monkeypatch
):
    bad = {**profile, "Credit_Utilization_Ratio": 101}
    assert (
        inference.predict_fn({"contract_version": "1.0", "instances": [bad]}, service)[
            "code"
        ]
        == "INVALID_INPUT"
    )

    def broken(*args, **kwargs):
        raise RuntimeError("internal sensitive detail")

    monkeypatch.setattr(service.pipeline, "predict", broken)
    result = inference.predict_fn(request_for(profile), service)
    assert result == {"status": "error", "code": "PREDICTION_UNAVAILABLE"}
    assert "labels" not in result


@pytest.mark.parametrize(
    "body",
    [
        "{}",
        '{"contract_version":"1.0","instances":[]}',
        '{"contract_version":"1.0","instances":NaN}',
    ],
)
def test_sagemaker_rejects_invalid_json_contract(body):
    with pytest.raises((ValueError, ContractError)):
        inference.input_fn(body, "application/json")


def test_missing_and_corrupt_bundle_fail_closed(tmp_path):
    with pytest.raises(PredictionError):
        load_service(tmp_path)
    from pathlib import Path

    original = Path(__file__).resolve().parents[1] / "artifacts"
    (tmp_path / "metadata.json").write_bytes((original / "metadata.json").read_bytes())
    (tmp_path / "pipeline.joblib").write_bytes(b"broken")
    with pytest.raises(PredictionError, match="manifest"):
        load_service(tmp_path)


@pytest.mark.parametrize(
    "response",
    [
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
    ],
)
def test_controlled_cloud_responses_fail_closed(monkeypatch, profile, response):
    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "test-endpoint")

    class Runtime:
        def invoke_endpoint(self, **kwargs):
            return {"Body": BytesIO(json.dumps(response).encode())}

    with pytest.raises(PredictionError):
        invoke_sagemaker(request_for(profile), Runtime())


def test_cloud_timeout_and_missing_config_are_recoverable(monkeypatch, profile):
    monkeypatch.delenv("CREDIT_SCORE_ENDPOINT", raising=False)
    with pytest.raises(PredictionError, match="belum diatur"):
        invoke_sagemaker(request_for(profile))
    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "test-endpoint")

    class Runtime:
        def invoke_endpoint(self, **kwargs):
            raise ReadTimeoutError(endpoint_url="https://example.invalid")

    with pytest.raises(PredictionError, match="belum dapat dihubungi"):
        invoke_sagemaker(request_for(profile), Runtime())
