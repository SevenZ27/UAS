"""SageMaker Python hooks: raw named values enter the same complete pipeline."""

import json

from credit_score.contract import ContractError, PredictionError, validate_request
from credit_score.serving_policy import load_exploration_service


def model_fn(model_dir):
    return load_exploration_service(model_dir)


def input_fn(request_body, request_content_type):
    if request_content_type.split(";")[0].strip() != "application/json":
        raise ValueError("Only application/json is supported.")
    try:
        payload = json.loads(
            request_body,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid JSON body.") from exc
    validate_request(payload)
    return payload


def predict_fn(input_data, service):
    try:
        return service.predict(input_data)
    except ContractError as exc:
        return {"status": "error", "code": "INVALID_INPUT", "issues": exc.issues}
    except PredictionError:
        return {"status": "error", "code": "PREDICTION_UNAVAILABLE"}


def output_fn(prediction, accept):
    if accept.split(";")[0].strip() != "application/json":
        raise ValueError("Only application/json is supported.")
    return json.dumps(prediction, allow_nan=False), "application/json"
