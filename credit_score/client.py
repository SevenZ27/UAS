"""Explicit backend selection; AWS failures never silently change prediction source."""

import json
import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .contract import PredictionError, validate_request
from .service import validate_client_result


def invoke_sagemaker(payload, runtime=None):
    records = validate_request(payload)
    endpoint = os.environ.get("CREDIT_SCORE_ENDPOINT", "").strip()
    if not endpoint:
        raise PredictionError(
            "Endpoint AWS belum diatur. Pilih mode Lokal atau atur endpoint di server."
        )
    try:
        if runtime is None:
            runtime = boto3.client(
                "sagemaker-runtime",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
                config=Config(
                    connect_timeout=3, read_timeout=15, retries={"max_attempts": 1}
                ),
            )
        response = runtime.invoke_endpoint(
            EndpointName=endpoint,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload, allow_nan=False),
        )
        result = json.loads(
            response["Body"].read(),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (
        BotoCoreError,
        ClientError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
    ) as exc:
        raise PredictionError(
            "Layanan AWS belum dapat dihubungi atau responsnya tidak valid. Coba lagi nanti."
        ) from exc
    return validate_client_result(result, len(records))
