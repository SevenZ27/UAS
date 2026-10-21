"""Local and SageMaker call the same validated service and complete model bundle."""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import joblib
import pandas as pd

from .contract import (
    CLASSES,
    CONTRACT_VERSION,
    FEATURE_NAMES,
    FEATURES,
    ContractError,
    PredictionError,
    validate_instances,
    validate_request,
    validate_response,
)


class CreditScoreService:
    def __init__(self, pipeline, metadata):
        self.pipeline, self.metadata = pipeline, metadata
        if metadata.get("contract_version") != CONTRACT_VERSION or metadata.get(
            "features"
        ) != list(FEATURE_NAMES):
            raise PredictionError("Bundle model tidak sesuai kontrak fitur aplikasi.")
        if sorted(pipeline.classes_.tolist()) != sorted(CLASSES):
            raise PredictionError("Bundle model memiliki kategori yang tidak didukung.")
        try:
            if (
                not isinstance(metadata["model_version"], str)
                or not metadata["model_version"].strip()
            ):
                raise ValueError("Missing version")
            if set(metadata["supported_ranges"]) != set(FEATURE_NAMES):
                raise ValueError("Missing feature bounds")
            for feature in FEATURES:
                bounds = metadata["supported_ranges"][feature.name]
                if not all(
                    isinstance(bounds[k], (int, float)) and math.isfinite(bounds[k])
                    for k in ["min", "max"]
                ):
                    raise ValueError("Invalid bounds")
                if bounds["min"] > bounds["max"]:
                    raise ValueError("Reversed bounds")
            validate_instances([metadata["defaults"]])
            for feature in FEATURES:
                default = metadata["defaults"][feature.name]
                bounds = metadata["supported_ranges"][feature.name]
                if default is None or not bounds["min"] <= default <= bounds["max"]:
                    raise ValueError("Default outside supported bounds")
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionError(
                "Metadata bundle model tidak lengkap atau tidak valid."
            ) from exc

    def predict(self, payload):
        records = validate_request(payload)
        issues = []
        for i, record in enumerate(records):
            for feature in FEATURES:
                value = record[feature.name]
                bounds = self.metadata["supported_ranges"][feature.name]
                if value is not None and not bounds["min"] <= value <= bounds["max"]:
                    issues.append(
                        {
                            "row": i,
                            "field": feature.name,
                            "message": f"Di luar rentang model {bounds['min']:g}–{bounds['max']:g} {feature.unit}.",
                        }
                    )
        if issues:
            raise ContractError(issues)
        try:
            labels = self.pipeline.predict(
                pd.DataFrame.from_records(records, columns=FEATURE_NAMES)
            ).tolist()
        except Exception as exc:
            raise PredictionError(
                "Prediksi gagal diproses. Coba lagi atau muat ulang model."
            ) from exc
        warnings = []
        if any(value is None for record in records for value in record.values()):
            warnings.append(
                "Nilai yang tidak diketahui diisi menggunakan median data training."
            )
        response = {
            "status": "ok",
            "contract_version": CONTRACT_VERSION,
            "model_version": self.metadata["model_version"],
            "labels": labels,
            "warnings": warnings,
        }
        return validate_response(response, len(records))


def load_service(folder):
    """Load only the locally built, hash-verified bundle from a trusted directory."""
    folder = Path(folder)
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf8"))
        model_file = folder / "pipeline.joblib"
        if metadata["runtime"]["scikit-learn"] != importlib.metadata.version(
            "scikit-learn"
        ):
            raise PredictionError(
                "Versi scikit-learn berbeda. Gunakan environment yang dikunci untuk bundle ini."
            )
        if (
            hashlib.sha256(model_file.read_bytes()).hexdigest()
            != metadata["model_sha256"]
        ):
            raise PredictionError(
                "Berkas model tidak cocok dengan manifest. Bangun ulang bundle."
            )
        # Hashes detect corruption, not malicious authorship: trust the bundle source.
        pipeline = joblib.load(model_file)
        return CreditScoreService(pipeline, metadata)
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(
            "Bundle model belum tersedia atau tidak kompatibel. Jalankan proses build model."
        ) from exc


def validate_client_result(result, expected_rows=1):
    if isinstance(result, dict) and result.get("status") == "error":
        if result.get("code") == "INVALID_INPUT" and isinstance(
            result.get("issues"), list
        ):
            raise ContractError(result["issues"])
        raise PredictionError("Layanan prediksi belum tersedia. Coba lagi nanti.")
    return validate_response(result, expected_rows)


def request_for(record):
    validate_instances([record])
    return {"contract_version": CONTRACT_VERSION, "instances": [record]}
