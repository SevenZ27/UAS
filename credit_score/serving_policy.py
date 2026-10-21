"""Exploration policy: keep the frozen model, disclose unfamiliar inputs and clipping."""

import math

import pandas as pd

from .contract import (
    FEATURE_NAMES,
    FEATURES,
    PredictionError,
    validate_request,
    validate_response,
)
from .service import load_service

RANGE_POLICY = "explore-with-range-warnings-v1"


def display_number(value):
    return (
        f"{value:,.6f}".rstrip("0")
        .rstrip(".")
        .replace(",", "_")
        .replace(".", ",")
        .replace("_", ".")
    )


def range_deviations(records, metadata):
    return [
        (i, feature)
        for i, record in enumerate(records)
        for feature in FEATURES
        if record[feature.name] is not None
        and not metadata["supported_ranges"][feature.name]["min"]
        <= record[feature.name]
        <= metadata["supported_ranges"][feature.name]["max"]
    ]


class ExplorationService:
    """Strict structural/domain rules remain; observed training ranges warn only."""

    def __init__(self, base):
        self.base = base
        self.pipeline = base.pipeline
        self.metadata = base.metadata

    def predict(self, payload):
        records = validate_request(payload)
        outside = range_deviations(records, self.metadata)
        if not outside:
            return self.base.predict(payload)
        frame = pd.DataFrame.from_records(records, columns=FEATURE_NAMES)
        try:
            # Use the original pipeline, not clipping to observed min/max values.
            labels = self.pipeline.predict(frame).tolist()
            transformed = self.pipeline[:-1].transform(frame)
        except Exception as exc:
            raise PredictionError(
                "Prediksi gagal diproses. Coba lagi atau muat ulang model."
            ) from exc
        warnings = [
            "Profil di luar rentang training. Prediksi tetap ditampilkan untuk eksplorasi; keandalannya untuk profil ini belum teruji."
        ]
        if any(value is None for record in records for value in record.values()):
            warnings.append(
                "Nilai yang tidak diketahui diisi menggunakan median data training."
            )
        for i, feature in outside:
            original = records[i][feature.name]
            used = float(transformed[i, FEATURE_NAMES.index(feature.name)])
            bounds = self.metadata["supported_ranges"][feature.name]
            prefix = f"Profil {i + 1} · " if len(records) > 1 else ""
            detail = f"{prefix}{feature.label}: {display_number(original)}; rentang contoh training {display_number(bounds['min'])}–{display_number(bounds['max'])}."
            if not math.isclose(float(original), used, rel_tol=1e-12, abs_tol=1e-12):
                detail += f" Pipeline membatasi nilai menjadi {display_number(used)} sebelum klasifikasi."
            else:
                detail += " Nilai ini tidak dibatasi oleh langkah IQR pipeline."
            warnings.append(detail)
        return validate_response(
            {
                "status": "ok",
                "contract_version": self.metadata["contract_version"],
                "model_version": self.metadata["model_version"],
                "labels": labels,
                "warnings": warnings,
            },
            len(records),
        )


def load_exploration_service(folder):
    return ExplorationService(load_service(folder))
