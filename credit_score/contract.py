"""Versioned, named-feature request/response contract. No positional guessing."""

import math
from dataclasses import asdict, dataclass
from numbers import Real

CONTRACT_VERSION = "1.0"
CLASSES = ("Good", "Poor", "Standard")
MAX_BATCH = 100


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    unit: str
    minimum: float | None = 0
    maximum: float | None = None
    integer: bool = False
    nullable: bool = True
    help: str = ""


FEATURES = (
    Feature("Age", "Usia", "tahun", maximum=120, integer=True, nullable=False),
    Feature("Annual_Income", "Pendapatan tahunan", "unit uang dataset"),
    Feature("Monthly_Inhand_Salary", "Pendapatan bersih bulanan", "unit uang dataset"),
    Feature("Num_Bank_Accounts", "Jumlah rekening bank", "rekening", integer=True),
    Feature("Num_Credit_Card", "Jumlah kartu kredit", "kartu", integer=True),
    Feature("Interest_Rate", "Suku bunga", "%", maximum=100),
    Feature("Num_of_Loan", "Jumlah pinjaman", "pinjaman", integer=True),
    Feature(
        "Delay_from_due_date",
        "Selisih dari jatuh tempo",
        "hari",
        minimum=None,
        integer=True,
        help="Nilai negatif dipertahankan sebagai selisih sebelum jatuh tempo; definisi sumber masih perlu dikonfirmasi.",
    ),
    Feature(
        "Num_of_Delayed_Payment",
        "Jumlah pembayaran terlambat",
        "kejadian",
        integer=True,
        help="Periode penghitungan belum terdokumentasi pada sumber.",
    ),
    Feature(
        "Changed_Credit_Limit",
        "Perubahan limit kredit",
        "unit dataset",
        minimum=None,
        help="Nilai negatif boleh diisi. Satuan asli belum dipastikan sebagai persen atau nominal.",
    ),
    Feature(
        "Num_Credit_Inquiries", "Jumlah pemeriksaan kredit", "pemeriksaan", integer=True
    ),
    Feature("Outstanding_Debt", "Sisa utang", "unit uang dataset"),
    Feature("Credit_Utilization_Ratio", "Penggunaan kredit", "%", maximum=100),
    Feature("Credit_History_Age", "Usia riwayat kredit", "bulan", integer=True),
    Feature(
        "Payment_of_Min_Amount",
        "Membayar jumlah minimum",
        "biner",
        maximum=1,
        integer=True,
        help="Mengikuti encoding dataset: Ya = 1, Tidak = 0. Makna operasional label belum tervalidasi.",
    ),
    Feature("Total_EMI_per_month", "Total cicilan bulanan", "unit uang dataset"),
    Feature("Amount_invested_monthly", "Investasi bulanan", "unit uang dataset"),
    Feature("Monthly_Balance", "Saldo bulanan", "unit uang dataset", minimum=None),
)
FEATURE_NAMES = tuple(f.name for f in FEATURES)
BY_NAME = {f.name: f for f in FEATURES}


class ContractError(ValueError):
    def __init__(self, issues: list[dict]):
        self.issues = issues
        super().__init__("Input tidak sesuai kontrak data.")


class PredictionError(RuntimeError):
    """A technical failure must never become a credit category."""


def validate_instances(instances: object) -> list[dict]:
    issues = []
    if not isinstance(instances, list) or not 1 <= len(instances) <= MAX_BATCH:
        raise ContractError(
            [{"field": "instances", "message": f"Kirim 1–{MAX_BATCH} profil bernama."}]
        )
    records = []
    for i, record in enumerate(instances):
        if not isinstance(record, dict):
            issues.append(
                {
                    "row": i,
                    "field": "instances",
                    "message": "Setiap profil harus berupa objek dengan nama fitur.",
                }
            )
            continue
        for key in set(record) - set(FEATURE_NAMES):
            issues.append({"row": i, "field": key, "message": "Fitur tidak dikenal."})
        canonical = {}
        for feature in FEATURES:
            value = record.get(feature.name)
            message = None
            if feature.name not in record:
                message = "Fitur wajib dikirim; gunakan null bila nilainya tidak diketahui dan diizinkan."
            elif value is None:
                if not feature.nullable:
                    message = "Nilai wajib diisi."
            elif (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                message = "Gunakan angka terbatas atau null; teks, boolean, NaN dan Infinity tidak diterima."
            elif feature.integer and not float(value).is_integer():
                message = "Gunakan bilangan bulat."
            elif feature.minimum is not None and value < feature.minimum:
                message = f"Nilai minimum {feature.minimum:g}."
            elif feature.maximum is not None and value > feature.maximum:
                message = f"Nilai maksimum {feature.maximum:g}."
            if message:
                issues.append({"row": i, "field": feature.name, "message": message})
            canonical[feature.name] = None if value is None else value
        records.append(canonical)
    if issues:
        raise ContractError(issues)
    return records


def validate_request(payload: object) -> list[dict]:
    if not isinstance(payload, dict) or set(payload) != {
        "contract_version",
        "instances",
    }:
        raise ContractError(
            [
                {
                    "field": "request",
                    "message": "Request harus berisi contract_version dan instances saja.",
                }
            ]
        )
    if payload["contract_version"] != CONTRACT_VERSION:
        raise ContractError(
            [{"field": "contract_version", "message": "Versi kontrak tidak didukung."}]
        )
    return validate_instances(payload["instances"])


def validate_response(response: object, expected_rows: int) -> dict:
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise PredictionError("Respons layanan prediksi tidak valid. Coba lagi.")
    if response.get("contract_version") != CONTRACT_VERSION:
        raise PredictionError("Versi respons layanan tidak cocok.")
    if (
        not isinstance(response.get("model_version"), str)
        or not response["model_version"].strip()
    ):
        raise PredictionError("Versi model tidak tersedia pada respons.")
    labels = response.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) != expected_rows
        or any(label not in CLASSES for label in labels)
    ):
        raise PredictionError("Kategori hasil layanan tidak valid. Coba lagi.")
    if not isinstance(response.get("warnings"), list) or not all(
        isinstance(x, str) for x in response["warnings"]
    ):
        raise PredictionError("Metadata respons layanan tidak valid.")
    return response


def contract_document() -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "classes": list(CLASSES),
        "max_batch": MAX_BATCH,
        "features": [asdict(f) for f in FEATURES],
        "money_unit": "Unspecified source dataset units; no currency conversion",
    }
