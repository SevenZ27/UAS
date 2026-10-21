"""Accessible presentation of categories and frozen aggregate evidence, no ML fitting."""

import hashlib
import json
import math
import re
from html import escape
from pathlib import Path

from .contract import BY_NAME, CLASSES

LABELS = {"Good": "Baik", "Standard": "Standar", "Poor": "Kurang baik"}


def input_label(feature):
    if feature.unit in {"unit uang dataset", "unit dataset"}:
        return feature.label
    return f"{feature.label} ({feature.unit})"


def form_default(name, value):
    if name == "Payment_of_Min_Amount":
        return value
    return str(int(value)) if BY_NAME[name].integer else f"{value:.2f}"


def parse_form_inputs(raw, ranges=None):
    """No thousands separator guessing or silent truncation of integer fields."""
    values, issues = {}, []
    for name, value in raw.items():
        if name == "Payment_of_Min_Amount":
            values[name] = value
            continue
        text = value.strip()
        if not text:
            values[name] = None
        elif not re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)", text):
            issues.append(
                {
                    "field": name,
                    "message": "Gunakan angka tanpa pemisah ribuan. Tanda desimal boleh titik atau koma.",
                }
            )
        else:
            numeric = float(text.replace(",", "."))
            if not math.isfinite(numeric):
                issues.append(
                    {
                        "field": name,
                        "message": "Gunakan angka terbatas; nilai terlalu besar tidak diterima.",
                    }
                )
            else:
                values[name] = numeric
    return values, issues


def number(value, digits=2):
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def load_evidence(root, metadata):
    """Return only the known report whose bytes and model match the active bundle."""
    path = Path(root) / "docs/evaluation/stage-two/holdout-results.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata.get("evaluation_report_sha256"):
        raise ValueError("Evaluation report hash differs from active metadata")
    report = json.loads(raw)
    if (
        report["model_version"] != metadata["model_version"]
        or report["model_sha256"] != metadata["model_sha256"]
        or report["classes"] != list(CLASSES)
    ):
        raise ValueError("Evaluation does not describe the active model")
    return report


def category_markup(selected=None):
    if selected is not None and selected not in CLASSES:
        raise ValueError("Unknown category")
    items = []
    for label in ("Poor", "Standard", "Good"):
        current = ' aria-current="true"' if label == selected else ""
        note = "Terpilih" if label == selected else ""
        items.append(f"<li{current}>{label}<small>{note}</small></li>")
    return (
        '<ul class="category-list" aria-label="Kategori keluaran model">'
        + "".join(items)
        + "</ul>"
    )


def comparison_markup(report):
    rows = []
    for key, label, selected in [
        ("selected", "Random Forest terpilih", True),
        ("prior_baseline", "Baseline kelas mayoritas", False),
    ]:
        score = report[key]["accuracy"] * 100
        style = "bar-fill selected" if selected else "bar-fill"
        rows.append(
            f'<div class="bar-label"><span>{label}</span><strong>{number(score)}%</strong></div>'
            f'<div class="bar-track" aria-hidden="true"><span class="{style}" style="width:{score:.6f}%"></span></div>'
        )
    return (
        '<div class="comparison">'
        + "".join(rows)
        + '<div class="bar-axis" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div></div>'
    )


def confusion_markup(report):
    labels = report["classes"]
    head = (
        '<tr><th scope="col">Aktual / prediksi</th>'
        + "".join(f'<th scope="col">{escape(x)}</th>' for x in labels)
        + "</tr>"
    )
    rows = []
    for i, label in enumerate(labels):
        values = report["confusion_matrix"][i]
        cells = []
        for j, count in enumerate(values):
            style = ' class="diagonal"' if i == j else ""
            cells.append(
                f"<td{style}>{number(count, 0)}<small>{number(count / sum(values) * 100, 1)}%</small></td>"
            )
        rows.append(
            f'<tr><th scope="row">{escape(label)}</th>' + "".join(cells) + "</tr>"
        )
    return (
        '<div class="table-scroll" role="region" aria-label="Tabel confusion matrix" tabindex="0"><table class="data-table matrix"><caption>Confusion matrix holdout; persen per kelas aktual</caption><thead>'
        + head
        + "</thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def snapshot_markup(inputs):
    rows = []
    for name, value in inputs.items():
        feature = BY_NAME[name]
        if value is None:
            text = "Kosong; diimputasi median training"
        elif name == "Payment_of_Min_Amount":
            text = "Ya" if value else "Tidak"
        else:
            suffix = (
                ""
                if feature.unit in {"unit uang dataset", "unit dataset"}
                else " " + feature.unit
            )
            text = number(value, 0 if feature.integer else 2) + suffix
        rows.append(
            f'<tr><th scope="row">{escape(feature.label)}</th><td>{escape(text)}</td></tr>'
        )
    return (
        '<div class="table-scroll" role="region" aria-label="Input saat prediksi" tabindex="0"><table class="data-table"><caption>Profil yang dipakai pada hasil ini</caption><thead><tr><th scope="col">Fitur</th><th scope="col">Nilai</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
