import json
from pathlib import Path

import pytest

from credit_score.presentation import (
    category_markup,
    form_default,
    load_evidence,
    parse_form_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def test_evidence_requires_matching_bytes_and_active_model(tmp_path, service):
    meta = dict(service.metadata)
    report = load_evidence(ROOT, meta)
    assert report["model_version"] == meta["model_version"]
    meta["model_version"] = "other-version"
    with pytest.raises(ValueError, match="active model"):
        load_evidence(ROOT, meta)
    target = tmp_path / "docs/evaluation/stage-two/holdout-results.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(report) + " ")
    with pytest.raises(ValueError, match="hash"):
        load_evidence(tmp_path, service.metadata)


@pytest.mark.parametrize("label", ["Good", "Standard", "Poor"])
def test_category_has_one_text_marker_and_no_numeric_score(label):
    markup = category_markup(label)
    assert markup.count('aria-current="true"') == 1
    assert f">{label}<small>Terpilih</small>" in markup
    assert "progressbar" not in markup
    assert "%" not in markup


def test_empty_category_is_not_implicitly_poor():
    assert "aria-current" not in category_markup()
    with pytest.raises(ValueError):
        category_markup("UNKNOWN")


def test_decimal_parser_keeps_sign_missingness_and_fractional_counts():
    values, issues = parse_form_inputs(
        {
            "Annual_Income": "1250,50",
            "Changed_Credit_Limit": "-2.5",
            "Num_of_Loan": "1.5",
            "Monthly_Balance": " ",
            "Payment_of_Min_Amount": None,
        }
    )
    assert not issues
    assert values == {
        "Annual_Income": 1250.5,
        "Changed_Credit_Limit": -2.5,
        "Num_of_Loan": 1.5,
        "Monthly_Balance": None,
        "Payment_of_Min_Amount": None,
    }


@pytest.mark.parametrize("text", ["1.234,56", "NaN", "Infinity", "ten", "9" * 400])
def test_invalid_numeric_text_cannot_become_a_profile(text):
    values, issues = parse_form_inputs({"Annual_Income": text})
    assert "Annual_Income" not in values
    assert issues[0]["field"] == "Annual_Income"


def test_displayed_rounded_example_is_the_profile_actually_predicted(service):
    raw = {
        name: form_default(name, value)
        for name, value in service.metadata["defaults"].items()
    }
    values, issues = parse_form_inputs(raw, service.metadata["supported_ranges"])
    assert not issues
    assert raw["Monthly_Inhand_Salary"] == "3081.17"
    assert values["Monthly_Inhand_Salary"] == 3081.17
    from credit_score.service import request_for

    assert service.predict(request_for(values))["status"] == "ok"
