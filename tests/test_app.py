"""UI state regression with the actual local bundle; cloud responses are injected."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from credit_score.contract import PredictionError

APP = Path(__file__).resolve().parents[1] / "app_streamlit.py"


def app():
    result = AppTest.from_file(str(APP)).run(timeout=30)
    assert not result.exception
    return result


def test_missing_bundle_shows_recovery_and_no_form(monkeypatch, tmp_path):
    monkeypatch.setenv("CREDIT_SCORE_ARTIFACTS", str(tmp_path))
    at = app()
    assert any("Model lokal belum siap" in item.value for item in at.error)
    assert not at.text_input
    assert "prediction" not in at.session_state


def submit(at):
    next(
        button for button in at.button if button.label == "Prediksi kategori"
    ).click().run(timeout=30)
    return at


def test_missing_age_is_actionable_and_preserves_other_inputs():
    at = app()
    salary = at.text_input(key="feature_Annual_Income").value
    at.text_input(key="feature_Age").set_value("").run(timeout=30)
    submit(at)
    assert not at.exception
    assert "prediction" not in at.session_state
    assert any("Usia:" in item.value and "wajib" in item.value for item in at.error)
    assert at.text_input(key="feature_Annual_Income").value == salary


@pytest.mark.parametrize("value", ["999", "33.5", "not a number"])
def test_invalid_typed_age_never_predicts_and_reset_recovers(value):
    at = app()
    submit(at)
    at.text_input(key="feature_Age").set_value(value).run(timeout=30)
    assert "prediction" not in at.session_state
    submit(at)
    assert "prediction" not in at.session_state
    assert any("Usia:" in item.value for item in at.error)
    next(
        button for button in at.button if button.label == "Muat contoh nilai dataset"
    ).click().run(timeout=30)
    submit(at)
    assert "prediction" in at.session_state


def test_invalid_schema_is_rejected_before_any_aws_request(monkeypatch):
    calls = []
    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "controlled-test")
    monkeypatch.setattr(
        "credit_score.client.invoke_sagemaker", lambda payload: calls.append(payload)
    )
    at = app()
    at.radio[0].set_value("AWS SageMaker").run(timeout=30)
    at.text_input(key="feature_Age").set_value("150").run(timeout=30)
    submit(at)
    assert not calls
    assert "prediction" not in at.session_state
    assert any("Usia:" in item.value for item in at.error)


def test_outside_training_range_gets_result_and_keeps_original_snapshot():
    at = app()
    at.text_input(key="feature_Age").set_value("70").run(timeout=30)
    at.text_input(key="feature_Annual_Income").set_value("120000000").run(timeout=30)
    submit(at)
    assert not at.exception and not at.error
    assert "prediction" in at.session_state
    assert at.session_state["submitted_inputs"]["Annual_Income"] == 120000000
    assert any("belum teruji" in item.value for item in at.warning)


def test_nullable_input_is_disclosed_and_snapshot_remains_missing():
    at = app()
    at.text_input(key="feature_Monthly_Inhand_Salary").set_value("").run(timeout=30)
    submit(at)
    assert not at.exception
    assert "prediction" in at.session_state
    assert at.session_state["submitted_inputs"]["Monthly_Inhand_Salary"] is None
    assert any("median data training" in item.value for item in at.warning)


def test_missing_evidence_does_not_transfer_metrics_from_another_model(monkeypatch):
    def mismatch(root, metadata):
        raise ValueError("Wrong report")

    monkeypatch.setattr("credit_score.presentation.load_evidence", mismatch)
    at = app()
    assert any("cocok dengan bundle" in item.value for item in at.warning)
    submit(at)
    assert "prediction" in at.session_state


def test_defaults_predict_and_edit_clear_result():
    at = app()
    assert len(at.text_input) == 17
    assert len(at.selectbox) == 1
    assert not at.error
    submit(at)
    assert not at.exception
    assert "prediction" in at.session_state
    assert not at.error
    widget = at.text_input(key="feature_Age")
    widget.set_value(str(float(widget.value) + 1)).run(timeout=30)
    assert "prediction" not in at.session_state


def test_reset_restores_training_example_and_clears_prediction(service):
    at = app()
    at.text_input(key="feature_Age").set_value("40").run(timeout=30)
    submit(at)
    next(
        button for button in at.button if button.label == "Muat contoh nilai dataset"
    ).click().run(timeout=30)
    assert at.text_input(key="feature_Age").value == str(
        int(service.metadata["defaults"]["Age"])
    )
    assert "prediction" not in at.session_state


def test_aws_missing_configuration_does_not_show_a_credit_result(monkeypatch):
    monkeypatch.delenv("CREDIT_SCORE_ENDPOINT", raising=False)
    at = app()
    at.radio[0].set_value("AWS SageMaker").run(timeout=30)
    submit(at)
    assert not at.exception
    assert "prediction" not in at.session_state
    assert any("belum diatur" in message.value for message in at.error)


@pytest.mark.parametrize("label", ["Good", "Poor", "Standard"])
def test_each_valid_category_has_neutral_copy(monkeypatch, service, label):
    response = {
        "status": "ok",
        "contract_version": "1.0",
        "model_version": service.metadata["model_version"],
        "labels": [label],
        "warnings": [],
    }
    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "controlled-test")
    monkeypatch.setattr(
        "credit_score.client.invoke_sagemaker", lambda payload: response
    )
    at = app()
    at.radio[0].set_value("AWS SageMaker").run(timeout=30)
    submit(at)
    assert not at.exception
    assert at.session_state["prediction"]["labels"] == [label]
    assert not at.error
    assert any(
        "Kategori model" in item.value and label in item.value for item in at.markdown
    )


def test_unexpected_backend_failure_is_recoverable(monkeypatch):
    def fail(payload):
        raise PredictionError("Layanan sementara tidak tersedia. Coba lagi.")

    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "controlled-test")
    monkeypatch.setattr("credit_score.client.invoke_sagemaker", fail)
    at = app()
    at.radio[0].set_value("AWS SageMaker").run(timeout=30)
    submit(at)
    assert not at.exception
    assert "prediction" not in at.session_state
    assert any("sementara tidak tersedia" in item.value for item in at.error)


def test_model_version_mismatch_never_shows_prediction(monkeypatch):
    response = {
        "status": "ok",
        "contract_version": "1.0",
        "model_version": "different-model",
        "labels": ["Good"],
        "warnings": [],
    }
    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "controlled-test")
    monkeypatch.setattr(
        "credit_score.client.invoke_sagemaker", lambda payload: response
    )
    at = app()
    at.radio[0].set_value("AWS SageMaker").run(timeout=30)
    submit(at)
    assert not at.exception
    assert "prediction" not in at.session_state
    assert any("Versi model AWS berbeda" in item.value for item in at.error)


@pytest.mark.parametrize(
    "response", [{}, {"labels": []}, {"labels": ["UNKNOWN"]}, {"labels": [None]}]
)
def test_historical_invalid_response_never_becomes_poor_or_crashes(
    monkeypatch, response
):
    monkeypatch.setenv("CREDIT_SCORE_ENDPOINT", "controlled-test")
    monkeypatch.setattr(
        "credit_score.client.invoke_sagemaker", lambda payload: response
    )
    at = app()
    at.radio[0].set_value("AWS SageMaker").run(timeout=30)
    submit(at)
    assert not at.exception
    assert "prediction" not in at.session_state
    assert at.error
    assert all("BURUK" not in item.value for item in at.error)
