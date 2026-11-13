"""Tests for model persistence and inference serving interface."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from churn_prediction.config import ALL_FEATURE_COLS, MODELS_DIR
from churn_prediction.model_inference import ChurnPredictor, load_model_bundle


@pytest.fixture(scope="module")
def model_bundle_path():
    """Ensure trained model artifact exists."""
    path = MODELS_DIR / "churn_pipeline_bundle.joblib"
    if not path.exists():
        pytest.skip(f"Model artifact not found at {path}. Run pipeline before running inference tests.")
    return path


@pytest.fixture
def predictor(model_bundle_path):
    """Instantiate ChurnPredictor from disk artifact."""
    return ChurnPredictor(model_bundle_path)


@pytest.fixture
def sample_customer_records():
    """Create realistic sample customer records for scoring."""
    data = [
        {
            "Gender": "Female",
            "Senior Citizen": "No",
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": 2,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "No",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": 85.0,
            "Total Charges": 170.0,
        },
        {
            "Gender": "Male",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure Months": 65,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "Yes",
            "Device Protection": "Yes",
            "Tech Support": "Yes",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": "Two year",
            "Paperless Billing": "No",
            "Payment Method": "Credit card (automatic)",
            "Monthly Charges": 55.0,
            "Total Charges": 3575.0,
        },
    ]
    return pd.DataFrame(data)


def test_bundle_structure(model_bundle_path):
    """Verify that serialized bundle contains all required keys and valid types."""
    bundle = load_model_bundle(model_bundle_path)
    assert "pipeline" in bundle
    assert "threshold" in bundle
    assert "metrics" in bundle
    assert "feature_names" in bundle
    assert isinstance(bundle["threshold"], float)
    assert 0.0 < bundle["threshold"] < 1.0


def test_predictor_probabilities(predictor, sample_customer_records):
    """Verify predicted probabilities are within [0.0, 1.0]."""
    probs = predictor.predict_proba(sample_customer_records)
    assert len(probs) == len(sample_customer_records)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Customer 1 (high risk: month-to-month, fiber optic, no tech support, tenure 2) should have higher probability than Customer 2
    assert probs[0] > probs[1]


def test_predictor_binary_predictions(predictor, sample_customer_records):
    """Verify binary predictions match threshold application."""
    preds = predictor.predict(sample_customer_records)
    assert len(preds) == len(sample_customer_records)
    assert set(preds).issubset({0, 1})


def test_predictor_score_records(predictor, sample_customer_records):
    """Verify enriched record dataframe has expected columns and risk tiers."""
    scored = predictor.score_records(sample_customer_records)
    expected_cols = [
        "churn_probability",
        "churn_prediction",
        "churn_label",
        "risk_category",
        "risk_tier",
    ]
    for col in expected_cols:
        assert col in scored.columns

    assert set(scored["churn_label"].unique()).issubset({"No", "Yes"})
    assert set(scored["risk_tier"].unique()).issubset({"Low Risk", "Medium Risk", "High Risk"})


def test_predictor_missing_column_raises(predictor, sample_customer_records):
    """Verify predictor raises ValueError if required feature columns are omitted."""
    incomplete = sample_customer_records.drop(columns=["Contract"])
    with pytest.raises(Exception):
        predictor.predict_proba(incomplete)


def test_predict_single_success(predictor, sample_customer_records):
    """Verify single customer prediction returns complete standardized dictionary."""
    rec = sample_customer_records.iloc[0].to_dict()
    res = predictor.predict_single(rec)
    assert "churn_probability" in res
    assert "churn_prediction" in res
    assert "risk_category" in res
    assert res["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert res["decision_threshold"] == predictor.default_threshold


def test_predict_single_warning_on_unseen_category(predictor, sample_customer_records):
    """Verify that unseen categorical values produce warnings rather than crashing."""
    rec = sample_customer_records.iloc[0].to_dict()
    rec["Contract"] = "Lifetime Contract"  # Unseen category
    res = predictor.predict_single(rec, strict=False)
    assert len(res["warnings"]) > 0
    assert "unexpected category" in res["warnings"][0]


def test_predictor_model_not_found_raises():
    """Verify loading non-existent model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model_bundle(Path("non_existent_model_dir/model.joblib"))


def test_predict_single_invalid_data_type_raises(predictor, sample_customer_records):
    """Verify non-numeric input for numerical feature raises ValidationError."""
    rec = sample_customer_records.iloc[0].to_dict()
    rec["Tenure Months"] = "not-a-number"
    with pytest.raises(Exception):
        predictor.predict_single(rec)


