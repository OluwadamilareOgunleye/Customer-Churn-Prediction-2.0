"""Integration and End-to-End workflow tests for Churn Prediction System."""

import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from churn_prediction.config import PROCESSED_DATA_DIR
from churn_prediction.model_inference import ChurnPredictor
from frontend.utils.api_client import ChurnAPIClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def test_dataset():
    path = PROCESSED_DATA_DIR / "test.csv"
    if not path.exists():
        pytest.skip(f"Test dataset not found at {path}")
    return pd.read_csv(path)


def test_e2e_single_customer_lifecycle(client):
    """End-to-end test: Health check -> Model Info -> Single Prediction -> Verification."""
    # 1. Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["model_loaded"] is True

    # 2. Model info
    info_resp = client.get("/model-info")
    assert info_resp.status_code == 200
    assert info_resp.json()["decision_threshold"] == 0.30

    # 3. Submit single prediction
    customer = {
        "CustomerID": "E2E-TEST-001",
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 1,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 75.0,
        "Total Charges": 75.0,
    }
    pred_resp = client.post("/predict", json=customer)
    assert pred_resp.status_code == 200
    data = pred_resp.json()
    assert data["customer_id"] == "E2E-TEST-001"
    assert data["churn_prediction"] == 1
    assert data["churn_label"] == "Yes"
    assert data["risk_category"] in {"Medium Risk", "High Risk"}


def test_e2e_batch_csv_workflow(client, test_dataset):
    """End-to-end test: Sample from test.csv -> Serialize -> Batch Inference -> CSV Output."""
    sample_records = test_dataset.head(10).drop(columns=["Churn Value"], errors="ignore").to_dict(orient="records")
    payload = {"records": sample_records}

    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["total_records"] == 10
    assert len(res_data["predictions"]) == 10

    # Verify CSV export structure
    df_out = pd.DataFrame(res_data["predictions"])
    assert "churn_probability" in df_out.columns
    assert "risk_category" in df_out.columns


def test_e2e_frontend_api_client_fallback(test_dataset):
    """Verify frontend ChurnAPIClient functions seamlessly with local engine fallback."""
    api_client = ChurnAPIClient(base_url="http://invalid-url-that-fails.local:9999")
    # Health check should fallback to local engine
    health = api_client.check_health()
    assert health["model_loaded"] is True
    assert "direct engine" in health["status"]

    # Prediction should fallback gracefully
    record = test_dataset.iloc[0].drop("Churn Value", errors="ignore").to_dict()
    pred = api_client.predict_single(record)
    assert "churn_probability" in pred
    assert pred["decision_threshold"] == 0.30


def test_input_validation_resilience_to_extra_columns():
    """Verify inference engine safely ignores extra, unexpected columns."""
    predictor = ChurnPredictor()
    sample = {
        "CustomerID": "EXTRA-COL-TEST",
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "One year",
        "Paperless Billing": "No",
        "Payment Method": "Credit card (automatic)",
        "Monthly Charges": 50.0,
        "Total Charges": 600.0,
        "Extra_Unrelated_Field": "Some random value",
        "Another_Bonus_Column": 12345,
    }
    res = predictor.predict_single(sample)
    assert res["churn_prediction"] in {0, 1}
    assert res["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
