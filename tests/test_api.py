"""Unit and integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    """Valid customer input dictionary."""
    return {
        "CustomerID": "TEST-1001",
        "Gender": "Male",
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
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.50,
        "Total Charges": 171.00,
    }


def test_root_endpoint(client):
    """Verify root / endpoint returns operational status and links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "operational"
    assert "/docs" in data["docs_url"]


def test_health_endpoint(client):
    """Verify /health endpoint reports healthy and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    """Verify /model-info returns metadata, features, and decision threshold."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 19
    assert data["decision_threshold"] == 0.30
    assert "metrics" in data


def test_predict_single_endpoint(client, valid_customer_payload):
    """Verify /predict returns valid prediction, probability, and risk category."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "TEST-1001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in {0, 1}
    assert data["churn_label"] in {"Yes", "No"}
    assert data["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert data["decision_threshold"] == 0.30


def test_predict_single_negative_tenure_fails(client, valid_customer_payload):
    """Verify Pydantic validation rejects negative numeric values."""
    payload = valid_customer_payload.copy()
    payload["Tenure Months"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_missing_field_fails(client, valid_customer_payload):
    """Verify Pydantic validation rejects incomplete payload."""
    payload = valid_customer_payload.copy()
    del payload["Contract"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_endpoint(client, valid_customer_payload):
    """Verify /predict/batch scores multiple customer records."""
    second_customer = valid_customer_payload.copy()
    second_customer["CustomerID"] = "TEST-2002"
    second_customer["Contract"] = "Two year"
    second_customer["Tenure Months"] = 65

    batch_payload = {"records": [valid_customer_payload, second_customer]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["customer_id"] == "TEST-1001"
    assert data["predictions"][1]["customer_id"] == "TEST-2002"


def test_predict_batch_empty_fails(client):
    """Verify empty batch request returns error."""
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code == 400
