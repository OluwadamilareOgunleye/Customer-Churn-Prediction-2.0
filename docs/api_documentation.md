# Phase 8: Backend REST API Documentation

## 1. Overview

The Customer Churn Prediction REST API is built with **FastAPI**. It exposes synchronous and batch prediction endpoints, health checks, and metadata endpoints.

- **Base URL (Local):** `http://127.0.0.1:8000`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **Alternative ReDoc UI:** `http://127.0.0.1:8000/redoc`

---

## 2. API Endpoints

### A. Root
- **`GET /`**
- **Description:** Returns service name, version, and operational links.
- **Response `200 OK`:**
  ```json
  {
    "message": "Welcome to the Telco Customer Churn Prediction API",
    "version": "1.0.0",
    "docs_url": "/docs",
    "health_check": "/health",
    "model_info": "/model-info",
    "status": "operational"
  }
  ```

### B. Health Probe
- **`GET /health`**
- **Description:** Liveness probe reporting memory residency of the model pipeline.
- **Response `200 OK`:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0",
    "app_name": "Customer Churn Prediction API"
  }
  ```

### C. Model Metadata
- **`GET /model-info`**
- **Description:** Returns classifier type, calibrated threshold, feature list, and evaluation metrics.
- **Response `200 OK`:**
  ```json
  {
    "model_name": "Customer Churn Prediction Engine",
    "model_type": "HistGradientBoostingClassifier",
    "version": "1.0.0",
    "decision_threshold": 0.30,
    "feature_count": 19,
    "features": ["Tenure Months", "Monthly Charges", "Total Charges", ...],
    "metrics": {
      "calibrated_threshold": {
        "roc_auc": 0.8570,
        "pr_auc": 0.6740,
        "recall": 0.7701,
        "precision": 0.5424,
        "f1": 0.6365,
        "accuracy": 0.7665
      }
    }
  }
  ```

### D. Single Customer Prediction
- **`POST /predict`**
- **Request Body:** JSON dictionary containing all 19 customer features.
  ```json
  {
    "CustomerID": "CUST-7043",
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 3,
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
    "Monthly Charges": 89.85,
    "Total Charges": 269.55
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "customer_id": "CUST-7043",
    "churn_probability": 0.7255,
    "churn_prediction": 1,
    "churn_label": "Yes",
    "risk_category": "High Risk",
    "decision_threshold": 0.30,
    "warnings": [],
    "calibration_note": "Probability represents estimated empirical likelihood from cross-validated gradient boosting. It is a risk propensity score, not a deterministic guarantee of departure."
  }
  ```

### E. Batch Customer Prediction
- **`POST /predict/batch`**
- **Request Body:**
  ```json
  {
    "records": [
      { ...customer_1... },
      { ...customer_2... }
    ]
  }
  ```
- **Response `200 OK`:**
  ```json
  {
    "total_records": 2,
    "predictions": [
      {
        "customer_id": "CUST-1",
        "churn_probability": 0.7255,
        "churn_prediction": 1,
        "churn_label": "Yes",
        "risk_category": "High Risk",
        ...
      },
      ...
    ]
  }
  ```

---

## 3. Starting the Server Locally

```powershell
.\.venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000 --reload
```
