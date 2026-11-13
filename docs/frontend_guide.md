# Phase 9: Streamlit Frontend Interface Guide

## 1. Overview & Architecture

The Customer Churn Prediction user interface is built with **Streamlit**. It decouples UI presentation from machine learning logic by communicating directly with the FastAPI backend via `ChurnAPIClient` (located in `frontend/utils/api_client.py`). A built-in fallback automatically routes requests to the local `ChurnPredictor` if the FastAPI service is not running.

---

## 2. Navigation & Interface Sections

### 1. Overview & Dashboard
- **Executive Metrics:** Displays verified test-set performance metrics (ROC-AUC 0.8570, PR-AUC 0.6740, Recall 77.01%, Precision 54.24%).
- **Threshold Context:** Explains the business justification for calibrating the classification threshold to `0.30`.
- **System Health:** Live connection indicator reflecting API and model pipeline status.

### 2. Single Customer Prediction
- **Interactive Form:** Collects 19 customer features across Demographics, Account Terms, Subscribed Services, and Billing Details.
- **Risk Categorization:** Displays churn probability ($\hat{p}$), discrete prediction (`Yes`/`No`), and color-coded risk badge (`Low Risk`, `Medium Risk`, `High Risk`).
- **Actionable Guidance:** Dynamically presents tailored retention recommendations based on customer attributes and risk level.

### 3. Batch Prediction & CSV Export
- **File Upload:** Ingests CSV files containing multiple customer records.
- **Pre-loaded Test Set:** One-click button loads 100 sample records from `data/processed/test.csv` for demonstration.
- **Bulk Inference:** Scores records in parallel and displays aggregated risk tier counts.
- **Data Export:** Allows users to download the scored dataset as a CSV file (`churn_predictions_scored.csv`).

### 4. Model Performance & Evaluation Curves
- Displays test set diagnostic visualizations:
  - Confusion Matrix at Calibrated Threshold ($\tau = 0.30$) vs Standard Threshold ($\tau = 0.50$).
  - Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves.
  - Threshold Sensitivity & Trade-off Curve.

### 5. Business Insights & Retention Strategy
- Visualizes global SHAP feature importance and contract retention differentials.
- Outlines the 3-pillar Strategic Retention Playbook:
  1. First-Year Onboarding & Hazard Mitigation.
  2. Annual Contract Lock-in Incentives.
  3. Fiber Optic Technical Support Bundling.

---

## 3. Launching the Frontend Locally

```powershell
.\.venv\Scripts\streamlit.exe run frontend/app.py --server.port 8501
```
The application will open at `http://localhost:8501`.
