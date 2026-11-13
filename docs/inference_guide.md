# Phase 7: Model Inference and Prediction System Guide

## 1. Overview & Architecture

The inference engine exposes a clean, decoupled interface for scoring customer churn records in real-time or batch pipelines.

### Key Capabilities
- **Strict Schema Validation:** Validates required columns, bounds (non-negative numeric spend and tenure), and categorical enumerations.
- **Graceful Fallbacks:** In non-strict mode, unseen categorical levels encounter `OneHotEncoder(handle_unknown="ignore")` and emit transparent warnings rather than terminating execution.
- **Calibrated Decision Threshold:** Employs the empirical **$\tau = 0.30$** threshold calibrated in Phase 5 to maximize recall and retention capture.
- **Risk Stratification:** Classifies probability $\hat{p}$ into justified operational tiers (`Low Risk`, `Medium Risk`, `High Risk`).

---

## 2. Risk Tier Definitions

| Risk Category | Calibrated Probability Range | Operational Interpretation | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Low Risk** | $\hat{p} < 0.30$ | Below the intervention threshold. Subscriber is stable with strong anchors. | No active intervention; protect account profitability. |
| **Medium Risk** | $0.30 \le \hat{p} < 0.60$ | Above decision threshold. Exhibiting early defection markers (e.g. month-to-month contract, lack of support add-on). | Low-cost proactive retention nudges (digital tips, feature discovery, self-service tools). |
| **High Risk** | $\hat{p} \ge 0.60$ | Acute defection hazard. High probability of discontinuing service within the current cycle. | Direct customer success outreach, loyalty contract renewal incentives, discount offers. |

---

## 3. Python API Usage

### Single Customer Scoring
```python
from churn_prediction.model_inference import ChurnPredictor

# Initialize engine
predictor = ChurnPredictor()

# Customer dictionary
customer = {
    "CustomerID": "CUST-9876",
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
    "Total Charges": 269.55,
}

result = predictor.predict_single(customer)
print(result)
```

**Output:**
```json
{
  "customer_id": "CUST-9876",
  "churn_probability": 0.7255,
  "churn_prediction": 1,
  "churn_label": "Yes",
  "risk_category": "High Risk",
  "decision_threshold": 0.30,
  "warnings": [],
  "calibration_note": "Probability represents estimated empirical likelihood from cross-validated gradient boosting. It is a risk propensity score, not a deterministic guarantee of departure."
}
```

### Batch DataFrame Scoring
```python
import pandas as pd
from churn_prediction.model_inference import ChurnPredictor

predictor = ChurnPredictor()
df_batch = pd.read_csv("data/processed/test.csv")

scored_df, warnings = predictor.predict_batch(df_batch)
print(scored_df[["CustomerID", "churn_probability", "churn_label", "risk_category"]].head())
```
