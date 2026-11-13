# Responsible Data Privacy & Governance Specification

## 1. Data Minimization Principles

The Customer Churn Prediction system enforces privacy-by-design standards aligned with responsible machine learning engineering:

1. **Pruning Non-Essential Identifiers:**
   - Raw telecommunications customer records often contain geographic or personal information (names, street addresses, exact GPS coordinates, IP addresses).
   - In `src/churn_prediction/config.py`, all high-cardinality spatial identifiers (`City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) are **completely excluded** from model feature pipelines.
2. **Identifier Masking:**
   - The operational `CustomerID` attribute is strictly utilized as a routing key and is never fed as a numerical/categorical predictor to estimators.
   - Telemetry logging redacts customer IDs to prefix form (`7590-****`) via `src/churn_prediction/security.py`.

---

## 2. Non-PII Runtime Telemetry

- **Operational Logging:** When inference requests are processed, the system records only:
  - Timestamp (UTC)
  - Latency in milliseconds
  - Calibrated churn probability ($0.0 \dots 1.0$)
  - Predicted risk category (`Low`, `Moderate`, `Elevated`, `High`)
  - Aggregated input feature distributions for Population Stability Index (PSI) drift tracking.
- **Zero Raw Data Persistence:** Individual customer payloads submitted to `/predict` or `/predict/batch` are held only in volatile container memory during execution and are discarded immediately after response serialization.

---

## 3. Data Retention & Compliance Notice

- **Development Data Retention:** Processed training and testing splits (`data/processed/`) are stored locally for reproducible validation. In production deployments, training splits should follow a 90-day retention schedule with automated purge jobs.
- **Regulatory Frameworks (GDPR / CCPA):** While this project demonstrates responsible privacy controls and PII masking, production deployment in regulated jurisdictions requires formal Data Protection Impact Assessments (DPIA), privacy notices, and "Right to be Forgotten" deletion pipelines.
