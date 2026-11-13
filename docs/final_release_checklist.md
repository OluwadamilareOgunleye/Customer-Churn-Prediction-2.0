# Final Portfolio Release & Deployment Verification Checklist

This checklist confirms verification across all critical phases of the Customer Churn Prediction project.

---

## 1. Data Integrity & Preprocessing
- [x] Raw IBM dataset intact with verified SHA-256 hash (`1BCBC0CCC9B352175216979102628E579CFBDE2C3FF57B005DE168A433122640`).
- [x] Target leakage columns (`Churn Reason`, `Churn Score`, `CLTV`) strictly pruned.
- [x] Preprocessing ColumnTransformer fully bundled inside serialized pipeline.
- [x] Stratified 80/20 train/test split preserved without leakage.

## 2. Model Performance & Evaluation
- [x] 5-fold cross-validation benchmarking completed across candidate models.
- [x] Test set holdout ROC-AUC verified at **0.8570**; PR-AUC at **0.6740**.
- [x] Decision threshold calibrated to **$\tau = 0.30$** (Recall: 77.01%, Precision: 54.24%).
- [x] Probability calibration verified via Brier Score (**0.1322**).
- [x] Subgroup performance audits completed across Contract, Tenure, Payment, and Internet Service.

## 3. Serving & Application Stack
- [x] FastAPI REST service operational with Pydantic v2 validation.
- [x] Liveness (`/health`) and Readiness (`/ready`) probes active.
- [x] Streamlit web application running with SHAP explainability and K-Means segmentation.
- [x] Interactive Retention Campaign ROI simulator active in UI.
- [x] Autonomous dual-engine fallback in Streamlit client.

## 4. Software Quality & Automated Testing
- [x] 34 automated pytest tests passing with **100% success rate**.
- [x] End-to-end integration tests verified.
- [x] Model registry rollback tests verified.
- [x] Code coverage on active serving modules $\ge 79\%$.

## 5. Security & Privacy
- [x] Sliding-window rate limiting active (180 req/min).
- [x] Batch size limit (2,500 records) and payload size limit (10 MB) enforced.
- [x] Zero committed secrets; `.env` and `mlflow.db` in `.gitignore`.
- [x] Customer IDs masked in telemetry (`7590-****`); zero raw PII logged.
- [x] Dependency scan completed (0 critical or high vulnerabilities).

## 6. MLOps, CI/CD & Infrastructure
- [x] MLflow experiment tracking integrated with SQLite backend (`mlflow.db`).
- [x] Model registry implemented with promotion and rollback gates (`models/registry.json`).
- [x] GitHub Actions CI/CD workflows prepared (`.github/workflows/`).
- [x] Dockerfile.api, Dockerfile.frontend, and docker-compose.yml verified.
- [x] Terraform specification authored (`infrastructure/main.tf`).
- [x] Zero paid cloud resources incurred.

## 7. Portfolio Presentation Assets
- [x] 15-slide PowerPoint deck generated (`presentations/customer_churn_project.pptx`).
- [x] Comprehensive technical case study authored (`docs/technical_case_study.md`).
- [x] Technical interview preparation Q&A guide authored (`docs/interview_preparation.md`).
- [x] Live demonstration script authored (`docs/project_demonstration_guide.md`).
- [x] 24 publication figures generated in `reports/figures/`.
- [x] Master README updated.
