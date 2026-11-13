# Comprehensive Project Report: Telecom Customer Churn Prediction System

**Author:** Data Science & Machine Learning Engineering Team  
**Dataset:** IBM Telco Customer Churn (7,043 Records)  
**Status:** Phases 1–14 Implemented, Verified, and Containerized  
**Test Suite Status:** 28 / 28 Tests Passed (100% Pass Rate)

---

## 1. Project Overview & Deliverables Summary

| Phase | Title | Primary Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Problem Definition & Business Understanding | `docs/problem_definition.md` | **Completed & Verified** |
| **Phase 2** | Data Acquisition & Understanding | `docs/dataset_documentation.md`, `data/raw/Telco_customer_churn.xlsx` | **Completed & Verified** |
| **Phase 3** | Exploratory Data Analysis (EDA) | `notebooks/03_exploratory_data_analysis.ipynb`, `docs/eda_findings.md`, `reports/figures/01-08` | **Completed & Verified** |
| **Phase 4** | Data Quality, Cleaning & Preprocessing | `notebooks/04_data_cleaning_preprocessing.ipynb`, `docs/preprocessing_decisions.md`, `data/processed/train.csv`, `test.csv` | **Completed & Verified** |
| **Phase 5** | Model Development, Training & Evaluation | `notebooks/05_model_development_evaluation.ipynb`, `docs/model_selection.md`, `models/churn_pipeline_bundle.joblib` | **Completed & Verified** |
| **Phase 6** | Model Interpretability & Business Insights | `notebooks/06_model_interpretability.ipynb`, `docs/model_interpretability.md`, `reports/figures/14-17` | **Completed & Verified** |
| **Phase 7** | Model Inference & Prediction System | `src/churn_prediction/model_inference.py`, `input_validation.py`, `prediction_utils.py`, `docs/inference_guide.md` | **Completed & Verified** |
| **Phase 8** | Backend REST API Development | `api/main.py`, `api/schemas.py`, `api/routes/`, `tests/test_api.py`, `docs/api_documentation.md` | **Completed & Verified** |
| **Phase 9** | Frontend Development & User Interface | `frontend/app.py`, `frontend/utils/api_client.py`, `docs/frontend_guide.md` | **Completed & Verified** |
| **Phase 10** | Testing & Quality Assurance | `tests/`, `reports/phase_10/test_report.md`, `docs/testing_strategy.md` (28/28 tests passed) | **Completed & Verified** |
| **Phase 11** | Containerization with Docker | `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`, `docs/docker_setup.md` | **Completed & Verified** |
| **Phase 12** | Deployment & Production Configuration | `docs/deployment_guide.md`, `docs/production_configuration.md` | **Completed & Verified** |
| **Phase 13** | Monitoring, Maintenance & Model Governance | `src/churn_prediction/drift_detection.py`, `monitoring.py`, `docs/model_monitoring.md`, `docs/model_governance.md` | **Completed & Verified** |
| **Phase 14** | Final Documentation & Portfolio Handover | `README.md`, `docs/architecture.md`, `docs/portfolio_case_study.md`, `reports/final_project_report.md` | **Completed & Verified** |

---

## 2. Final Verified Model Performance Metrics

All metrics represent performance evaluated on the **untouched test split (1,409 customer records)** using the production `HistGradientBoostingClassifier` pipeline:

| Evaluation Metric | Calibrated Threshold ($\tau = 0.30$) | Standard Threshold ($\tau = 0.50$) | Majority Dummy Baseline |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8570** | 0.8570 | 0.5000 |
| **PR-AUC (Average Precision)** | **0.6740** | 0.6740 | 0.2654 |
| **Recall (Sensitivity)** | **77.01% (288 / 374)** | 55.35% (207 / 374) | 0.00% (0 / 374) |
| **Precision** | **54.24%** | 68.32% | 0.00% |
| **F1-Score** | **0.6365** | 0.6115 | 0.0000 |
| **Accuracy** | **76.65%** | 82.26% | 73.46% |
| **True Positives (TP)** | **288** | 207 | 0 |
| **False Negatives (FN - Missed Churners)** | **86** | 167 | 374 |
| **True Negatives (TN)** | **792** | 939 | 1,035 |
| **False Positives (FP)** | **243** | 96 | 0 |

---

## 3. Production Verification & Artifact Integrity

- **Fitted Pipeline Bundle:** Persisted at `models/churn_pipeline_bundle.joblib` (contains preprocessor, model weights, calibrated threshold $\tau=0.30$, and evaluation metadata).
- **Raw Data Preservation:** Confirmed raw Excel files at `data/raw/Telco_customer_churn.xlsx` and `Data/Telco_customer_churn.xlsx` have matching SHA-256 hashes (`1BCBC0CCC9B352175216979102628E579CFBDE2C3FF57B005DE168A433122640`), verifying zero modifications to raw data.
- **Test Suite Results:** 28 passing automated tests in Pytest covering data validation, schema boundary checking, API routes, inference fallback, and integration workflows.
