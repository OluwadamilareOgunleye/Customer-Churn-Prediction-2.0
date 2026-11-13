# Telco Customer Churn Prediction & Retention Optimization System

An industry-grade, reproducible, end-to-end Machine Learning Engineering & MLOps system developed to predict customer churn in the telecommunications sector and optimize customer retention strategies.

Built following production MLOps standards, featuring leakage-free Scikit-Learn pipelines, stratified validation, systematic hyperparameter optimization, business-driven probability threshold calibration, model interpretability with SHAP, MLflow experiment tracking, versioned model registry, a FastAPI REST microservice, a Streamlit frontend dashboard, K-Means customer segmentation with retention ROI modeling, Docker containerization, CI/CD automation, cloud architecture with Terraform, and population drift monitoring.

---

## 1. Verified Model Performance (Untouched Test Set: 1,409 Records)

| Metric | Majority Baseline (Dummy) | Standard Threshold ($\tau = 0.50$) | Tuned Production Model ($\tau = 0.30$) |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.5000 | 0.8570 | **0.8570** |
| **PR-AUC (Avg. Precision)** | 0.2654 | 0.6740 | **0.6740** |
| **Recall (Sensitivity)** | 0.00% | 52.67% (197 / 374) | **77.01% (288 / 374 captured)** |
| **Precision** | 0.00% | 67.24% | **54.24%** |
| **F1-Score** | 0.0000 | 0.5907 | **0.6365** |
| **Accuracy** | 73.46% | 80.62% | **76.65%** |
| **Missed Churners (FN)** | 374 | 177 | **86 (-51.4% reduction)** |
| **Brier Calibration Score**| 0.2654 | 0.1322 | **0.1322 (Near-Optimal)** |

> **Business Justification for Calibrated Threshold ($\tau = 0.30$):** In subscription telecommunications, losing an existing subscriber is estimated to cost 5–7x more than an unnecessary retention contact. Calibrating the threshold to `0.30` captures **91 additional churners** (reducing missed churners from 177 down to 86) while maintaining an intervention precision above 54%, generating a **+\$15,700 net profit advantage** in campaign ROI over default classification.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion & Preprocessing
        RawData[IBM Telco Dataset .xlsx] --> Clean[Data Cleaning & Missing Value Handler]
        Clean --> FeatureEng[Feature Selection & Stratified Split 80/20]
        FeatureEng --> CleanTrain[train.csv]
        FeatureEng --> CleanTest[test.csv]
    end

    subgraph Training, Tracking & Governance
        CleanTrain --> SklearnPipeline[Scikit-learn Pipeline:<br/>Imputer + OneHotEncoder/StandardScaler + HistGradientBoosting]
        SklearnPipeline --> MLflow[MLflow Tracking: SQLite Backend mlflow.db]
        SklearnPipeline --> GridSearch[GridSearchCV 5-Fold Stratified CV]
        GridSearch --> CalibCheck[Probability Calibration & Subgroup Audit]
        CalibCheck --> ModelBundle[models/churn_pipeline_bundle.joblib]
        ModelBundle --> Registry[(Model Registry: models/registry.json)]
    end

    subgraph Serving & Microservices
        Registry --> ServingEngine[ChurnPredictor & Artifact Storage Abstraction]
        ServingEngine --> FastAPI[FastAPI REST API :8000<br/>/predict, /predict/batch, /health]
        FastAPI --> Streamlit[Streamlit Interactive UI :8501]
    end

    subgraph Business Intelligence & Monitoring
        FastAPI --> Telemetry[Non-PII Request & Drift Logger]
        Telemetry --> Monitoring[PSI & KS Distribution Shift Audit]
        ServingEngine --> BI[K-Means Customer Segmentation & ROI Simulator]
        BI --> Streamlit
    end

    subgraph CI/CD & Cloud Infrastructure
        GitHub[GitHub Actions Automation] --> CI[Lint, Pytest, Coverage]
        GitHub --> DockerBuild[Docker Multi-Container Validation]
        GitHub --> Retrain[Controlled Retraining Quality Gate]
        Terraform[infrastructure/main.tf] --> AWS[AWS ECS Fargate & S3 IaC Specification]
    end
```

---

## 3. Project Directory Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   │   └── Telco_customer_churn.xlsx       # Untouched original raw IBM dataset (SHA-256 verified)
│   ├── interim/
│   │   └── raw_cache.csv                   # Cached high-speed CSV for rapid iteration
│   └── processed/
│       ├── train.csv                       # Cleaned, stratified training split (5,634 rows)
│       └── test.csv                        # Cleaned, stratified untouched test split (1,409 rows)
│
├── configs/
│   └── train_config.json                   # Decoupled hyperparameters & training configuration
│
├── infrastructure/
│   └── main.tf                             # Terraform configuration for AWS ECS, ECR, S3, CloudWatch
│
├── notebooks/
│   ├── 03_exploratory_data_analysis.ipynb          # Phase 3 EDA notebook
│   ├── 04_data_cleaning_preprocessing.ipynb        # Phase 4 Preprocessing notebook
│   ├── 05_model_development_evaluation.ipynb      # Phase 5 Modeling & Tuning notebook
│   ├── 06_model_interpretability.ipynb            # Phase 6 SHAP Interpretability notebook
│   ├── 17_model_validation_robustness.ipynb       # Phase 17 Calibration & Robustness notebook
│   └── 19_customer_segmentation.ipynb             # Phase 19 K-Means & ROI simulation notebook
│
├── src/
│   └── churn_prediction/
│       ├── __init__.py                     # Package metadata
│       ├── config.py                       # Global paths, constants, and feature definitions
│       ├── data_loader.py                  # Ingestion and caching utilities
│       ├── data_cleaning.py                # Type coercion and whitespace sanitization
│       ├── data_validation.py              # Pandera schema integrity and leakage assertions
│       ├── input_validation.py             # Runtime serving boundary checking
│       ├── preprocessing.py                # Scikit-learn ColumnTransformer & stratified splitting
│       ├── model_training.py               # Cross-validation benchmarking & tuning
│       ├── evaluation.py                   # Classification metrics & threshold curves
│       ├── experiment_tracking.py          # MLflow SQLite-backed tracking manager
│       ├── model_registry.py               # Versioned local model registry
│       ├── artifact_storage.py             # S3 & Local artifact store abstraction
│       ├── calibration.py                  # Probability calibration & threshold sweeps
│       ├── robustness.py                   # Subgroup performance & stress testing
│       ├── customer_segmentation.py        # K-Means clustering & persona profiling
│       ├── business_impact.py              # Campaign ROI and financial simulation
│       ├── interpretability.py             # Global and local SHAP explanations
│       ├── model_inference.py              # Production ChurnPredictor serving engine
│       ├── prediction_utils.py             # Output formatting & risk categorization
│       ├── drift_detection.py              # Population Stability Index (PSI) & KS test
│       └── monitoring.py                   # Non-PII telemetry and event logger
│
├── api/
│   ├── main.py                             # FastAPI application entrypoint with lifespan
│   ├── config.py                           # API settings and CORS configuration
│   ├── schemas.py                          # Pydantic request/response schemas
│   ├── services/
│   │   └── prediction_service.py           # Model singleton manager
│   └── routes/
│       ├── health.py                       # GET /health probe
│       ├── info.py                         # GET /model-info metadata route
│       └── prediction.py                   # POST /predict and /predict/batch endpoints
│
├── frontend/
│   ├── app.py                              # Streamlit application with Persona & ROI tabs
│   └── utils/
│       └── api_client.py                   # HTTP API client with local engine fallback
│
├── scripts/
│   ├── train.py                            # Reproducible training & MLflow tracking script
│   ├── run_phase17_validation.py           # Model calibration & stress testing runner
│   └── run_phase19_segmentation.py         # K-Means segmentation & ROI runner
│
├── .github/workflows/
│   ├── ci.yml                              # GitHub Actions: Linting, Pytest, Coverage
│   ├── docker-build.yml                    # GitHub Actions: Docker container build checks
│   ├── model-training.yml                  # GitHub Actions: Controlled model retraining gate
│   └── deployment.yml                      # GitHub Actions: Environment release workflow
│
├── models/
│   ├── churn_pipeline_bundle.joblib        # Fitted pipeline + metadata
│   └── registry.json                       # Versioned model registry ledger
│
├── presentations/
│   └── customer_churn_project.pptx         # 15-slide executive presentation deck
│
├── reports/
│   ├── figures/                            # 24 high-resolution publication charts
│   ├── phase_15/                           # Training run summary & MLflow artifacts
│   ├── phase_16/                           # CI/CD implementation report
│   ├── phase_17/                           # Calibration & subgroup metrics
│   ├── phase_18/                           # Cloud architecture & scalability report
│   ├── phase_19/                           # Persona profiles & ROI simulations
│   ├── phase_20/                           # Final MLOps integration report
│   ├── phase_21/                           # Security & dependency audit report
│   ├── phase_22/                           # Performance benchmark report
│   ├── phase_23/                           # System reliability report
│   ├── phase_24/                           # Portfolio demonstration report
│   ├── final_technical_audit.md            # Independent technical audit report
│   ├── final_project_scorecard.md          # 11-dimension project scorecard (9.27/10)
│   ├── technical_debt_register.md          # Prioritized technical debt register
│   └── final_project_report.md             # Executive completion report
│
├── docs/
│   ├── problem_definition.md               # Business context & formulation
│   ├── dataset_documentation.md            # Data dictionary & categorization
│   ├── eda_findings.md                     # Exploratory findings
│   ├── preprocessing_decisions.md          # Pipeline architecture & leakage pruning
│   ├── model_selection.md                  # Benchmark comparison & tuning
│   ├── model_interpretability.md           # SHAP attribution analysis
│   ├── experiment_tracking.md              # MLflow architecture & tracking guide
│   ├── model_registry.md                   # Model lifecycle & governance rules
│   ├── ci_cd.md                            # GitHub Actions workflows & setup
│   ├── model_validation.md                 # Calibration, subgroups & stress tests
│   ├── cloud_architecture.md               # Cloud systems design & Mermaid diagrams
│   ├── cloud_cost_estimate.md              # Multi-cloud pricing comparisons
│   ├── customer_segmentation.md            # K-Means personas & risk tiers
│   ├── business_impact_analysis.md         # Retention ROI financial analysis
│   ├── security_assessment.md              # Threat modeling & security controls
│   ├── data_privacy.md                     # Data minimization & PII masking
│   ├── performance_analysis.md             # Latency, throughput & scaling benchmarks
│   ├── reliability_strategy.md             # Dual-probe health & client fallback
│   ├── recovery_procedures.md              # Disaster recovery & instant model rollback
│   ├── technical_case_study.md             # In-depth technical portfolio case study
│   ├── interview_preparation.md            # Technical interview Q&A guide
│   ├── project_demonstration_guide.md      # Live demo script & presentation guide
│   ├── final_release_checklist.md          # Final verification checklist
│   └── final_architecture.md               # 10-stage lifecycle documentation
│
├── tests/
│   ├── test_data_validation.py             # Schema, cleaning, and leakage unit tests
│   ├── test_model_inference.py             # Model loading, thresholding & inference tests
│   ├── test_api.py                         # FastAPI endpoints & Pydantic validation tests
│   ├── test_integration_e2e.py            # End-to-end lifecycle & client fallback tests
│   ├── test_advanced_features.py           # Registry, storage, calibration & ROI tests
│   ├── test_reliability.py                 # Health probes, liveness/readiness & rollback tests
│   └── performance/                        # Benchmark & Locust load test scripts
│
├── Dockerfile.api                          # Multi-stage Dockerfile for FastAPI backend
├── Dockerfile.frontend                     # Optimized Dockerfile for Streamlit frontend
├── docker-compose.yml                      # Multi-container orchestration
├── .dockerignore                           # Docker build exclusion rules
├── pyproject.toml                          # Project configuration & test settings
├── requirements.txt                        # Pinned dependencies
├── .env.example                            # Environment configuration template
└── README.md                               # Project documentation
```

---

## 4. Key Business Insights & Strategic Levers

1. **The 12-Month Hazard Peak:** Over 50% of churn occurs in months 1–12 (median tenure of churners is 10 months vs 38 months for loyal customers).
2. **Contract Lock-in Dynamics:** Month-to-month contracts exhibit a **42.56% churn rate**, compared to **12.00%** for 1-year and **2.68%** for 2-year contracts.
3. **The Fiber Optic Support Shield:** Fiber optic subscribers churn at **41.11%** despite paying premium fees ($70–$100). Adding Tech Support cuts churn by more than half, acting as a crucial retention anchor.
4. **Frictionless Payments:** Electronic check payers churn at **43.46%**, while automated bank transfer payers churn at only **19.00%**.
5. **Retention Financial ROI:** Operating at $\tau = 0.30$ captures 288 churners on the test set, delivering a **+\$15,700 net profit advantage** in targeted campaigns compared to standard 0.50 classification.

---

## 5. Measured Performance Benchmarks

All performance metrics are directly measured on actual project hardware:
- **Model Load Time:** $0.745$ seconds
- **Single Inference Latency (p50):** $36.43$ ms
- **Single Inference Latency (p95):** $45.68$ ms
- **Full Batch (1,409 records) Latency:** $1.115$ seconds ($0.792$ ms/record)
- **Automated Test Suite:** **39 passed / 39 total (100% pass rate)** in $20.8$ seconds.
- **Population Stability Index (PSI):** Maximum feature PSI on test set is **$0.0124$** ($\ll 0.10$ stability threshold).

---

## 6. Quickstart & Reproducibility

### A. Local Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package and dependencies in editable mode
pip install --upgrade pip
pip install -e .
```

### B. Run Automated Test Suite
```powershell
pytest -v
```
*Result: 39 passed across unit, API, integration, advanced features, and reliability suites.*

### C. Run Reproducible Model Training & Tracking
```powershell
python scripts/train.py --config configs/train_config.json
```
Logs cross-validation metrics and artifacts to local SQLite MLflow store (`mlflow.db`) and registers the model in `models/registry.json`.

### D. Run FastAPI REST Backend
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation opens at: `http://localhost:8000/docs`.

### E. Run Streamlit Frontend Dashboard
```powershell
streamlit run frontend/app.py --server.port 8501
```
Web dashboard opens at: `http://localhost:8501`.

### F. Run Multi-Container Stack via Docker Compose
```bash
docker compose up --build -d
```
- Streamlit UI: `http://localhost:8501`
- FastAPI Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
