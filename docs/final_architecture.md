# Final End-to-End MLOps System Architecture

## 1. Complete End-to-End System Flow

```mermaid
flowchart TD
    subgraph 1. Ingestion & Preprocessing
        RawData[IBM Telco Dataset .xlsx] --> Clean[Data Cleaning & Missing Value Handler]
        Clean --> FeatureEng[Feature Selection & Stratified Split 80/20]
        FeatureEng --> CleanTrain[train.csv]
        FeatureEng --> CleanTest[test.csv]
    end

    subgraph 2. Training, Tracking & Governance
        CleanTrain --> SklearnPipeline[Scikit-learn Pipeline:<br/>Median/Mode Imputer + StandardScaler/OHE + HistGradientBoost]
        SklearnPipeline --> MLflow[MLflow Tracking: SQLite Backend mlflow.db]
        SklearnPipeline --> GridSearch[GridSearchCV 5-Fold Cross Validation]
        GridSearch --> CalibCheck[Probability Calibration & Subgroup Audit]
        CalibCheck --> ModelBundle[models/churn_pipeline_bundle.joblib]
        ModelBundle --> Registry[(Model Registry: models/registry.json)]
    end

    subgraph 3. Serving & Microservices
        Registry --> ServingEngine[ChurnPredictor & Artifact Storage Abstraction]
        ServingEngine --> FastAPI[FastAPI REST API :8000<br/>/predict, /predict/batch, /health]
        FastAPI --> Streamlit[Streamlit Interactive UI :8501]
    end

    subgraph 4. Business Intelligence & Monitoring
        FastAPI --> Telemetry[Non-PII Request & Drift Logger]
        Telemetry --> Monitoring[PSI & KS Distribution Shift Audit]
        ServingEngine --> BI[K-Means Customer Segmentation & ROI Simulator]
        BI --> Streamlit
    end

    subgraph 5. CI/CD & Operations
        GitHub[GitHub Actions Automation] --> CI[Lint, Pytest, Coverage]
        GitHub --> DockerBuild[Docker Multi-Container Validation]
        GitHub --> Retrain[Controlled Retraining Quality Gate]
    end
```

---

## 2. The 10-Stage Machine Learning Lifecycle

1. **Dataset Ingestion:** SHA-256 verified raw data ingestion (`data/raw/Telco_customer_churn.xlsx`).
2. **Data Validation:** Strict Pandera schema checks verifying ranges, null boundaries, and column integrity.
3. **Reproducible Preprocessing:** Explicit exclusion of post-churn target leakage columns (`Churn Reason`, `Churn Score`, `CLTV`).
4. **Experiment Tracking:** MLflow tracking store recording candidate algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, HistGradientBoosting) with parameter and metric histories.
5. **Model Evaluation & Calibration:** Verified holdout ROC-AUC ($0.8570$) and Brier calibration score ($0.1322$).
6. **Decision Threshold Calibration:** Calibrated operating threshold ($\tau = 0.30$) reducing missed churners by $51.4\%$ with $77.01\%$ recall.
7. **Model Registry & Governance:** Immutable versioned metadata in `models/registry.json` requiring human promotion gates.
8. **Stateless API Serving:** Production FastAPI service with Pydantic request validation and sub-40ms single prediction latency.
9. **Interactive Business Application:** Streamlit application with single/batch inferencing, SHAP explanations, K-Means customer profiling, and interactive ROI simulator.
10. **Telemetry & Drift Monitoring:** Real-time logging of prediction events and automated calculation of Population Stability Index (PSI).
