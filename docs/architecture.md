# End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data & Preprocessing Layer"]
        RawData["Raw Dataset (data/raw/Telco_customer_churn.xlsx)"]
        Cleaner["Data Cleaning & Type Remediation (Total Charges whitespace -> 0.0)"]
        Splitter["Stratified Train/Test Split (80:20, Random State 42)"]
        ProcessedTrain["Training Split (5,634 rows)"]
        ProcessedTest["Untouched Test Split (1,409 rows)"]
        
        RawData --> Cleaner
        Cleaner --> Splitter
        Splitter --> ProcessedTrain
        Splitter --> ProcessedTest
    end

    subgraph Pipeline_Layer ["2. Machine Learning Pipeline Layer"]
        Preprocessor["ColumnTransformer (SimpleImputer + StandardScaler + OneHotEncoder)"]
        Classifier["HistGradientBoostingClassifier (Tuned: 100 trees, lr=0.03, leaves=15)"]
        Pipeline["Fitted Pipeline (models/churn_pipeline_bundle.joblib)"]
        CalibratedThresh["Calibrated Decision Threshold (τ = 0.30)"]
        
        ProcessedTrain --> Preprocessor
        Preprocessor --> Classifier
        Classifier --> Pipeline
        Pipeline --> CalibratedThresh
    end

    subgraph Serving_Layer ["3. Production Serving & API Layer"]
        Engine["Inference Engine (src/churn_prediction/model_inference.py)"]
        FastAPI["FastAPI REST API (api/main.py)"]
        Endpoints["Endpoints: /health, /model-info, /predict, /predict/batch"]
        
        Pipeline --> Engine
        CalibratedThresh --> Engine
        Engine --> FastAPI
        FastAPI --> Endpoints
    end

    subgraph UI_Layer ["4. Application & Interface Layer"]
        Streamlit["Streamlit Dashboard (frontend/app.py)"]
        HTTPClient["API Client with Fallback (frontend/utils/api_client.py)"]
        User["Business Analysts, Retention Agents, Stakeholders"]
        
        Endpoints <--> HTTPClient
        HTTPClient <--> Streamlit
        Streamlit <--> User
    end

    subgraph Governance_Layer ["5. Monitoring & Governance Layer"]
        DriftDetector["Data Drift Audit (PSI & Kolmogorov-Smirnov)"]
        TelemetryLogger["Non-PII Prediction Telemetry (reports/monitoring_events.jsonl)"]
        SHAP["Model Interpretability (Global & Local Explanations)"]
        
        ProcessedTest --> DriftDetector
        Engine --> TelemetryLogger
        Pipeline --> SHAP
    end
```

---

## Architecture Flow Breakdown

1. **Data Ingestion & Integrity:**
   - Raw data is loaded and validated for primary keys and target distributions.
   - Whitespace anomalies in `Total Charges` are resolved without record deletion.
   - Stratified train/test split isolates 20% of data for final evaluation.
2. **Leakage-Free Transformation Pipeline:**
   - Preprocessing steps (`StandardScaler`, `OneHotEncoder`) are encapsulated inside a Scikit-Learn `ColumnTransformer`.
   - Transformations are fitted exclusively on training folds during 5-fold cross-validation.
3. **Optimized Inference Engine:**
   - Persisted as a self-contained Joblib bundle combining pipeline weights, threshold (`0.30`), and metadata.
   - Fast sub-15ms CPU inference.
4. **Decoupled REST API & Frontend:**
   - FastAPI handles Pydantic schema validation, batching, and health probes.
   - Streamlit provides an intuitive interface for frontline retention agents.
5. **Observability & Drift Monitoring:**
   - Telemetry tracks serving latency, risk category distributions, and population stability indices.
