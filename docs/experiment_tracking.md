# Experiment Tracking with MLflow

## Overview

The Customer Churn Prediction system uses **MLflow** for structured experiment tracking, parameter logging, metric recording, and candidate benchmark comparison across training runs.

## Architecture

```
scripts/train.py
       │
       ▼
src/churn_prediction/experiment_tracking.py
       │
       ├── Backend Store: SQLite (`sqlite:///mlflow.db`)
       │     - Experiments metadata
       │     - Run parameters & tags
       │     - Numeric metric histories
       │
       └── Artifact Store: Local directory (`models/`)
             - Scikit-learn Pipeline Bundles (.joblib)
             - Run summaries (.json)
```

### Why SQLite over File Store?
In MLflow 3+, file-based tracking stores (`mlruns/`) are deprecated in favor of SQL database backends to ensure atomic transactions, thread safety, and concurrency support without requiring an external server daemon. An embedded SQLite file (`mlflow.db`) provides an ideal zero-overhead, production-grade tracking backend.

## Tracked Elements

Every training execution via `scripts/train.py` logs the following:

1. **Parameters & Metadata:**
   - Model algorithm name (`HistGradientBoostingClassifier`, `LogisticRegression`, etc.)
   - Preprocessing parameters (missing value indicators, scaler choices, imputation strategies)
   - Hyperparameter configurations (`learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`)
   - Dataset fingerprint (SHA-256 hash of raw input data: `1BCBC0CCC9B352175216979102628E579CFBDE2C3FF57B005DE168A433122640`)
   - Decision threshold (calibrated $\tau = 0.30$)

2. **Metrics:**
   - 5-Fold Stratified Cross-Validation ROC-AUC for all candidate models:
     - Dummy Classifier
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - HistGradientBoosting
     - XGBoost
   - Test Set Holdout Metrics:
     - ROC-AUC ($0.8570$)
     - PR-AUC ($0.6740$)
     - Recall ($77.01\%$)
     - Precision ($54.24\%$)
     - F1-Score ($0.6365$)
     - Accuracy ($76.65\%$)
     - Confusion Matrix elements ($TP=288, FP=243, TN=792, FN=86$)
   - Training duration in seconds.

3. **Artifacts:**
   - Full pipeline bundle (`models/churn_pipeline_bundle.joblib`)
   - Training configuration snapshot (`configs/train_config.json`)
   - Summary JSON report (`reports/phase_15/training_run_summary.json`)

## Usage

### Running Training & Tracking
```bash
python scripts/train.py --config configs/train_config.json
```

### Launching MLflow UI
To explore and compare runs visually:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Then navigate to `http://localhost:5000`.
