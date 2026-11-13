# Model Registry

## Overview

The model registry manages machine learning model lifecycles, ensuring traceable versioning, lineage tracking, and strict release promotion without accidental overwrites.

## Design Philosophy

In production ML systems, training a new model must **never automatically replace** an active production model. Replacement requires:
1. Strict regression testing against holdout validation benchmarks.
2. Calibration and bias checks.
3. Formal transition through lifecycle stages: `Staging` -> `Production` -> `Archived`.

## Registry Structure

The local registry is maintained at `models/registry.json` using the `ModelRegistry` manager (`src/churn_prediction/model_registry.py`).

### Schema Definition
Each model version entry contains:
- `model_name`: Logical identifier (e.g., `telco_churn_gradient_booster`).
- `version`: Monotonically increasing integer ($1, 2, 3 \dots$).
- `model_type`: Underlaying estimator (e.g., `HistGradientBoosting`).
- `status`: Lifecycle stage (`Candidate`, `Staging`, `Production`, `Archived`).
- `created_at`: UTC timestamp of registration.
- `artifact_path`: Absolute path to serialized `.joblib` bundle.
- `decision_threshold`: Calibrated operating threshold ($\tau = 0.30$).
- `metrics`: Dictionary of evaluated performance metrics on holdout test set.
- `run_id`: Associated MLflow run ID linking to the exact training experiment.
- `notes`: Metadata regarding hyperparameter tuning and model lineage.

## Python API

```python
from churn_prediction.model_registry import ModelRegistry

registry = ModelRegistry()

# List all versions
versions = registry.get_model_versions("telco_churn_gradient_booster")

# Retrieve active production model
prod_model = registry.get_production_model("telco_churn_gradient_booster")
print(f"Active Production Model: Version {prod_model['version']}, ROC-AUC: {prod_model['metrics']['roc_auc']:.4f}")

# Promote a candidate after verification
registry.promote_model(
    model_name="telco_churn_gradient_booster",
    version=2,
    target_status="Production"
)
```

## Governance Rule
Only one model version can hold the `Production` status per model family at any given time. When a new version is promoted to `Production`, the previous production model is automatically transitioned to `Archived` to preserve rollback capability.
