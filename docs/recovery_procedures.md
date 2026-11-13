# Operational Disaster Recovery & Model Rollback Procedures

## 1. Model Rollback Procedure

If a newly deployed model candidate exhibits unexpected performance regression, extreme feature drift, or biased subgroup accuracy, initiate the following verified rollback sequence:

### Step 1: Query Model Registry History
```python
from churn_prediction.model_registry import ModelRegistry

registry = ModelRegistry()
versions = registry.list_versions("telco_churn_gradient_booster")
for v in versions:
    print(f"Version {v.version}: Status={v.status}, ROC-AUC={v.metrics.get('roc_auc')}")
```

### Step 2: Instant Stage Demotion and Incumbent Promotion
Execute rollback in Python or via management CLI:
```python
# Demote flawed candidate (e.g. Version 2) and reinstate Version 1
registry.transition_stage(
    model_name="telco_churn_gradient_booster",
    version=1,
    new_stage="Production"
)
```
*Note: The ModelRegistry automatically transitions the flawed Version 2 model to `Archived`.*

### Step 3: Trigger Service Reload
Restart the API service or trigger container deployment:
```bash
docker compose restart api
```

---

## 2. Recovery from Corrupted Model Bundles

- **Symptom:** API returns `HTTP 503` on `/ready` with log entry `"Corrupted model bundle"`.
- **Root Cause:** Incomplete file transfer, interrupted serialization, or disk corruption.
- **Recovery Action:**
  1. Retrieve the backup bundle from local cache or cloud object storage:
     ```powershell
     python scripts/train.py --config configs/train_config.json
     ```
  2. Verify bundle integrity via test script:
     ```powershell
     python -c "import joblib; b = joblib.load('models/churn_pipeline_bundle.joblib'); print('Bundle Valid:', 'pipeline' in b)"
     ```
  3. Validate readiness probe:
     ```powershell
     curl -i http://localhost:8000/ready
     ```

---

## 3. Data Backup & Local Registry Preservation

- **Artifacts Backed Up:**
  - `models/registry.json`: JSON ledger of all registered model iterations.
  - `models/churn_pipeline_bundle.joblib`: Serialized Scikit-learn Pipeline with tuned hyperparameters.
  - `data/processed/`: Deterministically reproducible training and evaluation CSVs.
- **Restoration Guarantee:**
  - Because `scripts/train.py` is configured with deterministic random seeds (`RANDOM_STATE=42`) and explicit dataset SHA-256 checks, the entire model bundle and evaluation benchmarks can be reconstructed from scratch in $<45$ seconds.
