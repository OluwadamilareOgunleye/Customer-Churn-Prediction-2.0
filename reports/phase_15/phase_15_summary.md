# Phase 15 Summary: MLOps Automation and Experiment Tracking

## Objectives Achieved
1. **Experiment Tracking Integration**: Implemented `src/churn_prediction/experiment_tracking.py` leveraging MLflow with an embedded SQLite tracking store (`sqlite:///mlflow.db`) and local artifact store.
2. **Reproducible Training Pipeline**: Created `scripts/train.py` taking structured configuration (`configs/train_config.json`) to automate data validation, cross-validation benchmarking, hyperparameter search, holdout evaluation, artifact bundling, and MLflow logging.
3. **Model Registry Implementation**: Built `src/churn_prediction/model_registry.py` managing versioned models (`models/registry.json`) with enforced lifecycle states (`Candidate`, `Staging`, `Production`, `Archived`).
4. **Candidate Comparison**: 5-fold cross-validation logged for baseline (Dummy), Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, and XGBoost.
5. **Verified Verification Run**:
   - Run ID: `94aa55661d42478690ce1fb2c47a76ec`
   - Test ROC-AUC: `0.8570`
   - Test Recall: `0.7701` (at $\tau = 0.30$)
   - Best Hyperparameters: `{'learning_rate': 0.03, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 50}`
   - Registered as: `telco_churn_gradient_booster` Version 1 (`Production`).
