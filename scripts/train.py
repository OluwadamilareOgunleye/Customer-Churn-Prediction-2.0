"""Reproducible training and experiment tracking entrypoint."""

import json
import logging
from pathlib import Path
import time
import pandas as pd

from churn_prediction.config import (
    ALL_FEATURE_COLS,
    FIGURES_DIR,
    MODELS_DIR,
    PHASE_15_REPORTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
    TARGET_COL,
)
from churn_prediction.evaluation import evaluate_predictions, sweep_thresholds
from churn_prediction.experiment_tracking import MLflowTracker
from churn_prediction.model_inference import save_model_bundle
from churn_prediction.model_registry import ModelRegistry
from churn_prediction.model_training import cross_validate_models, tune_best_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    PHASE_15_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = Path("configs/train_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Initialize tracker and registry
    tracker = MLflowTracker(experiment_name=config.get("experiment_name", "customer_churn_benchmarks"))
    registry = ModelRegistry(Path(config["registry"]["registry_store"]))

    # 2. Ingest train and test data
    train_df = pd.read_csv(Path(config["data"]["processed_dir"]) / "train.csv")
    test_df = pd.read_csv(Path(config["data"]["processed_dir"]) / "test.csv")

    X_train = train_df[ALL_FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[ALL_FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    start_time = time.time()
    run_name = f"train_{config['tuning']['target_model']}_{int(start_time)}"
    active_run = tracker.start_run(run_name=run_name)
    run_id = active_run.info.run_id

    try:
        # Log metadata & dataset fingerprint
        tracker.log_dataset_metadata(RAW_DATA_PATH, "raw_dataset")
        tracker.log_params({
            "test_size": config["data"]["test_size"],
            "random_state": config["data"]["random_state"],
            "cv_folds": config["training"]["cv_folds"],
            "model_architecture": config["tuning"]["target_model"],
        })

        # 3. Model Benchmark
        logger.info("Executing cross-validation benchmark across candidate models...")
        cv_summary = cross_validate_models(
            X_train, y_train, cv_folds=config["training"]["cv_folds"], random_state=config["data"]["random_state"]
        )

        import re
        for _, row in cv_summary.iterrows():
            clean_m = re.sub(r"[^a-zA-Z0-9_\-]", "_", row["Model"]).strip("_").lower()
            tracker.log_metrics({
                f"cv_roc_auc_{clean_m}": row["ROC-AUC (Mean)"],
                f"cv_recall_{clean_m}": row["Recall (Mean)"],
                f"cv_f1_{clean_m}": row["F1-Score (Mean)"],
            })

        # 4. Hyperparameter Tuning
        logger.info(f"Tuning {config['tuning']['target_model']}...")
        best_pipeline, best_params = tune_best_model(
            X_train,
            y_train,
            model_type=config["tuning"]["target_model"],
            cv_folds=config["training"]["cv_folds"],
            random_state=config["data"]["random_state"],
        )
        tracker.log_params(best_params)

        # 5. Fit full pipeline & threshold optimization
        best_pipeline.fit(X_train, y_train)

        from sklearn.model_selection import cross_val_predict
        cv_probs = cross_val_predict(best_pipeline, X_train, y_train, cv=5, method="predict_proba")[:, 1]
        sweep_df = sweep_thresholds(y_train.values, cv_probs)
        best_f1_idx = sweep_df["f1"].idxmax()
        calibrated_threshold = float(sweep_df.loc[best_f1_idx, "threshold"])

        tracker.log_params({"calibrated_threshold": calibrated_threshold})

        # 6. Evaluation on test set
        test_probs = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics_cal = evaluate_predictions(y_test.values, test_probs, threshold=calibrated_threshold)
        test_metrics_std = evaluate_predictions(y_test.values, test_probs, threshold=0.50)

        tracker.log_metrics({
            "test_roc_auc": test_metrics_cal["roc_auc"],
            "test_pr_auc": test_metrics_cal["pr_auc"],
            "test_recall_calibrated": test_metrics_cal["recall"],
            "test_precision_calibrated": test_metrics_cal["precision"],
            "test_f1_calibrated": test_metrics_cal["f1"],
            "test_accuracy_calibrated": test_metrics_cal["accuracy"],
            "test_recall_standard": test_metrics_std["recall"],
            "duration_seconds": time.time() - start_time,
        })

        # 7. Persist and Register Model
        bundle_path = MODELS_DIR / "churn_pipeline_bundle.joblib"
        save_model_bundle(
            pipeline=best_pipeline,
            threshold=calibrated_threshold,
            metrics={"calibrated_threshold": test_metrics_cal, "standard_threshold": test_metrics_std},
            metadata={"run_id": run_id, "best_params": best_params},
            save_path=bundle_path,
        )

        tracker.log_artifact(bundle_path, artifact_path="model")

        # Register in model registry
        registered_meta = registry.register_model(
            model_name=config["registry"]["model_name"],
            artifact_path=bundle_path,
            model_type=config["tuning"]["target_model"],
            decision_threshold=calibrated_threshold,
            metrics=test_metrics_cal,
            stage="Production",  # Baseline approved production release
            run_id=run_id,
            notes=f"Automated training run. Best params: {best_params}",
        )

        # 8. Save summary
        run_summary = {
            "run_id": run_id,
            "duration_seconds": float(round(time.time() - start_time, 2)),
            "model_version": registered_meta.version,
            "status": registered_meta.status,
            "metrics": test_metrics_cal,
            "best_hyperparameters": best_params,
        }
        with open(PHASE_15_REPORTS_DIR / "training_run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        print("\n=== Training & Tracking Completed Successfully ===")
        print(f"MLflow Run ID: {run_id}")
        print(f"Registered Model: '{registered_meta.model_name}' Version {registered_meta.version} [{registered_meta.status}]")
        print(f"Test Set ROC-AUC: {test_metrics_cal['roc_auc']:.4f} | Recall: {test_metrics_cal['recall']:.4f}")

    finally:
        tracker.end_run()


if __name__ == "__main__":
    main()
