"""Experiment tracking module integrating MLflow for experiment reproducibility."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import mlflow
import pandas as pd

from churn_prediction.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_MLRUNS_DIR = PROJECT_ROOT / "mlruns"
DEFAULT_DB_PATH = PROJECT_ROOT / "mlflow.db"


class MLflowTracker:
    """Interface managing MLflow experiments, parameter logging, and artifact tracking."""

    def __init__(self, experiment_name: str = "customer_churn_benchmarks", tracking_uri: Optional[Path] = None):
        import os
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

        if tracking_uri is None:
            db_path_str = str(DEFAULT_DB_PATH).replace("\\", "/")
            self.tracking_uri = f"sqlite:///{db_path_str}"
        else:
            self.tracking_uri = str(tracking_uri)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        self.active_run = None
        logger.info(f"MLflow initialized: experiment='{experiment_name}', uri='{self.tracking_uri}'")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow tracking run."""
        run_tags = tags or {}
        run_tags["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.active_run = mlflow.start_run(run_name=run_name, tags=run_tags)
        logger.info(f"Started MLflow run: '{run_name}' (ID: {self.active_run.info.run_id})")
        return self.active_run

    def end_run(self):
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameter dictionary to active run."""
        # Sanitize parameter values
        clean_params = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in params.items()}
        mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics."""
        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, file_path: Path, artifact_path: Optional[str] = None):
        """Save a file artifact (plot, table, model bundle) to MLflow run."""
        p = Path(file_path)
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path=artifact_path)
        else:
            logger.warning(f"Artifact not found at {p}")

    def log_dataset_metadata(self, dataset_path: Path, dataset_name: str = "raw_data"):
        """Compute and log SHA-256 fingerprint and row dimensions."""
        p = Path(dataset_path)
        if not p.exists():
            return
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        mlflow.log_param(f"{dataset_name}_sha256", sha)
        mlflow.log_param(f"{dataset_name}_path", str(p.name))


import numpy as np
