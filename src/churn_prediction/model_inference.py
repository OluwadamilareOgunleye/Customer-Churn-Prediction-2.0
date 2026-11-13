"""Model persistence, loading, and inference serving interface."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_prediction.config import ALL_FEATURE_COLS, MODELS_DIR
from churn_prediction.input_validation import (
    ValidationError,
    validate_customer_record,
    validate_input_dataframe,
)
from churn_prediction.prediction_utils import (
    assign_risk_category,
    format_prediction_result,
)

logger = logging.getLogger(__name__)


def save_model_bundle(
    pipeline: Pipeline,
    threshold: float,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
) -> Path:
    """Serialize full pipeline, optimal classification threshold, and evaluation metadata."""
    path = Path(save_path) if save_path else MODELS_DIR / "churn_pipeline_bundle.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "metrics": metrics,
        "feature_names": ALL_FEATURE_COLS,
        "metadata": metadata or {},
    }

    joblib.dump(bundle, path)
    logger.info(f"Model bundle successfully saved to {path}")
    return path


def load_model_bundle(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load model bundle from disk."""
    path = Path(model_path) if model_path else MODELS_DIR / "churn_pipeline_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}")

    bundle = joblib.load(path)
    return bundle


class ChurnPredictor:
    """Production inference engine for Customer Churn scoring with full schema validation."""

    def __init__(self, bundle_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None):
        if bundle_or_path is None:
            bundle_or_path = MODELS_DIR / "churn_pipeline_bundle.joblib"

        if isinstance(bundle_or_path, (str, Path)):
            bundle = load_model_bundle(Path(bundle_or_path))
        elif isinstance(bundle_or_path, dict):
            bundle = bundle_or_path
        else:
            raise TypeError("Expected file path or bundle dictionary.")

        self.pipeline: Pipeline = bundle["pipeline"]
        self.default_threshold: float = float(bundle.get("threshold", 0.30))
        self.feature_names: List[str] = bundle.get("feature_names", ALL_FEATURE_COLS)
        self.metrics: Dict[str, Any] = bundle.get("metrics", {})
        self.metadata: Dict[str, Any] = bundle.get("metadata", {})

    def predict_single(
        self,
        record: Union[Dict[str, Any], pd.Series],
        threshold: Optional[float] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Validate and score a single customer record.
        
        Parameters
        ----------
        record : dict or pd.Series
            Customer record dictionary.
        threshold : float, optional
            Custom decision threshold. Defaults to calibrated threshold (0.30).
        strict : bool, default=False
            Strict validation mode.
            
        Returns
        -------
        dict
            Formatted prediction result dictionary.
        """
        clean_rec, warnings = validate_customer_record(record, strict=strict)
        cust_id = clean_rec.get("CustomerID")
        
        # Prepare 1-row DataFrame
        df_row = pd.DataFrame([clean_rec])[self.feature_names]
        prob = float(self.pipeline.predict_proba(df_row)[:, 1][0])
        thresh = threshold if threshold is not None else self.default_threshold

        return format_prediction_result(
            customer_id=cust_id,
            probability=prob,
            threshold=thresh,
            warnings=warnings,
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        strict: bool = False,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Validate and score a batch of customer records.
        
        Parameters
        ----------
        df : pd.DataFrame
            Customer records DataFrame.
        threshold : float, optional
            Decision threshold.
        strict : bool, default=False
            Strict validation mode.
            
        Returns
        -------
        Tuple[pd.DataFrame, List[dict]]
            Enriched DataFrame and list of row validation warnings.
        """
        valid_df, warnings = validate_input_dataframe(df, strict=strict)
        thresh = threshold if threshold is not None else self.default_threshold

        probs = self.pipeline.predict_proba(valid_df[self.feature_names])[:, 1]
        preds = (probs >= thresh).astype(int)

        out_df = df.copy()
        out_df["churn_probability"] = np.round(probs, 4)
        out_df["churn_prediction"] = preds
        out_df["churn_label"] = np.where(preds == 1, "Yes", "No")
        out_df["risk_category"] = [assign_risk_category(p) for p in probs]
        out_df["risk_tier"] = out_df["risk_category"]

        return out_df, warnings

    def score_records(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """Convenience method returning scored records DataFrame."""
        out_df, _ = self.predict_batch(df, threshold=threshold, strict=False)
        return out_df

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Direct probability scoring for raw validated DataFrame."""
        valid_df, _ = validate_input_dataframe(df, strict=False)
        return self.pipeline.predict_proba(valid_df[self.feature_names])[:, 1]

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Direct binary predictions for raw validated DataFrame."""
        thresh = threshold if threshold is not None else self.default_threshold
        probs = self.predict_proba(df)
        return (probs >= thresh).astype(int)
