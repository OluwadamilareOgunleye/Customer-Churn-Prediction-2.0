"""HTTP client for communicating with the FastAPI backend prediction service."""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


class ChurnAPIClient:
    """Client for calling Churn Prediction REST API with local predictor fallback."""

    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._local_predictor = None

    def _get_local_predictor(self):
        """Lazy load local ChurnPredictor as fallback."""
        if self._local_predictor is None:
            try:
                from churn_prediction.model_inference import ChurnPredictor
                self._local_predictor = ChurnPredictor()
            except Exception as e:
                logger.error(f"Failed to load local predictor fallback: {e}")
        return self._local_predictor

    def check_health(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback
        predictor = self._get_local_predictor()
        return {
            "status": "connected (direct engine)" if predictor else "offline",
            "model_loaded": predictor is not None,
            "version": "1.0.0",
            "app_name": "Direct Inference Engine",
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Fetch model metadata."""
        try:
            resp = requests.get(f"{self.base_url}/model-info", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback
        predictor = self._get_local_predictor()
        if predictor:
            clf_name = type(predictor.pipeline.named_steps["classifier"]).__name__
            return {
                "model_name": "Customer Churn Prediction Engine",
                "model_type": clf_name,
                "version": "1.0.0",
                "decision_threshold": predictor.default_threshold,
                "feature_count": len(predictor.feature_names),
                "features": predictor.feature_names,
                "metrics": predictor.metrics,
                "metadata": predictor.metadata,
            }
        return {}

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predict churn for a single customer."""
        try:
            resp = requests.post(f"{self.base_url}/predict", json=record, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback
        predictor = self._get_local_predictor()
        if predictor:
            return predictor.predict_single(record)
        raise RuntimeError("Neither API nor local inference engine is reachable.")

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict churn for a batch of customers."""
        try:
            resp = requests.post(
                f"{self.base_url}/predict/batch",
                json={"records": records},
                timeout=self.timeout * 3,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("predictions", [])
        except Exception:
            pass

        # Fallback
        predictor = self._get_local_predictor()
        if predictor:
            import pandas as pd
            df = pd.DataFrame(records)
            scored_df, _ = predictor.predict_batch(df)
            return scored_df.to_dict(orient="records")

        raise RuntimeError("Neither API nor local inference engine is reachable.")
