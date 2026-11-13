"""Service layer managing model lifecycle and inference requests."""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from api.config import settings
from api.schemas import (
    CustomerRecordInput,
    ModelInfoResponse,
    PredictionResponse,
)
from churn_prediction.model_inference import ChurnPredictor

logger = logging.getLogger(__name__)


class PredictionService:
    """Singleton service wrapping the ChurnPredictor engine."""

    _predictor: Optional[ChurnPredictor] = None

    @classmethod
    def load_model(cls) -> ChurnPredictor:
        """Initialize or return existing predictor instance."""
        if cls._predictor is None:
            logger.info(f"Loading prediction model from {settings.model_path}")
            cls._predictor = ChurnPredictor(settings.model_path)
            logger.info("Model loaded successfully.")
        return cls._predictor

    @classmethod
    def is_model_loaded(cls) -> bool:
        """Check if model is currently loaded in memory."""
        return cls._predictor is not None

    @classmethod
    def get_model_info(cls) -> ModelInfoResponse:
        """Extract sanitized metadata from loaded model bundle."""
        predictor = cls.load_model()
        clf_type = type(predictor.pipeline.named_steps["classifier"]).__name__
        return ModelInfoResponse(
            model_name="Customer Churn Prediction Engine",
            model_type=clf_type,
            version=settings.app_version,
            decision_threshold=predictor.default_threshold,
            feature_count=len(predictor.feature_names),
            features=predictor.feature_names,
            metrics=predictor.metrics,
            metadata=predictor.metadata,
        )

    @classmethod
    def predict_single(cls, record_input: CustomerRecordInput) -> PredictionResponse:
        """Score single customer record."""
        predictor = cls.load_model()
        rec_dict = record_input.to_dict_features()
        res_dict = predictor.predict_single(rec_dict)
        return PredictionResponse(**res_dict)

    @classmethod
    def predict_batch(cls, records_input: List[CustomerRecordInput]) -> List[PredictionResponse]:
        """Score batch of customer records."""
        predictor = cls.load_model()
        records_dict = [r.to_dict_features() for r in records_input]
        df_batch = pd.DataFrame(records_dict)
        scored_df, warnings_list = predictor.predict_batch(df_batch)

        responses = []
        for idx, row in scored_df.iterrows():
            # Find any warnings for this row
            row_warns = []
            for w in warnings_list:
                if w.get("row_index") == idx:
                    row_warns.extend(w.get("warnings", []))

            res = PredictionResponse(
                customer_id=str(row["CustomerID"]) if pd.notna(row.get("CustomerID")) else None,
                churn_probability=float(row["churn_probability"]),
                churn_prediction=int(row["churn_prediction"]),
                churn_label=str(row["churn_label"]),
                risk_category=str(row["risk_category"]),
                decision_threshold=predictor.default_threshold,
                warnings=row_warns,
                calibration_note=(
                    "Probability represents estimated empirical likelihood from cross-validated gradient boosting. "
                    "It is a risk propensity score, not a deterministic guarantee of departure."
                ),
            )
            responses.append(res)
        return responses
