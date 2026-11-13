"""Model metadata and governance information route."""

from fastapi import APIRouter
from api.schemas import ModelInfoResponse
from api.services.prediction_service import PredictionService

router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> ModelInfoResponse:
    """Return model specifications, decision threshold, and performance metrics."""
    return PredictionService.get_model_info()
