"""Health, liveness, and readiness probes."""

from fastapi import APIRouter, Response, status
from api.config import settings
from api.schemas import HealthResponse, ReadinessResponse
from api.services.prediction_service import PredictionService
from churn_prediction.health import HealthChecker

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Liveness probe: returns basic service responsiveness and process status."""
    return HealthResponse(
        status="healthy",
        model_loaded=PredictionService.is_model_loaded(),
        version=settings.app_version,
        app_name=settings.app_name,
    )


@router.get("/ready", response_model=ReadinessResponse)
def get_readiness(response: Response) -> ReadinessResponse:
    """
    Readiness probe: validates pipeline bundle integrity, memory headroom, and serving readiness.
    Returns HTTP 200 if ready, HTTP 503 Service Unavailable if unready.
    """
    readiness_data = HealthChecker.check_readiness(
        model_path=settings.model_path,
        predictor_instance=PredictionService._predictor
    )

    if not readiness_data["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(**readiness_data)
