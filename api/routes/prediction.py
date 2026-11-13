"""Inference endpoints for single and batch predictions."""

from fastapi import APIRouter, HTTPException, status
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerRecordInput,
    PredictionResponse,
)
from api.services.prediction_service import PredictionService
from api.config import settings

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_single(record: CustomerRecordInput) -> PredictionResponse:
    """Predict churn status and probability for an individual customer record."""
    try:
        return PredictionService.predict_single(record)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}",
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict churn status and probability for multiple customer records with strict batch size caps."""
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request batch cannot be empty.",
        )

    # Security check: DoS prevention via batch size ceiling
    if len(request.records) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size exceeds maximum limit of {settings.max_batch_size} records (received {len(request.records)}).",
        )

    try:
        predictions = PredictionService.predict_batch(request.records)
        return BatchPredictionResponse(
            total_records=len(predictions),
            predictions=predictions,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference error: {str(e)}",
        )
