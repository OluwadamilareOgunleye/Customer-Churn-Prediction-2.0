"""FastAPI application configuration settings."""

import os
from pathlib import Path
from typing import List
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class APISettings(BaseModel):
    """Application settings with environment variable overrides."""
    app_name: str = "Customer Churn Prediction API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    model_path: Path = PROJECT_ROOT / "models" / "churn_pipeline_bundle.joblib"
    
    # Security Configuration
    cors_origins: List[str] = [
        origin.strip() for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501,http://localhost:8000,http://127.0.0.1:8000"
        ).split(",") if origin.strip()
    ]
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "2500"))
    max_payload_bytes: int = int(os.getenv("MAX_PAYLOAD_BYTES", str(10 * 1024 * 1024))) # 10 MB
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "180")) # 180 req/min
    enable_rate_limiting: bool = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"


settings = APISettings()
