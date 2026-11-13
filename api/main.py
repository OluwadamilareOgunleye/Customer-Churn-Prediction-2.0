"""Main FastAPI application entrypoint with security hardening."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.routes import health, info, prediction
from api.services.prediction_service import PredictionService
from churn_prediction.security import InMemoryRateLimiter, SECURITY_HEADERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize rate limiter
rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.rate_limit_rpm)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model pipeline during application startup."""
    logger.info("Initializing prediction engine at application startup...")
    try:
        PredictionService.load_model()
        logger.info("Model pipeline loaded and ready for serving.")
    except Exception as e:
        logger.error(f"Failed to load model bundle at startup: {e}")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production REST API for Telecom Customer Churn Prediction and Risk Scoring.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    """
    Middleware enforcing:
    1. Payload content-length size limit
    2. Client rate-limiting
    3. Production HTTP security headers
    """
    # 1. Check Content-Length to mitigate large payload DoS
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_payload_bytes:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"detail": f"Request body exceeds maximum allowed size of {settings.max_payload_bytes} bytes."},
        )

    # 2. Rate limiting check
    if settings.enable_rate_limiting:
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = rate_limiter.is_allowed(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Maximum 180 requests per minute allowed."},
                headers={"Retry-After": "60"},
            )

    response = await call_next(request)

    # 3. Inject standard security headers
    for header_name, header_val in SECURITY_HEADERS.items():
        response.headers[header_name] = header_val

    return response


# Register routers
app.include_router(health.router)
app.include_router(info.router)
app.include_router(prediction.router)


@app.get("/", tags=["Root"])
def root():
    """Welcome endpoint with API navigation information."""
    return {
        "message": "Welcome to the Telco Customer Churn Prediction API",
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_check": "/health",
        "model_info": "/model-info",
        "status": "operational",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Clean global exception handler preventing internal stack traces from leaking."""
    logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the prediction request."},
    )
