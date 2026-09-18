"""
FastAPI inference service.

Route handlers here only translate HTTP <-> the PredictionService's
domain objects. All actual business logic (which prediction source to
use, how to compute it) lives in inference/service.py.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quickcart_ml.inference.service import PredictionService
from quickcart_ml.validation.schemas import PredictionRequest

logger = logging.getLogger("quickcart_ml.api")

API_VERSION = "1.0.0"

# Loaded once at app startup, via the lifespan context manager below --
# not per-request, and not using the deprecated @app.on_event API.
service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = PredictionService()
    yield


app = FastAPI(
    title="QuickCart Delivery Prediction API",
    version=API_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------
# Request/response models
# --------------------------------------------------------------------
class PredictionResponseModel(BaseModel):
    estimated_delivery_days: float
    decision_source: str
    model_version: str
    request_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    details: dict


class VersionResponse(BaseModel):
    api_version: str
    model_name: str | None = None
    model_version: str | None = None
    rule_engine_enabled: bool


# --------------------------------------------------------------------
# Structured exception handling
#
# Note: request IDs are generated locally within each handler/route
# rather than via a global @app.middleware("http") (BaseHTTPMiddleware).
# That style of middleware has a known interaction with Starlette's
# exception-handling stack: exceptions raised while resolving request
# parameters (including RequestValidationError) can bypass registered
# @app.exception_handler callbacks entirely when a BaseHTTPMiddleware
# sits in the stack, incorrectly surfacing as a generic 500 instead of
# the correct 422. Generating request IDs locally sidesteps this
# entirely -- verified against a live server before shipping.
# --------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # exc.errors() can contain raw exception objects (e.g. a ValueError
    # from a custom @field_validator, inside each error's "ctx" dict),
    # which plain json.dumps cannot serialize. jsonable_encoder performs
    # the same safe conversion FastAPI's own default handler uses
    # internally -- verified against a live server with a custom
    # validator error before shipping.
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "detail": jsonable_encoder(exc.errors()),
            "request_id": str(uuid.uuid4()),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    # Full traceback goes to logs only -- never to the response body.
    logger.exception("Unhandled exception (request_id=%s)", request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "request_id": request_id,
        },
    )


# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------
@app.post("/v1/delivery/predict", response_model=PredictionResponseModel)
def predict(payload: PredictionRequest) -> PredictionResponseModel:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = service.predict(payload)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponseModel(
        estimated_delivery_days=result.estimated_delivery_days,
        decision_source=result.decision_source,
        model_version=result.model_version,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: is the process up at all? Always returns healthy if the app is running."""
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse)
def ready(response_model=None):
    """
    Readiness: can this instance actually serve traffic as configured?
    Fails (503) if the config requires the ML model and it isn't loaded.
    """
    is_ready, details = service.is_ready()
    status_code = (
        status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(
        status_code=status_code,
        content={"ready": is_ready, "details": details},
    )


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    model_name = (
        service.model_metadata["model_name"] if service.model_metadata else None
    )
    model_version = (
        service.model_metadata["model_version"] if service.model_metadata else None
    )
    return VersionResponse(
        api_version=API_VERSION,
        model_name=model_name,
        model_version=model_version,
        rule_engine_enabled=service.rule_config.enabled,
    )
