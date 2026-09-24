"""
FastAPI inference service, now with shadow mode.

Shadow ML inference is scheduled via FastAPI's BackgroundTasks, which
run AFTER the response has already been sent to the client. This is
what "run ML asynchronously within request path boundaries" means in
practice: the shadow comparison is still part of handling this exact
request (not deferred to an external queue or separate service), but
its latency is never added to what the caller waits for.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quickcart_ml.inference.service import PredictionService
from quickcart_ml.validation.schemas import PredictionRequest

logger = logging.getLogger("quickcart_ml.api")

API_VERSION = "1.0.0"

service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    global service
    service = PredictionService()
    yield


app = FastAPI(
    title="QuickCart Delivery Prediction API",
    version=API_VERSION,
    lifespan=lifespan,
)


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
    shadow_enabled: bool


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
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
    logger.exception("Unhandled exception (request_id=%s)", request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_server_error", "request_id": request_id},
    )


@app.post("/v1/delivery/predict", response_model=PredictionResponseModel)
def predict(
    payload: PredictionRequest, background_tasks: BackgroundTasks
) -> PredictionResponseModel:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = service.predict(payload)
    latency_ms = (time.perf_counter() - start) * 1000

    # Scheduled here, executed by Starlette AFTER the response below is
    # sent to the client -- shadow latency is invisible to the caller.
    if service.should_shadow_sample(payload, request_id):
        background_tasks.add_task(
            service.run_shadow_comparison, payload, request_id, result
        )

    return PredictionResponseModel(
        estimated_delivery_days=result.estimated_delivery_days,
        decision_source=result.decision_source,
        model_version=result.model_version,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse)
def ready():
    is_ready, details = service.is_ready()
    status_code = (
        status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(
        status_code=status_code, content={"ready": is_ready, "details": details}
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
        shadow_enabled=service.shadow_config.enabled,
    )
