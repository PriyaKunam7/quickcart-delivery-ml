# API Contract -- QuickCart Delivery Prediction Service

Base URL (local dev): http://127.0.0.1:8000

## POST /v1/delivery/predict

Returns a delivery-time estimate, using the ML model if available and configured as primary, falling back to the deterministic rule engine otherwise.

Request body (all fields required except order_id):
{
  "origin_zip": "94107",
  "destination_zip": "10001",
  "carrier_code": "CARRIER_A",
  "item_category": "standard",
  "distance_miles": 500.0,
  "warehouse_processing_hours": 12.0,
  "weather_score": 3.0,
  "holiday_flag": 0,
  "weekend_flag": 1,
  "historical_carrier_delay_days": 0.5,
  "order_id": null
}

Success response (200):
{
  "estimated_delivery_days": 1.74,
  "decision_source": "ML",
  "model_version": "delivery_regressor:1.0.0",
  "request_id": "c8c60b29-6d39-4aa9-ac91-965941f284bf",
  "latency_ms": 17.79
}

Field meanings:
- estimated_delivery_days: the prediction, always > 0.
- decision_source: "ML" if the trained model produced this estimate, "RULES" if the deterministic rule engine did (either because rules are configured as primary, or because the ML model isn't available).
- model_version: "{model_name}:{model_version}" when ML was used, or "rule-engine-v1" when rules were used.
- request_id: unique per request, for tracing/log correlation.
- latency_ms: time spent computing the prediction (service layer only).

Validation error response (422):
{
  "error": "validation_error",
  "detail": [
    {"type": "greater_than", "loc": ["body", "distance_miles"], "msg": "Input should be greater than 0", "input": -50.0, "ctx": {"gt": 0.0}}
  ],
  "request_id": "7f6eda4d-92bb-4293-a033-c3ae19b6f096"
}
Triggered by any invalid field: negative distance, malformed ZIP code, unknown carrier or item category, out-of-range weather score, etc.

Unexpected error response (500):
{
  "error": "internal_server_error",
  "request_id": "84efe755-10f3-4be5-a9d7-2317dde00907"
}
The full stack trace is written to server logs only, keyed by request_id -- never included in the response body.

## GET /health

Liveness check: is the process running at all?
Response (200, always, as long as the process is up):
{"status": "healthy"}

## GET /ready

Readiness check: can this instance actually serve traffic as currently configured?

Response when ready (200):
{
  "ready": true,
  "details": {"rule_engine_loaded": true, "model_required": true, "model_loaded": true}
}

Response when not ready (503) -- e.g. the config requires the ML model (inference.require_model: true) but the model artifact failed to load:
{
  "ready": false,
  "details": {
    "rule_engine_loaded": true,
    "model_required": true,
    "model_loaded": false,
    "model_load_error": "FileNotFoundError: [Errno 2] No such file or directory: 'models/delivery_regressor/1.0.0/model.joblib'"
  }
}

Important distinction from /health: /ready can report false (and return 503) while POST /v1/delivery/predict continues to serve real answers via the rule-engine fallback. Readiness reflects "is the preferred configuration active," not "can this instance answer requests at all" -- the service is designed to always be able to answer, per the Week 1 Day 4 platform requirement that the system must never depend exclusively on the ML model.

## GET /version

Returns build/deployment metadata.
Response (200):
{
  "api_version": "1.0.0",
  "model_name": "delivery_regressor",
  "model_version": "1.0.0",
  "rule_engine_enabled": true
}
model_name/model_version are null if no model is currently loaded.

## Interactive Documentation

FastAPI auto-generates OpenAPI docs from the route and Pydantic model definitions:
- Swagger UI: GET /docs
- Raw OpenAPI schema: GET /openapi.json

## Decision Source Logic

inference.primary_source == "ML" and model successfully loaded -> decision_source = "ML"
otherwise (primary_source == "RULES", OR ML configured but model failed to load) -> decision_source = "RULES"

This logic lives in PredictionService.predict() (src/quickcart_ml/inference/service.py), not in the route handler -- route handlers only translate between HTTP and this service's plain domain objects.