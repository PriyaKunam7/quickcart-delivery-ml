"""
API tests for the QuickCart delivery prediction service.

Uses FastAPI's TestClient, which runs the app in-process (including
its startup event, so the real PredictionService and model artifact
get loaded exactly as they would in production).
"""

from fastapi.testclient import TestClient

from quickcart_ml.api.main import app

VALID_PAYLOAD = {
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
}


def test_predict_valid_request_returns_full_response_shape():
    with TestClient(app) as client:
        response = client.post("/v1/delivery/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()

        for field in (
            "estimated_delivery_days",
            "decision_source",
            "model_version",
            "request_id",
            "latency_ms",
        ):
            assert field in body

        assert body["decision_source"] in ("RULES", "ML")
        assert body["estimated_delivery_days"] > 0
        assert body["latency_ms"] >= 0


def test_predict_negative_distance_returns_422():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "distance_miles": -50.0}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"
        assert "request_id" in body


def test_predict_invalid_zip_returns_422():
    """
    Regression test for a real bug: a custom @field_validator's
    ValueError includes a raw exception object inside errors()' "ctx"
    dict, which is not JSON-serializable by default. This must be
    encoded safely (via jsonable_encoder) rather than crashing into a
    500.
    """
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "origin_zip": "941"}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"


def test_predict_unknown_carrier_returns_422():
    with TestClient(app) as client:
        payload = {**VALID_PAYLOAD, "carrier_code": "CARRIER_ZZZ"}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 422


def test_predict_missing_required_field_returns_422():
    with TestClient(app) as client:
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "distance_miles"}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 422


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_ready_endpoint_when_model_available():
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True
        assert body["details"]["model_loaded"] is True


def test_version_endpoint():
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        body = response.json()
        assert "api_version" in body
        assert body["model_name"] == "delivery_regressor"
        assert body["model_version"] == "1.0.0"


def test_openapi_docs_are_generated():
    with TestClient(app) as client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        for expected_path in (
            "/v1/delivery/predict",
            "/health",
            "/ready",
            "/version",
        ):
            assert expected_path in paths


def test_swagger_ui_is_served():
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200
