"""
Integration tests for shadow mode through the actual FastAPI app --
not just the sampling math in isolation, but the real end-to-end
behavior: does the API response stay RULES-only while shadow is
active, at 0%, 25%, and 100% configurations.
"""

import logging

import pytest
from fastapi.testclient import TestClient

from quickcart_ml.api import main as api_main
from quickcart_ml.config import load_config
from quickcart_ml.inference.service import PredictionService

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


def make_service_with_shadow(
    percentage: int, primary_source: str = "ML"
) -> PredictionService:
    """
    Builds a real PredictionService with shadow config overridden in
    memory, without needing a separate config file per test case.
    """
    load_config.cache_clear()
    service = PredictionService()
    service.shadow_config.enabled = True
    service.shadow_config.percentage = percentage
    service.inference_config["primary_source"] = primary_source
    return service


@pytest.fixture()
def client_factory():
    def _make(percentage: int, primary_source: str = "ML"):
        test_service = make_service_with_shadow(percentage, primary_source)
        api_main.app.dependency_overrides = (
            {}
        )  # not using Depends() here; set module global instead
        api_main.service = test_service
        return TestClient(api_main.app), test_service

    return _make


def test_shadow_at_0_percent_never_samples_and_returns_rules(client_factory):
    client, service = client_factory(percentage=0)

    for order_id in range(20):
        payload = {**VALID_PAYLOAD, "order_id": order_id}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["decision_source"] == "RULES"
        assert not service.should_shadow_sample(
            api_main.PredictionRequest(**payload), body["request_id"]
        )


def test_shadow_at_100_percent_always_samples_but_still_returns_rules(client_factory):
    client, service = client_factory(percentage=100)

    for order_id in range(20):
        payload = {**VALID_PAYLOAD, "order_id": order_id}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        # The critical assertion: even though shadow will sample every
        # single one of these requests, the returned decision is still
        # rules -- shadow mode NEVER changes the live response.
        assert body["decision_source"] == "RULES"
        assert body["model_version"] == "rule-engine-v1"


def test_shadow_at_25_percent_returns_rules_regardless_of_sampling(client_factory):
    client, service = client_factory(percentage=25)

    decision_sources = set()
    for order_id in range(50):
        payload = {**VALID_PAYLOAD, "order_id": order_id}
        response = client.post("/v1/delivery/predict", json=payload)
        assert response.status_code == 200
        decision_sources.add(response.json()["decision_source"])

    # Every single response must be RULES -- there should never be a
    # mix of "RULES" and "ML" in the returned decision_source while
    # shadow is enabled, regardless of which requests got sampled for
    # the background shadow comparison.
    assert decision_sources == {"RULES"}


def test_shadow_forces_rules_even_when_primary_source_is_ml(client_factory):
    """
    The key design guarantee: primary_source=ML must NOT win over an
    active shadow configuration. Shadow mode exists specifically to
    validate ML before trusting it for real decisions.
    """
    client, service = client_factory(percentage=100, primary_source="ML")
    assert service.inference_config["primary_source"] == "ML"

    response = client.post(
        "/v1/delivery/predict", json={**VALID_PAYLOAD, "order_id": 1}
    )
    assert response.json()["decision_source"] == "RULES"


def test_shadow_disabled_allows_ml_as_primary(client_factory):
    """Control case: with shadow OFF, ML can be the live decision source again."""
    load_config.cache_clear()
    service = PredictionService()
    service.shadow_config.enabled = False
    api_main.service = service
    client = TestClient(api_main.app)

    response = client.post(
        "/v1/delivery/predict", json={**VALID_PAYLOAD, "order_id": 1}
    )
    assert response.json()["decision_source"] == "ML"


def test_sampled_request_emits_comparison_event_in_background(client_factory, caplog):
    client, service = client_factory(percentage=100)

    with caplog.at_level(logging.INFO, logger="quickcart_ml.shadow"):
        response = client.post(
            "/v1/delivery/predict", json={**VALID_PAYLOAD, "order_id": 7}
        )

    assert response.status_code == 200
    shadow_logs = [r for r in caplog.records if r.name == "quickcart_ml.shadow"]
    assert len(shadow_logs) == 1

    import json

    event = json.loads(shadow_logs[0].message)
    assert event["request_id"] == response.json()["request_id"]
    assert "rule_prediction" in event
    assert "ml_prediction" in event
    assert "difference" in event
    assert "model_version" in event
    assert "sampled_bucket" in event
    assert "ml_latency_ms" in event


def test_unsampled_request_at_0_percent_emits_no_comparison_event(
    client_factory, caplog
):
    client, service = client_factory(percentage=0)

    with caplog.at_level(logging.INFO, logger="quickcart_ml.shadow"):
        response = client.post(
            "/v1/delivery/predict", json={**VALID_PAYLOAD, "order_id": 7}
        )

    assert response.status_code == 200
    shadow_logs = [r for r in caplog.records if r.name == "quickcart_ml.shadow"]
    assert len(shadow_logs) == 0
