import pytest
from pydantic import ValidationError

from quickcart_ml.validation.schemas import OrderFeatures, PredictionRequest

VALID_PAYLOAD = {
    "origin_zip": "94107",
    "destination_zip": "10001",
    "carrier_code": "CARRIER_A",
    "item_category": "standard",
    "distance_miles": 500.0,
    "warehouse_processing_hours": 12.0,
    "weather_score": 3.0,
    "holiday_flag": 0,
    "weekend_flag": 0,
    "historical_carrier_delay_days": 0.5,
}


def test_valid_request():
    order = OrderFeatures(**VALID_PAYLOAD)
    assert order.carrier_code == "CARRIER_A"


def test_missing_carrier():
    payload = {**VALID_PAYLOAD, "carrier_code": ""}
    with pytest.raises(ValidationError):
        OrderFeatures(**payload)


def test_invalid_zip():
    payload = {**VALID_PAYLOAD, "origin_zip": "941"}
    with pytest.raises(ValidationError):
        OrderFeatures(**payload)


def test_negative_distance():
    payload = {**VALID_PAYLOAD, "distance_miles": -10}
    with pytest.raises(ValidationError):
        OrderFeatures(**payload)


def test_weather_score_too_high():
    payload = {**VALID_PAYLOAD, "weather_score": 15}
    with pytest.raises(ValidationError):
        OrderFeatures(**payload)


def test_unsupported_item_category():
    payload = {**VALID_PAYLOAD, "item_category": "furniture"}
    with pytest.raises(ValidationError):
        OrderFeatures(**payload)


def test_unknown_carrier_rejected():
    payload = {**VALID_PAYLOAD, "carrier_code": "CARRIER_Z"}
    with pytest.raises(ValidationError):
        OrderFeatures(**payload)


def test_prediction_request_allows_optional_order_id():
    req = PredictionRequest(**VALID_PAYLOAD)
    assert req.order_id is None
