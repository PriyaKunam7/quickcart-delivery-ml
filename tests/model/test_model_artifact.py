import json

import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = "models/delivery_regressor/1.0.0/model.joblib"
METADATA_PATH = "models/delivery_regressor/1.0.0/metadata.json"

SAMPLE_ORDER = {
    "order_id": 1,
    "order_timestamp": pd.Timestamp("2024-06-15T14:30:00"),
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


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def make_sample_df(**overrides):
    row = {**SAMPLE_ORDER, **overrides}
    return pd.DataFrame([row])


def test_model_artifact_loads(model):
    assert model is not None


def test_prediction_returns_numeric_result(model):
    X = make_sample_df()
    prediction = model.predict(X)
    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.floating))
    assert not np.isnan(prediction[0])


def test_preprocessing_handles_unknown_category(model):
    X = make_sample_df(carrier_code="CARRIER_NEVER_SEEN_BEFORE")
    # Should not raise, thanks to OneHotEncoder(handle_unknown="ignore")
    prediction = model.predict(X)
    assert not np.isnan(prediction[0])


def test_prediction_output_range_is_valid(model):
    X = make_sample_df()
    prediction = model.predict(X)
    assert prediction[0] > 0  # delivery time can't be zero or negative


def test_same_artifact_produces_deterministic_inference(model):
    X = make_sample_df()
    prediction_1 = model.predict(X)
    prediction_2 = model.predict(X)
    np.testing.assert_allclose(prediction_1, prediction_2, rtol=1e-10)


def test_metadata_version_is_present(metadata):
    assert "model_version" in metadata
    assert metadata["model_version"] == "1.0.0"


def test_metadata_contains_required_fields(metadata):
    required_fields = [
        "model_name",
        "model_version",
        "algorithm",
        "training_dataset",
        "random_state",
        "metrics",
        "git_commit",
    ]
    for field in required_fields:
        assert field in metadata, f"Missing required metadata field: {field}"


def test_metadata_metrics_are_populated_not_placeholder(metadata):
    assert metadata["metrics"]["mae"] > 0
    assert metadata["metrics"]["rmse"] > 0


def test_no_nan_predictions_across_a_batch(model):
    batch = pd.concat(
        [make_sample_df(distance_miles=d) for d in [10, 500, 1500, 2500, 2999]]
    )
    predictions = model.predict(batch)
    assert not np.isnan(predictions).any()


def test_all_predictions_in_batch_are_positive(model):
    batch = pd.concat(
        [make_sample_df(distance_miles=d) for d in [10, 500, 1500, 2500, 2999]]
    )
    predictions = model.predict(batch)
    assert (predictions > 0).all()
