import numpy as np
import pandas as pd
import pytest

from quickcart_ml.features.build_features import (
    LEAKAGE_COLUMNS,
    MissingRequiredColumnsError,
    add_derived_features,
    build_features,
    build_preprocessing_pipeline,
)

TRAINING_ROW = {
    "order_id": 1,
    "order_timestamp": "2024-06-15T14:30:00",  # a Saturday
    "carrier_code": "CARRIER_A",
    "item_category": "standard",
    "distance_miles": 500.0,
    "warehouse_processing_hours": 24.0,
    "weather_score": 5.0,
    "holiday_flag": 0,
    "weekend_flag": 1,
    "historical_carrier_delay_days": 0.5,
    "actual_delivery_days": 4.2,
}


def make_training_df(n=5, **overrides):
    rows = [{**TRAINING_ROW, "order_id": i, **overrides} for i in range(1, n + 1)]
    return pd.DataFrame(rows)


def make_inference_df(n=5, **overrides):
    """Same shape as the current PredictionRequest schema: no order_timestamp."""
    row = {
        k: v
        for k, v in TRAINING_ROW.items()
        if k not in ("order_timestamp", "order_id", "actual_delivery_days")
    }
    rows = [{**row, **overrides} for _ in range(n)]
    return pd.DataFrame(rows)


def test_same_input_produces_same_derived_features():
    df = make_training_df()
    features_1 = add_derived_features(df)
    features_2 = add_derived_features(df)
    pd.testing.assert_frame_equal(features_1, features_2)


def test_derived_features_from_timestamp_are_correct():
    df = make_training_df(n=1)
    features = add_derived_features(df)
    assert features.loc[0, "order_hour"] == 14
    assert features.loc[0, "order_day_of_week"] == 5
    assert features.loc[0, "is_weekend"] == 1


def test_missing_timestamp_falls_back_to_weekend_flag():
    df = make_inference_df(n=1, weekend_flag=1)
    features = add_derived_features(df)
    assert features.loc[0, "is_weekend"] == 1
    assert features.loc[0, "order_hour"] == -1


def test_unknown_carrier_is_accepted_by_encoder():
    df = make_training_df(n=20)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(df)

    novel = make_training_df(n=1, carrier_code="CARRIER_NEVER_SEEN")
    result = pipeline.transform(novel)
    assert result.shape[0] == 1


def test_input_column_ordering_does_not_break_pipeline():
    df = make_training_df(n=20)
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(df)

    shuffled = df[df.columns[::-1]]
    result_normal = pipeline.transform(df)
    result_shuffled = pipeline.transform(shuffled)

    normal_arr = (
        result_normal.toarray() if hasattr(result_normal, "toarray") else result_normal
    )
    shuffled_arr = (
        result_shuffled.toarray()
        if hasattr(result_shuffled, "toarray")
        else result_shuffled
    )
    np.testing.assert_array_almost_equal(normal_arr, shuffled_arr)


def test_missing_required_feature_fails_fast():
    df = make_training_df().drop(columns=["distance_miles"])
    with pytest.raises(MissingRequiredColumnsError):
        build_features(df)


def test_target_column_is_excluded_from_inference_features():
    df = make_training_df()
    features = build_features(df)
    for leaky_col in LEAKAGE_COLUMNS:
        assert leaky_col not in features.columns


def test_build_features_works_without_target_present():
    df = make_inference_df()
    features = build_features(df)
    assert "actual_delivery_days" not in features.columns


def test_pipeline_produces_stable_output_shape():
    df = make_training_df(n=50)
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 50
