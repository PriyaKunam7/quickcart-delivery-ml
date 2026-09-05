import pandas as pd

from quickcart_ml.validation.dataset_validator import validate_dataset

VALID_ROW = {
    "order_id": 1,
    "order_timestamp": "2024-01-01T00:00:00",
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
    "actual_delivery_days": 4.2,
}


def make_df(overrides_list):
    rows = []
    for i, overrides in enumerate(overrides_list, start=1):
        row = {**VALID_ROW, "order_id": i, **overrides}
        rows.append(row)
    return pd.DataFrame(rows)


def test_clean_dataset_passes():
    df = make_df([{}, {"order_id": 2}])
    report = validate_dataset(df)
    assert report["passed"] is True
    assert report["critical_schema_violations"] == 0


def test_duplicate_order_id_detected():
    df = make_df([{"order_id": 1}, {"order_id": 1}])
    report = validate_dataset(df)
    assert report["invalid_record_count"] > 0
    assert any(issue["check"] == "duplicate_order_id" for issue in report["issues"])


def test_missing_required_column_is_critical():
    df = make_df([{}])
    df = df.drop(columns=["carrier_code"])
    report = validate_dataset(df)
    assert report["critical_schema_violations"] > 0
    assert report["passed"] is False


def test_out_of_range_distance_flagged():
    df = make_df([{"distance_miles": 99999}])
    report = validate_dataset(df)
    assert report["invalid_record_count"] > 0


def test_target_not_positive_flagged():
    df = make_df([{"actual_delivery_days": 0}])
    report = validate_dataset(df)
    assert any(issue["check"] == "target_positive" for issue in report["issues"])
