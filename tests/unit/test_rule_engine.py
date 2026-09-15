import pytest
from pydantic import ValidationError

from quickcart_ml.rules.delivery_estimator import (
    DistanceBand,
    RuleEngineConfig,
    estimate_delivery_days,
)
from quickcart_ml.validation.schemas import OrderFeatures

TEST_CONFIG = RuleEngineConfig(
    enabled=True,
    distance_bands=[
        DistanceBand(max=249, days=1),
        DistanceBand(max=749, days=2),
        DistanceBand(max=1499, days=3),
        DistanceBand(max=2249, days=4),
    ],
    default_transit_days=5,
    severe_weather_threshold=7,
    severe_weather_extra_days=1,
    holiday_extra_days=1,
    carrier_adjustments={"CARRIER_A": -1, "CARRIER_B": 0, "CARRIER_E": 1},
    min_estimate_days=1,
)


def est(distance, hours=0.0, weather=0.0, holiday=0, carrier="CARRIER_B"):
    return estimate_delivery_days(
        distance, hours, weather, holiday, carrier, TEST_CONFIG
    )


# --------------------------------------------------------------------
# Distance band boundaries
# --------------------------------------------------------------------
def test_distance_exactly_249_is_one_day():
    assert est(249) == 1


def test_distance_exactly_250_is_two_days():
    assert est(250) == 2


def test_distance_exactly_749_is_two_days():
    assert est(749) == 2


def test_distance_exactly_750_is_three_days():
    assert est(750) == 3


def test_distance_exactly_1499_is_three_days():
    assert est(1499) == 3


def test_distance_exactly_1500_is_four_days():
    assert est(1500) == 4


def test_distance_exactly_2249_is_four_days():
    assert est(2249) == 4


def test_distance_exactly_2250_is_five_days():
    assert est(2250) == 5


def test_very_large_distance_uses_default_band():
    assert est(1_000_000) == 5


# --------------------------------------------------------------------
# Processing hours (ceiling division)
# --------------------------------------------------------------------
def test_zero_processing_hours_adds_nothing():
    assert est(300, hours=0) == 2  # base band only


def test_one_processing_hour_adds_one_day():
    assert est(300, hours=1) == 3  # ceil(1/24) = 1


def test_exactly_24_processing_hours_adds_one_day():
    assert est(300, hours=24) == 3  # ceil(24/24) = 1


def test_25_processing_hours_adds_two_days():
    assert est(300, hours=25) == 4  # ceil(25/24) = 2


# --------------------------------------------------------------------
# Severe weather boundary
# --------------------------------------------------------------------
def test_weather_just_below_threshold_adds_nothing():
    assert est(300, weather=6.99) == 2


def test_weather_exactly_at_threshold_adds_extra_day():
    assert est(300, weather=7.0) == 3


# --------------------------------------------------------------------
# Holiday on/off
# --------------------------------------------------------------------
def test_holiday_off_adds_nothing():
    assert est(300, holiday=0) == 2


def test_holiday_on_adds_one_day():
    assert est(300, holiday=1) == 3


# --------------------------------------------------------------------
# Carrier adjustments and unknown carrier
# --------------------------------------------------------------------
def test_fast_carrier_subtracts_a_day():
    assert est(300, carrier="CARRIER_A") == 1


def test_slow_carrier_adds_a_day():
    assert est(300, carrier="CARRIER_E") == 3


def test_unknown_carrier_gets_no_adjustment():
    assert est(300, carrier="CARRIER_NEVER_CONFIGURED") == 2


# --------------------------------------------------------------------
# Minimum floor
# --------------------------------------------------------------------
def test_estimate_never_goes_below_minimum():
    # Very short distance + fast carrier could theoretically go to 0;
    # must floor at min_estimate_days instead.
    assert est(1, carrier="CARRIER_A") == 1


# --------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------
def test_same_inputs_always_produce_same_output():
    result_1 = est(837, hours=13, weather=8.2, holiday=1, carrier="CARRIER_D")
    result_2 = est(837, hours=13, weather=8.2, holiday=1, carrier="CARRIER_D")
    assert result_1 == result_2


# --------------------------------------------------------------------
# Negative input is rejected by the schema, before it ever reaches the
# rule engine -- the rule engine itself trusts its inputs.
# --------------------------------------------------------------------
def test_negative_distance_rejected_by_schema_before_rules_run():
    with pytest.raises(ValidationError):
        OrderFeatures(
            origin_zip="94107",
            destination_zip="10001",
            carrier_code="CARRIER_A",
            item_category="standard",
            distance_miles=-50.0,  # invalid
            warehouse_processing_hours=12.0,
            weather_score=3.0,
            holiday_flag=0,
            weekend_flag=0,
            historical_carrier_delay_days=0.5,
        )
