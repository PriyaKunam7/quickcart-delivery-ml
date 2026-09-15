"""
Deterministic delivery-time estimator.

This is the production-safe fallback: it never depends on a trained
model, has no external dependencies at call time beyond the
already-loaded configuration, and always returns an answer. It also
serves as the baseline that the ML model must beat to be worth
deploying.

Note on input validation: this module trusts that distance_miles,
warehouse_processing_hours, weather_score, etc. are already valid
(positive, in-range) by the time they reach here. That validation is
enforced upstream by the Pydantic schemas in
quickcart_ml.validation.schemas (OrderFeatures and its subclasses) --
this module deliberately does not re-validate, to keep a single
source of truth for "what is a valid order."
"""

import math
from dataclasses import dataclass, field

from quickcart_ml.config import get_rule_engine_config


@dataclass
class DistanceBand:
    max: float
    days: int


@dataclass
class RuleEngineConfig:
    enabled: bool
    distance_bands: list[DistanceBand]
    default_transit_days: int
    severe_weather_threshold: float
    severe_weather_extra_days: int
    holiday_extra_days: int
    carrier_adjustments: dict[str, int] = field(default_factory=dict)
    min_estimate_days: int = 1

    @classmethod
    def from_dict(cls, raw: dict) -> "RuleEngineConfig":
        bands = [DistanceBand(**b) for b in raw["distance_bands"]]
        return cls(
            enabled=raw.get("enabled", True),
            distance_bands=sorted(bands, key=lambda b: b.max),
            default_transit_days=raw["default_transit_days"],
            severe_weather_threshold=raw["severe_weather_threshold"],
            severe_weather_extra_days=raw["severe_weather_extra_days"],
            holiday_extra_days=raw["holiday_extra_days"],
            carrier_adjustments=raw.get("carrier_adjustments", {}),
            min_estimate_days=raw.get("min_estimate_days", 1),
        )


def load_rule_engine_config(env: str | None = None) -> RuleEngineConfig:
    return RuleEngineConfig.from_dict(get_rule_engine_config(env))


def _transit_days_for_distance(
    distance_miles: float, bands: list[DistanceBand], default_days: int
) -> int:
    for band in bands:
        if distance_miles <= band.max:
            return band.days
    return default_days


def estimate_delivery_days(
    distance_miles: float,
    warehouse_processing_hours: float,
    weather_score: float,
    holiday_flag: int,
    carrier_code: str,
    config: RuleEngineConfig,
) -> int:
    """
    Deterministic delivery-day estimate. Same inputs always produce
    the same output -- no randomness, no model, no I/O beyond the
    already-loaded config object.
    """
    transit_days = _transit_days_for_distance(
        distance_miles, config.distance_bands, config.default_transit_days
    )

    # Ceiling division: any portion of a day counts as a full day.
    processing_days = math.ceil(warehouse_processing_hours / 24)

    weather_extra = (
        config.severe_weather_extra_days
        if weather_score >= config.severe_weather_threshold
        else 0
    )

    holiday_extra = config.holiday_extra_days if holiday_flag else 0

    # Unknown carriers get no adjustment rather than raising -- the
    # rule engine must never fail just because a new carrier was added
    # to the business without a config update yet.
    carrier_adjustment = config.carrier_adjustments.get(carrier_code, 0)

    total_days = (
        transit_days
        + processing_days
        + weather_extra
        + holiday_extra
        + carrier_adjustment
    )

    return max(total_days, config.min_estimate_days)
