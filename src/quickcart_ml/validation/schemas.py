"""
Pydantic data contracts for the QuickCart delivery prediction platform.

These models define the boundary between "raw data" and "data the rule
engine / ML pipeline are allowed to trust." Anything that doesn't pass
these validators should never reach downstream logic.
"""

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

ZIP_PATTERN = re.compile(r"^\d{5}$")

VALID_CARRIERS = {"CARRIER_A", "CARRIER_B", "CARRIER_C", "CARRIER_D", "CARRIER_E"}
VALID_ITEM_CATEGORIES = {"standard", "fragile", "oversize", "perishable"}


class OrderFeatures(BaseModel):
    """The core set of features describing a single order."""

    origin_zip: str
    destination_zip: str
    carrier_code: str
    item_category: str
    distance_miles: float = Field(gt=0)
    warehouse_processing_hours: float = Field(ge=0, le=168)
    weather_score: float = Field(ge=0, le=10)
    holiday_flag: int = Field(ge=0, le=1)
    weekend_flag: int = Field(ge=0, le=1)
    historical_carrier_delay_days: float = Field(ge=0)

    @field_validator("origin_zip", "destination_zip")
    @classmethod
    def validate_zip(cls, v: str) -> str:
        if not ZIP_PATTERN.match(v):
            raise ValueError(f"Invalid ZIP code format: {v!r}. Expected 5 digits.")
        return v

    @field_validator("carrier_code")
    @classmethod
    def validate_carrier(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("carrier_code must not be empty.")
        if v not in VALID_CARRIERS:
            raise ValueError(
                f"Unknown carrier_code: {v!r}. Must be one of {VALID_CARRIERS}."
            )
        return v

    @field_validator("item_category")
    @classmethod
    def validate_item_category(cls, v: str) -> str:
        if v not in VALID_ITEM_CATEGORIES:
            raise ValueError(
                f"Unknown item_category: {v!r}. Must be one of {VALID_ITEM_CATEGORIES}."
            )
        return v


class TrainingRecord(OrderFeatures):
    """A single labeled row used for model training."""

    order_id: int
    order_timestamp: datetime
    actual_delivery_days: float = Field(gt=0)


class PredictionRequest(OrderFeatures):
    """What a client sends in to get a delivery-time prediction."""

    order_id: int | None = None


class PredictionResponse(BaseModel):
    """What the API returns after making a prediction."""

    order_id: int | None = None
    predicted_delivery_days: float = Field(gt=0)
    model_version: str = "rule-engine-v1"
