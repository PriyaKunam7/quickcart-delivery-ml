"""
Reusable feature engineering for the QuickCart delivery-time model.

Design goal: the exact same code path builds features whether called
during training (on historical TrainingRecord-shaped data, which
includes order_timestamp) or during inference (on PredictionRequest-
shaped data, which currently does NOT include order_timestamp -- see
docs/feature_contract.md for why, and how that gap is handled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------
# Column contracts
# --------------------------------------------------------------------

REQUIRED_RAW_COLUMNS = [
    "carrier_code",
    "item_category",
    "distance_miles",
    "warehouse_processing_hours",
    "weather_score",
    "holiday_flag",
    "historical_carrier_delay_days",
]

OPTIONAL_RAW_COLUMNS = ["order_timestamp", "weekend_flag"]

LEAKAGE_COLUMNS = [
    "actual_delivery_days",
    "resolved_at",
]

DERIVED_FEATURE_COLUMNS = [
    "order_hour",
    "order_day_of_week",
    "is_weekend",
    "distance_band",
    "processing_days",
    "weather_delay_risk",
    "distance_x_weather",
]

PASSTHROUGH_RAW_COLUMNS = [
    "distance_miles",
    "warehouse_processing_hours",
    "weather_score",
    "holiday_flag",
    "historical_carrier_delay_days",
]

NUMERIC_FEATURE_COLUMNS = PASSTHROUGH_RAW_COLUMNS + [
    "order_hour",
    "order_day_of_week",
    "processing_days",
    "distance_x_weather",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "carrier_code",
    "item_category",
    "distance_band",
    "weather_delay_risk",
]

BINARY_FEATURE_COLUMNS = ["is_weekend"]

DEFAULT_ORDER_HOUR = -1
DEFAULT_ORDER_DAY_OF_WEEK = -1
DEFAULT_IS_WEEKEND = 0

DISTANCE_BAND_EDGES = [0, 100, 500, 1000, 2000, np.inf]
DISTANCE_BAND_LABELS = ["0-100", "100-500", "500-1000", "1000-2000", "2000+"]

WEATHER_RISK_EDGES = [-0.01, 3, 7, 10]
WEATHER_RISK_LABELS = ["low", "medium", "high"]


class MissingRequiredColumnsError(ValueError):
    """Raised when required raw input columns are absent."""


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise MissingRequiredColumnsError(
            f"Missing required columns for feature building: {missing}"
        )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function: takes a raw features DataFrame, returns a NEW
    DataFrame containing only the derived feature columns (stable
    column list, defined in DERIVED_FEATURE_COLUMNS).

    Works whether or not order_timestamp / weekend_flag are present,
    using the documented defaults when they're missing. Never reads or
    requires the target column -- if it's present in df, it's simply
    ignored, never included in the output.
    """
    _validate_required_columns(df)

    out = pd.DataFrame(index=df.index)

    if "order_timestamp" in df.columns:
        ts = pd.to_datetime(df["order_timestamp"])
        out["order_hour"] = ts.dt.hour
        out["order_day_of_week"] = ts.dt.dayofweek
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    else:
        out["order_hour"] = DEFAULT_ORDER_HOUR
        out["order_day_of_week"] = DEFAULT_ORDER_DAY_OF_WEEK
        if "weekend_flag" in df.columns:
            out["is_weekend"] = df["weekend_flag"].astype(int)
        else:
            out["is_weekend"] = DEFAULT_IS_WEEKEND

    out["distance_band"] = pd.cut(
        df["distance_miles"],
        bins=DISTANCE_BAND_EDGES,
        labels=DISTANCE_BAND_LABELS,
    ).astype(str)

    out["processing_days"] = df["warehouse_processing_hours"] / 24.0

    out["weather_delay_risk"] = pd.cut(
        df["weather_score"],
        bins=WEATHER_RISK_EDGES,
        labels=WEATHER_RISK_LABELS,
    ).astype(str)

    out["distance_x_weather"] = df["distance_miles"] * df["weather_score"]

    return out[DERIVED_FEATURE_COLUMNS]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature-building step: validates required columns, computes
    derived features, and returns a DataFrame combining the raw
    passthrough columns, categorical columns, and derived columns --
    everything the ColumnTransformer needs, nothing else.
    """
    _validate_required_columns(df)

    derived = add_derived_features(df)
    raw_subset = df[PASSTHROUGH_RAW_COLUMNS + ["carrier_code", "item_category"]]

    combined = pd.concat(
        [raw_subset.reset_index(drop=True), derived.reset_index(drop=True)], axis=1
    )
    return combined


class DerivedFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around build_features(), so the entire
    feature engineering step -- derivation AND encoding -- can live in
    one fitted, picklable Pipeline alongside the model.
    """

    def fit(self, X: pd.DataFrame, y=None):
        _validate_required_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_features(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            PASSTHROUGH_RAW_COLUMNS
            + ["carrier_code", "item_category"]
            + DERIVED_FEATURE_COLUMNS
        )


def build_preprocessing_pipeline() -> Pipeline:
    """
    Returns an unfitted sklearn Pipeline: derive features, then encode/
    scale them. This is the single object that should be fit once on
    training data and persisted (e.g. via joblib) alongside the model
    -- training and inference both call .transform() on this exact
    object, which is what guarantees parity between the two.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURE_COLUMNS,
            ),
            ("binary", "passthrough", BINARY_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("derive_features", DerivedFeatureBuilder()),
            ("encode", column_transformer),
        ]
    )
