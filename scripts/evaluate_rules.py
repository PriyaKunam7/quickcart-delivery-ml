"""
Evaluates the deterministic rule engine against historical order data,
producing MAE, RMSE, median absolute error, and percent-within-N-days
metrics -- overall and segmented by carrier, distance band, and item
category.

This is the benchmark the ML model must beat to be worth deploying.

Usage:
    python scripts/evaluate_rules.py
"""

import json
import os

import numpy as np
import pandas as pd

from quickcart_ml.rules.delivery_estimator import (
    estimate_delivery_days,
    load_rule_engine_config,
)

DATA_PATH = "data/raw/orders.csv"
REPORT_PATH = "reports/rules_baseline_metrics.json"

DISTANCE_BAND_EDGES = [0, 250, 750, 1500, 2250, np.inf]
DISTANCE_BAND_LABELS = ["0-249", "250-749", "750-1499", "1500-2249", "2250+"]


def compute_predictions(df: pd.DataFrame, config) -> np.ndarray:
    predictions = np.zeros(len(df), dtype=int)
    for i, row in enumerate(df.itertuples(index=False)):
        predictions[i] = estimate_delivery_days(
            distance_miles=row.distance_miles,
            warehouse_processing_hours=row.warehouse_processing_hours,
            weather_score=row.weather_score,
            holiday_flag=row.holiday_flag,
            carrier_code=row.carrier_code,
            config=config,
        )
    return predictions


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - actual
    abs_errors = np.abs(errors)

    return {
        "n": int(len(actual)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(abs_errors)),
        "pct_within_1_day": float(np.mean(abs_errors <= 1) * 100),
        "pct_within_2_days": float(np.mean(abs_errors <= 2) * 100),
    }


def segmented_metrics(df: pd.DataFrame, predicted: np.ndarray, group_col: str) -> dict:
    result = {}
    df = df.copy()
    df["_predicted"] = predicted
    for group_value, group_df in df.groupby(group_col, observed=True):
        result[str(group_value)] = compute_metrics(
            group_df["actual_delivery_days"].to_numpy(),
            group_df["_predicted"].to_numpy(),
        )
    return result


def main() -> None:
    config = load_rule_engine_config()
    df = pd.read_csv(DATA_PATH)

    predicted = compute_predictions(df, config)
    actual = df["actual_delivery_days"].to_numpy()

    overall = compute_metrics(actual, predicted)

    df["distance_band"] = pd.cut(
        df["distance_miles"], bins=DISTANCE_BAND_EDGES, labels=DISTANCE_BAND_LABELS
    )

    report = {
        "overall": overall,
        "by_carrier": segmented_metrics(df, predicted, "carrier_code"),
        "by_distance_band": segmented_metrics(df, predicted, "distance_band"),
        "by_item_category": segmented_metrics(df, predicted, "item_category"),
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Overall MAE: {overall['mae']:.3f} days")
    print(f"Overall RMSE: {overall['rmse']:.3f} days")
    print(f"Median absolute error: {overall['median_absolute_error']:.3f} days")
    print(f"Within ±1 day: {overall['pct_within_1_day']:.1f}%")
    print(f"Within ±2 days: {overall['pct_within_2_days']:.1f}%")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
