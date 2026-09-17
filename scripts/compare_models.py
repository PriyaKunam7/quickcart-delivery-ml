"""
Compares the trained ML model against the deterministic rule engine
baseline, producing a Markdown report with overall and segmented
metrics, the worst disagreements between the two, and a summary of
which one wins more often.

Important: comparison runs on the same held-out test split used
during training (same random_state and test_size), not the full
dataset. Evaluating against rows the model was trained on would
inflate its apparent performance -- this keeps the comparison honest.

Usage:
    python scripts/compare_models.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from quickcart_ml.rules.delivery_estimator import (
    estimate_delivery_days,
    load_rule_engine_config,
)
from quickcart_ml.training.train import RANDOM_STATE, TEST_SIZE

DATA_PATH = "data/raw/orders.csv"
MODEL_PATH = "models/delivery_regressor/1.0.0/model.joblib"
REPORT_PATH = "reports/model_vs_rules.md"

DISTANCE_BAND_EDGES = [0, 250, 750, 1500, 2250, np.inf]
DISTANCE_BAND_LABELS = ["0-249", "250-749", "750-1499", "1500-2249", "2250+"]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - actual
    abs_errors = np.abs(errors)
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def format_metrics_table(rows: list[tuple[str, dict, dict]]) -> str:
    """rows: list of (segment_name, ml_metrics, rules_metrics)"""
    lines = [
        "| Segment | n | ML MAE | Rules MAE | ML RMSE | Rules RMSE | ML Improvement |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, ml, rules in rows:
        improvement = (
            (rules["mae"] - ml["mae"]) / rules["mae"] * 100 if rules["mae"] > 0 else 0
        )
        lines.append(
            f"| {name} | {ml['n']} | {ml['mae']:.3f} | {rules['mae']:.3f} | "
            f"{ml['rmse']:.3f} | {rules['rmse']:.3f} | {improvement:+.1f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    full_df = pd.read_csv(
        DATA_PATH,
        dtype={"origin_zip": str, "destination_zip": str},
        parse_dates=["order_timestamp"],
    )
    model = joblib.load(MODEL_PATH)
    rule_config = load_rule_engine_config()

    feature_columns = [c for c in full_df.columns if c != "actual_delivery_days"]
    X = full_df[feature_columns]
    y = full_df["actual_delivery_days"]

    # Same split as training -- comparison only happens on data the
    # model never saw during fitting.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    df = X_test.copy()
    df["actual_delivery_days"] = y_test
    actual = y_test.to_numpy()

    ml_predictions = model.predict(X_test)

    rules_predictions = np.array(
        [
            estimate_delivery_days(
                distance_miles=row.distance_miles,
                warehouse_processing_hours=row.warehouse_processing_hours,
                weather_score=row.weather_score,
                holiday_flag=row.holiday_flag,
                carrier_code=row.carrier_code,
                config=rule_config,
            )
            for row in df.itertuples(index=False)
        ]
    )

    df["ml_pred"] = ml_predictions
    df["rules_pred"] = rules_predictions
    df["distance_band"] = pd.cut(
        df["distance_miles"], bins=DISTANCE_BAND_EDGES, labels=DISTANCE_BAND_LABELS
    )

    overall_ml = compute_metrics(actual, ml_predictions)
    overall_rules = compute_metrics(actual, rules_predictions)

    carrier_rows = []
    for carrier, group in df.groupby("carrier_code", observed=True):
        ml_m = compute_metrics(
            group["actual_delivery_days"].to_numpy(), group["ml_pred"].to_numpy()
        )
        rules_m = compute_metrics(
            group["actual_delivery_days"].to_numpy(), group["rules_pred"].to_numpy()
        )
        carrier_rows.append((carrier, ml_m, rules_m))

    distance_rows = []
    for band, group in df.groupby("distance_band", observed=True):
        ml_m = compute_metrics(
            group["actual_delivery_days"].to_numpy(), group["ml_pred"].to_numpy()
        )
        rules_m = compute_metrics(
            group["actual_delivery_days"].to_numpy(), group["rules_pred"].to_numpy()
        )
        distance_rows.append((str(band), ml_m, rules_m))

    category_rows = []
    for category, group in df.groupby("item_category", observed=True):
        ml_m = compute_metrics(
            group["actual_delivery_days"].to_numpy(), group["ml_pred"].to_numpy()
        )
        rules_m = compute_metrics(
            group["actual_delivery_days"].to_numpy(), group["rules_pred"].to_numpy()
        )
        category_rows.append((category, ml_m, rules_m))

    # Worst disagreements: largest absolute difference between the two
    # predictions, regardless of which one is closer to actual.
    df["disagreement"] = np.abs(df["ml_pred"] - df["rules_pred"])
    worst_20 = df.nlargest(20, "disagreement")[
        [
            "order_id",
            "carrier_code",
            "distance_miles",
            "actual_delivery_days",
            "ml_pred",
            "rules_pred",
            "disagreement",
        ]
    ]

    ml_abs_err = np.abs(ml_predictions - actual)
    rules_abs_err = np.abs(rules_predictions - actual)
    ml_wins = float(np.mean(ml_abs_err < rules_abs_err) * 100)
    rules_wins = float(np.mean(rules_abs_err < ml_abs_err) * 100)
    ties = max(0.0, 100 - ml_wins - rules_wins)

    overall_improvement = (
        (overall_rules["mae"] - overall_ml["mae"]) / overall_rules["mae"] * 100
    )

    # --- Build the report ---
    lines = ["# Model vs. Rules Comparison Report", ""]
    lines.append("## Overall Metrics")
    lines.append(format_metrics_table([("Overall", overall_ml, overall_rules)]))
    lines.append("")
    lines.append(
        f"**ML improves MAE over rules by {overall_improvement:.1f}% overall.**"
    )
    lines.append("")

    lines.append("## By Carrier")
    lines.append(format_metrics_table(carrier_rows))
    lines.append("")

    lines.append("## By Distance Band")
    lines.append(format_metrics_table(distance_rows))
    lines.append("")

    lines.append("## By Item Category")
    lines.append(format_metrics_table(category_rows))
    lines.append("")

    lines.append("## Win Rate")
    lines.append(f"- ML closer to actual: **{ml_wins:.1f}%** of orders")
    lines.append(f"- Rules closer to actual: **{rules_wins:.1f}%** of orders")
    lines.append(f"- Tied (identical error): {ties:.1f}%")
    lines.append("")

    lines.append("## Worst 20 Disagreements (largest |ML - Rules| gap)")
    lines.append(
        "| order_id | carrier | distance_miles | actual | ml_pred | rules_pred | disagreement |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in worst_20.itertuples(index=False):
        lines.append(
            f"| {row.order_id} | {row.carrier_code} | {row.distance_miles:.1f} | "
            f"{row.actual_delivery_days:.2f} | {row.ml_pred:.2f} | {row.rules_pred:.2f} | "
            f"{row.disagreement:.2f} |"
        )

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(
        f"Overall ML MAE: {overall_ml['mae']:.4f}  Rules MAE: {overall_rules['mae']:.4f}"
    )
    print(f"ML improvement over rules: {overall_improvement:.1f}%")
    print(f"ML wins: {ml_wins:.1f}%  Rules wins: {rules_wins:.1f}%")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
