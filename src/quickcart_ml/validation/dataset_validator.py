"""
Validates data/raw/orders.csv against the expected schema and writes a
JSON report to reports/data_validation.json.

Usage:
    python -m quickcart_ml.validation.dataset_validator
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

from quickcart_ml.validation.schemas import (
    VALID_CARRIERS,
    VALID_ITEM_CATEGORIES,
    ZIP_PATTERN,
)

DATA_PATH = "data/raw/orders.csv"
REPORT_PATH = "reports/data_validation.json"

REQUIRED_COLUMNS = [
    "order_id",
    "order_timestamp",
    "origin_zip",
    "destination_zip",
    "carrier_code",
    "item_category",
    "distance_miles",
    "warehouse_processing_hours",
    "weather_score",
    "holiday_flag",
    "weekend_flag",
    "historical_carrier_delay_days",
    "actual_delivery_days",
]

NUMERIC_RANGES = {
    "distance_miles": (5, 3000),
    "warehouse_processing_hours": (1, 72),
    "weather_score": (0, 10),
    "historical_carrier_delay_days": (0, 3),
}


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    # ZIP codes must be read as strings, or pandas will strip leading zeros.
    return pd.read_csv(path, dtype={"origin_zip": str, "destination_zip": str})


def validate_dataset(df: pd.DataFrame) -> dict:
    issues = []
    invalid_record_indices: set[int] = set()

    # 1. Required columns present
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        issues.append(
            {
                "check": "required_columns",
                "detail": f"Missing columns: {missing_columns}",
            }
        )

    # 2. No duplicate order_id
    dup_count = (
        int(df["order_id"].duplicated().sum()) if "order_id" in df.columns else None
    )
    if dup_count:
        issues.append(
            {
                "check": "duplicate_order_id",
                "detail": f"{dup_count} duplicate order_id values",
            }
        )
        invalid_record_indices.update(df.index[df["order_id"].duplicated(keep=False)])

    # 3. Null percentage by field
    null_pct_by_field = (df.isnull().mean() * 100).round(3).to_dict()

    # 4. Numeric ranges
    range_violations = {}
    for col, (low, high) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if len(out_of_range) > 0:
            range_violations[col] = len(out_of_range)
            invalid_record_indices.update(out_of_range.index)

    if range_violations:
        issues.append({"check": "numeric_ranges", "detail": range_violations})

    # 5. Controlled categorical values
    bad_carriers = (
        df[~df["carrier_code"].isin(VALID_CARRIERS)]
        if "carrier_code" in df.columns
        else pd.DataFrame()
    )
    bad_categories = (
        df[~df["item_category"].isin(VALID_ITEM_CATEGORIES)]
        if "item_category" in df.columns
        else pd.DataFrame()
    )
    if len(bad_carriers) > 0:
        issues.append(
            {
                "check": "carrier_code_vocabulary",
                "detail": f"{len(bad_carriers)} invalid carrier_code values",
            }
        )
        invalid_record_indices.update(bad_carriers.index)
    if len(bad_categories) > 0:
        issues.append(
            {
                "check": "item_category_vocabulary",
                "detail": f"{len(bad_categories)} invalid item_category values",
            }
        )
        invalid_record_indices.update(bad_categories.index)

    # ZIP format check
    bad_zip = pd.Series([False] * len(df))
    for col in ("origin_zip", "destination_zip"):
        if col in df.columns:
            bad_zip = bad_zip | ~df[col].astype(str).str.match(ZIP_PATTERN)
    bad_zip_count = int(bad_zip.sum())
    if bad_zip_count > 0:
        issues.append(
            {"check": "zip_format", "detail": f"{bad_zip_count} invalid ZIP values"}
        )
        invalid_record_indices.update(df.index[bad_zip])

    # 6. Target greater than zero
    if "actual_delivery_days" in df.columns:
        bad_target = df[df["actual_delivery_days"] <= 0]
        if len(bad_target) > 0:
            issues.append(
                {
                    "check": "target_positive",
                    "detail": f"{len(bad_target)} rows with actual_delivery_days <= 0",
                }
            )
            invalid_record_indices.update(bad_target.index)

    # 7. Dataset row count
    row_count = len(df)

    # 8. Invalid record count
    invalid_record_count = len(invalid_record_indices)

    critical_violations = len(missing_columns) + (1 if dup_count else 0)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "invalid_record_count": invalid_record_count,
        "null_percentage_by_field": null_pct_by_field,
        "issues": issues,
        "critical_schema_violations": critical_violations,
        "passed": critical_violations == 0,
    }
    return report


def main() -> None:
    df = load_dataset()
    report = validate_dataset(df)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Validated {report['row_count']:,} rows.")
    print(f"Invalid records: {report['invalid_record_count']}")
    print(f"Critical schema violations: {report['critical_schema_violations']}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
