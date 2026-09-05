"""
Generates a synthetic dataset of delivery orders for the QuickCart delivery
prediction platform. No real customer data is used anywhere in this script.

Usage:
    python scripts/generate_orders.py
"""

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 100_000

CARRIERS = ["CARRIER_A", "CARRIER_B", "CARRIER_C", "CARRIER_D", "CARRIER_E"]
ITEM_CATEGORIES = ["standard", "fragile", "oversize", "perishable"]

# Fixed per-carrier speed effect (in days). Negative = faster than baseline.
CARRIER_EFFECT_DAYS = {
    "CARRIER_A": -0.5,
    "CARRIER_B": 0.0,
    "CARRIER_C": 0.3,
    "CARRIER_D": 0.8,
    "CARRIER_E": 1.2,
}

OUTPUT_PATH = "data/raw/orders.csv"


def generate_zip_codes(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate US-style 5-digit ZIP code strings (as strings, zero-padded)."""
    codes = rng.integers(low=0, high=100_000, size=n)
    return np.array([f"{c:05d}" for c in codes])


def generate_orders(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    order_id = np.arange(1, n_rows + 1)

    # Random order timestamps over a 2-year window.
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-12-31")
    seconds_range = int((end - start).total_seconds())
    offsets = rng.integers(0, seconds_range, size=n_rows)
    order_timestamp = start + pd.to_timedelta(offsets, unit="s")

    origin_zip = generate_zip_codes(rng, n_rows)
    destination_zip = generate_zip_codes(rng, n_rows)

    carrier_code = rng.choice(CARRIERS, size=n_rows)
    item_category = rng.choice(ITEM_CATEGORIES, size=n_rows)

    distance_miles = rng.uniform(5, 3000, size=n_rows).round(1)
    warehouse_processing_hours = rng.uniform(1, 72, size=n_rows).round(1)
    weather_score = rng.uniform(0, 10, size=n_rows).round(1)
    holiday_flag = rng.choice([0, 1], size=n_rows, p=[0.92, 0.08])
    historical_carrier_delay_days = rng.uniform(0, 3, size=n_rows).round(2)

    weekend_flag = (order_timestamp.dayofweek >= 5).astype(int)

    # --- Hidden function combining signals into the target ---
    base_days = distance_miles / 500 + warehouse_processing_hours / 24
    carrier_effect = np.array([CARRIER_EFFECT_DAYS[c] for c in carrier_code])
    weather_effect = (weather_score / 10) * 2.0
    holiday_effect = holiday_flag * 1.0
    noise = rng.normal(loc=0.0, scale=0.4, size=n_rows)

    actual_delivery_days = (
        base_days + carrier_effect + weather_effect + holiday_effect + noise
    )
    # Delivery time can't be zero or negative.
    actual_delivery_days = np.clip(actual_delivery_days, 0.5, None).round(2)

    df = pd.DataFrame(
        {
            "order_id": order_id,
            "order_timestamp": order_timestamp,
            "origin_zip": origin_zip,
            "destination_zip": destination_zip,
            "carrier_code": carrier_code,
            "item_category": item_category,
            "distance_miles": distance_miles,
            "warehouse_processing_hours": warehouse_processing_hours,
            "weather_score": weather_score,
            "holiday_flag": holiday_flag,
            "weekend_flag": weekend_flag,
            "historical_carrier_delay_days": historical_carrier_delay_days,
            "actual_delivery_days": actual_delivery_days,
        }
    )
    return df


def main() -> None:
    df = generate_orders()
    import os

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
