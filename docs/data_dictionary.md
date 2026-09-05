# Data Dictionary — QuickCart Delivery Orders Dataset

Describes every field in `data/raw/orders.csv`. All data is synthetically generated and contains no real customer information.

| Field | Type | Valid Range / Values | Nullable | Business Meaning |
|---|---|---|---|---|
| `order_id` | integer | 1 to N, unique | No | Unique identifier for the order. Primary key. |
| `order_timestamp` | datetime (ISO 8601) | 2024-01-01 to 2025-12-31 | No | When the order was placed. Used to derive `weekend_flag` and seasonal patterns. |
| `origin_zip` | string (5 digits, zero-padded) | `00000`–`99999` | No | Warehouse/shipping origin ZIP code. |
| `destination_zip` | string (5 digits, zero-padded) | `00000`–`99999` | No | Customer delivery ZIP code. |
| `carrier_code` | string (categorical) | `CARRIER_A`, `CARRIER_B`, `CARRIER_C`, `CARRIER_D`, `CARRIER_E` | No | Which shipping carrier handled the order. Carriers have different baseline speed. |
| `item_category` | string (categorical) | `standard`, `fragile`, `oversize`, `perishable` | No | Type of item shipped. May affect handling time in future features. |
| `distance_miles` | float | 5–3000 | No | Straight-line distance between origin and destination. |
| `warehouse_processing_hours` | float | 1–72 | No | Time from order placement to leaving the warehouse. |
| `weather_score` | float | 0–10 | No | Synthetic weather-severity indicator along the delivery route (0 = clear, 10 = severe). |
| `holiday_flag` | integer (0/1) | 0 or 1 | No | Whether the order was placed on or near a holiday (higher delay risk). |
| `weekend_flag` | integer (0/1) | 0 or 1 | No | Whether the order was placed on a Saturday or Sunday. Derived from `order_timestamp`. |
| `historical_carrier_delay_days` | float | 0–3 | No | Rolling historical average delay (in days) for the assigned carrier at time of order. |
| `actual_delivery_days` | float | > 0 | No | **Target variable.** Actual number of days from order to delivery. Generated from distance, processing time, carrier effect, weather effect, holiday effect, and random noise. |

## Notes
- `origin_zip` and `destination_zip` must always be read/written as strings. Reading them as integers in pandas silently drops leading zeros (e.g. `00501` becomes `501`), which would corrupt the field. See `dataset_validator.py` for the correct `dtype={"origin_zip": str, ...}` pattern.
- The dataset generator (`scripts/generate_orders.py`) uses a fixed random seed (`42`) so the same command always produces byte-identical output — required for reproducible experiments and CI.
- No field in this dataset is derived from or represents any real person, order, or customer.