# Feature Contract -- Delivery Time Model

Documents every feature the model consumes: where it comes from, how it's derived, and the rules that keep training and inference consistent. Implemented in `src/quickcart_ml/features/build_features.py`.

## Target & Excluded Columns

| Column | Status | Reason |
|---|---|---|
| `actual_delivery_days` | **Excluded (target)** | This is what the model predicts. Must never appear as an input feature. |
| `order_id` | Excluded from fitting | Retained in raw data for traceability, never passed to the model. |
| `origin_zip`, `destination_zip` | Excluded from fitting | Cardinality (~63k unique values each) is too high for direct encoding at this stage; would require dedicated geographic bucketing (a future iteration). |

## Required vs. Optional Raw Inputs

**Required in every input** (training or inference) -- missing any of these raises `MissingRequiredColumnsError` immediately:
`carrier_code`, `item_category`, `distance_miles`, `warehouse_processing_hours`, `weather_score`, `holiday_flag`, `historical_carrier_delay_days`

**Optional** -- present on some inputs, not others:
- `order_timestamp` -- present in historical/training data (`TrainingRecord`), **not currently present** in the live inference schema (`PredictionRequest`).
- `weekend_flag` -- present on both training and inference schemas.

### Important design note: the `order_timestamp` gap

Three derived features (`order_hour`, `order_day_of_week`, `is_weekend`) are naturally computed from `order_timestamp`. But `order_timestamp` isn't part of the current `PredictionRequest` schema -- so at real inference time, it won't be available.

**Resolution implemented here:**
- If `order_timestamp` is present -> derive `order_hour`, `order_day_of_week`, and `is_weekend` directly from it.
- If `order_timestamp` is absent -> `order_hour`/`order_day_of_week` fall back to a documented sentinel default (`-1`, meaning "unknown"), and `is_weekend` falls back to the `weekend_flag` field if supplied, else `0`.

This means the *same code path* runs in both training and inference, producing a consistent (if sometimes less-informed) feature set either way. **Recommended follow-up:** add `order_timestamp` to `PredictionRequest` so `order_hour`/`order_day_of_week` are genuinely informative at inference time too.

## Derived Features

| Derived Feature | Source Field(s) | Transformation | Type | Default (if source missing) | Null Handling |
|---|---|---|---|---|---|
| `order_hour` | `order_timestamp` | Hour of day (0-23) | int | `-1` (unknown) | N/A |
| `order_day_of_week` | `order_timestamp` | Day of week (0=Mon...6=Sun) | int | `-1` (unknown) | N/A |
| `is_weekend` | `order_timestamp` or `weekend_flag` | `1` if Sat/Sun, else `0` | int (binary) | `0` (assume weekday) | N/A |
| `distance_band` | `distance_miles` | Binned: `0-100`,`100-500`,`500-1000`,`1000-2000`,`2000+` | categorical | N/A -- required | N/A -- required |
| `processing_days` | `warehouse_processing_hours` | Divided by 24 | float | N/A -- required | N/A -- required |
| `weather_delay_risk` | `weather_score` | Binned: `low`(0-3),`medium`(3-7),`high`(7-10) | categorical | N/A -- required | N/A -- required |
| `distance_x_weather` | `distance_miles`, `weather_score` | Product | float | N/A -- required | N/A -- required |

## Preprocessing (ColumnTransformer)

| Feature Group | Columns | Transformer |
|---|---|---|
| Numeric | `distance_miles`, `warehouse_processing_hours`, `weather_score`, `holiday_flag`, `historical_carrier_delay_days`, `order_hour`, `order_day_of_week`, `processing_days`, `distance_x_weather` | `StandardScaler` |
| Categorical | `carrier_code`, `item_category`, `distance_band`, `weather_delay_risk` | `OneHotEncoder(handle_unknown="ignore")` |
| Binary | `is_weekend` | Passthrough |

## Training/Inference Parity Requirements

1. Training and inference must call the same `build_preprocessing_pipeline()` object -- fit once, persisted via `joblib`, loaded as-is for inference.
2. `add_derived_features` is a pure function with no fitted state -- always safe to call identically at both times.
3. `OneHotEncoder`/`StandardScaler` have fitted state from training data only -- never refit at inference time.
4. Column ordering in the input DataFrame does not matter -- `ColumnTransformer` selects by name, not position.