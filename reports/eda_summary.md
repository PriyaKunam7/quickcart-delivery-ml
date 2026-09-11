# EDA Summary — Delivery Time Dataset

Findings from `notebooks/01_delivery_eda.ipynb`, run against `data/raw/orders.csv` (100,000 rows).

## Dataset Shape & Quality
- **Shape:** 100,000 rows x 13 columns.
- **Missing values:** zero across all columns. This is expected for a synthetically generated dataset, but real-world CMDB/order data typically won't be this clean.

## Cardinality of Categorical Fields
| Field | Unique Values |
|---|---|
| `carrier_code` | 5 |
| `item_category` | 4 |
| `origin_zip` | 63,227 |
| `destination_zip` | 63,322 |

`carrier_code` and `item_category` are low-cardinality and safe to one-hot encode directly. `origin_zip`/`destination_zip` have extremely high cardinality (~63k unique values each, close to the row count) -- one-hot encoding these directly would explode the feature space and mostly memorize noise. They are excluded from the model features for this iteration, retained only as identifiers for traceability.

## Target Distribution (`actual_delivery_days`)
- Mean: 5.95 days, median: 5.95 days, std: 2.15
- Range: 0.5 to 13.63 days
- Roughly symmetric, no heavy skew -- a plain regression target, no log-transform obviously needed.

## Delivery Days by Carrier
| Carrier | Mean Delivery Days |
|---|---|
| CARRIER_A | 5.12 |
| CARRIER_B | 5.59 |
| CARRIER_C | 5.88 |
| CARRIER_D | 6.40 |
| CARRIER_E | 6.79 |

Clear, meaningful separation between carriers (~1.7 day spread) -- `carrier_code` is a useful signal.

## Delivery Days by Item Category
| Category | Mean Delivery Days |
|---|---|
| standard | 5.94 |
| fragile | 5.95 |
| perishable | 5.96 |
| oversize | 5.96 |

Essentially no difference between categories (spread of 0.02 days) -- `item_category` carries very little signal for this target. Still worth keeping as a feature, but shouldn't be expected to drive much predictive power.

## Distance vs. Target
Correlation coefficient: **0.80** -- the strongest single relationship in the dataset, confirming `distance_miles` as a core feature.

## Weather Impact
| Weather Severity | Mean Delivery Days |
|---|---|
| low (0-3) | 5.27 |
| medium (3-7) | 5.97 |
| high (7-10) | 6.64 |

Clear monotonic relationship -- worse weather associated with longer delivery times, roughly +1.4 days from low to high severity.

## Holiday Impact
| Holiday Flag | Mean Delivery Days |
|---|---|
| No (0) | 5.87 |
| Yes (1) | 6.89 |

Holidays add roughly 1 day on average -- a meaningful signal despite being a small fraction of orders (~8%).

## Warehouse Processing Impact
Correlation with target: **0.40** -- real but more moderate than distance.

## Outlier Ranges (IQR method)
| Field | IQR Bounds | Outlier Count |
|---|---|---|
| `distance_miles` | [-1499.76, 4497.14] | 0 |
| `warehouse_processing_hours` | [-34.30, 107.30] | 0 |
| `actual_delivery_days` | [-0.36, 12.28] | 16 |

Only the target has any IQR-flagged outliers (16/100,000, all long-tail deliveries, not data errors). No outlier handling applied -- the model should learn from these.

## Key Takeaways for Feature Engineering
1. **Strong signals:** `distance_miles` (r=0.80), `carrier_code`, `weather_score`, `holiday_flag`.
2. **Moderate signal:** `warehouse_processing_hours` (r=0.40).
3. **Weak signal:** `item_category` -- keep, don't expect much.
4. **Exclude from modeling:** `origin_zip`, `destination_zip` (too high-cardinality), `order_id` (identifier only).

## Target Leakage Risks Identified
- `actual_delivery_days` is the target itself and must never be an input feature.
- No other columns in this dataset represent post-delivery information -- clean synthetic data by construction. In a real dataset, any field only known after delivery completes (actual delivery timestamp, customer-confirmed receipt, post-delivery survey data) would need the same exclusion treatment. See `docs/feature_contract.md` for the enforced leakage-column list.
- IDs (`order_id`) are retained for traceability but excluded from model fitting.