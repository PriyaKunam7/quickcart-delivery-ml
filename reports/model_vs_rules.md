# Model vs. Rules Comparison Report

## Overall Metrics
| Segment | n | ML MAE | Rules MAE | ML RMSE | Rules RMSE | ML Improvement |
|---|---|---|---|---|---|---|
| Overall | 20000 | 0.387 | 0.717 | 0.486 | 0.886 | +46.1% |

**ML improves MAE over rules by 46.1% overall.**

## By Carrier
| Segment | n | ML MAE | Rules MAE | ML RMSE | Rules RMSE | ML Improvement |
|---|---|---|---|---|---|---|
| CARRIER_A | 4027 | 0.361 | 0.723 | 0.450 | 0.900 | +50.1% |
| CARRIER_B | 3998 | 0.404 | 0.717 | 0.506 | 0.879 | +43.6% |
| CARRIER_C | 3925 | 0.407 | 0.681 | 0.512 | 0.845 | +40.3% |
| CARRIER_D | 4049 | 0.399 | 0.768 | 0.501 | 0.946 | +48.1% |
| CARRIER_E | 4001 | 0.364 | 0.695 | 0.457 | 0.857 | +47.7% |

## By Distance Band
| Segment | n | ML MAE | Rules MAE | ML RMSE | Rules RMSE | ML Improvement |
|---|---|---|---|---|---|---|
| 0-249 | 1576 | 0.431 | 0.617 | 0.539 | 0.760 | +30.2% |
| 250-749 | 3381 | 0.391 | 0.784 | 0.491 | 0.950 | +50.2% |
| 750-1499 | 4978 | 0.365 | 0.708 | 0.460 | 0.876 | +48.5% |
| 1500-2249 | 5092 | 0.384 | 0.640 | 0.483 | 0.797 | +40.1% |
| 2250+ | 4973 | 0.395 | 0.790 | 0.493 | 0.973 | +50.0% |

## By Item Category
| Segment | n | ML MAE | Rules MAE | ML RMSE | Rules RMSE | ML Improvement |
|---|---|---|---|---|---|---|
| fragile | 4921 | 0.388 | 0.727 | 0.487 | 0.897 | +46.6% |
| oversize | 4887 | 0.390 | 0.711 | 0.490 | 0.882 | +45.2% |
| perishable | 5080 | 0.385 | 0.713 | 0.485 | 0.878 | +45.9% |
| standard | 5112 | 0.384 | 0.717 | 0.483 | 0.889 | +46.5% |

## Win Rate
- ML closer to actual: **71.5%** of orders
- Rules closer to actual: **28.5%** of orders
- Tied (identical error): 0.0%

## Worst 20 Disagreements (largest |ML - Rules| gap)
| order_id | carrier | distance_miles | actual | ml_pred | rules_pred | disagreement |
|---|---|---|---|---|---|---|
| 89577 | CARRIER_D | 430.9 | 3.48 | 2.07 | 5.00 | 2.93 |
| 89832 | CARRIER_D | 795.3 | 4.60 | 4.08 | 7.00 | 2.92 |
| 9527 | CARRIER_D | 760.5 | 4.66 | 4.16 | 7.00 | 2.84 |
| 74807 | CARRIER_B | 785.5 | 3.06 | 3.16 | 6.00 | 2.84 |
| 66069 | CARRIER_D | 414.8 | 3.85 | 3.26 | 6.00 | 2.74 |
| 13520 | CARRIER_A | 2861.9 | 8.21 | 8.73 | 6.00 | 2.73 |
| 5555 | CARRIER_C | 816.3 | 6.02 | 4.30 | 7.00 | 2.70 |
| 17089 | CARRIER_D | 568.9 | 5.11 | 4.30 | 7.00 | 2.70 |
| 78049 | CARRIER_D | 751.4 | 3.42 | 3.32 | 6.00 | 2.68 |
| 9574 | CARRIER_C | 754.8 | 4.78 | 3.32 | 6.00 | 2.68 |
| 72335 | CARRIER_D | 634.5 | 4.49 | 3.33 | 6.00 | 2.67 |
| 99356 | CARRIER_D | 827.9 | 4.52 | 4.40 | 7.00 | 2.60 |
| 95800 | CARRIER_E | 753.6 | 5.40 | 4.40 | 7.00 | 2.60 |
| 12683 | CARRIER_E | 774.0 | 4.78 | 4.45 | 7.00 | 2.55 |
| 27716 | CARRIER_D | 778.7 | 4.56 | 4.48 | 7.00 | 2.52 |
| 77691 | CARRIER_B | 756.3 | 3.86 | 3.49 | 6.00 | 2.51 |
| 86986 | CARRIER_A | 2901.4 | 8.38 | 8.49 | 6.00 | 2.49 |
| 18648 | CARRIER_B | 828.5 | 2.99 | 3.52 | 6.00 | 2.48 |
| 27744 | CARRIER_D | 787.0 | 5.44 | 4.52 | 7.00 | 2.48 |
| 88236 | CARRIER_A | 2992.1 | 9.83 | 9.47 | 7.00 | 2.47 |