# Model Promotion Policy

Defines the minimum criteria a candidate model must meet before being promoted from a training artifact to something served in production, and before the rule-engine fallback can be safely bypassed for a given segment.

## Promotion Criteria

| Criterion | Threshold | Measured Result (v1.0.0) | Status |
|---|---|---|---|
| Overall MAE improvement over rule-engine baseline | >= 10% | 46.1% | Pass |
| No carrier segment degradation | No carrier's MAE may be worse than the rule engine's MAE for that carrier | All 5 carriers improve (see reports/model_vs_rules.md) | Pass |
| No NaN predictions | 0 NaN values across all predictions | 0 (verified in tests/model/test_model_artifact.py) | Pass |
| Predictions within valid range | All predictions must be positive (delivery time cannot be zero or negative days) | Verified in tests | Pass |
| p95 inference latency | < 250 ms | Not yet measured -- no serving API exists yet. Deferred to the day the model is exposed via an endpoint. | Deferred |
| Fallback support | Model must support fallback to the deterministic rule engine | Yes -- quickcart_ml.rules.delivery_estimator remains fully independent and callable at any time, with no dependency on the model artifact | Pass |

## Rationale

- **10% MAE improvement threshold**: a smaller improvement wouldn't justify the added operational complexity, latency, and maintenance burden of running a trained model instead of pure deterministic rules. The v1.0.0 model clears this threshold by a wide margin (46.1%), but the bar is intentionally conservative for future model versions that might show more marginal gains.
- **No carrier segment may degrade**: an overall MAE improvement can hide a model that's actually worse for one specific carrier -- averaging can mask real harm to a subset of orders. This is checked per-segment specifically to catch that failure mode.
- **Zero tolerance for NaN or out-of-range predictions**: a single NaN or negative prediction reaching a customer-facing estimate is a worse failure than a merely inaccurate one -- it's a visible bug, not a modeling limitation. This is treated as a hard gate, not a metric to optimize.
- **p95 latency target deferred, not skipped**: this can't be honestly measured until the model is actually served behind an API (a future day's task). Documenting the target now, with an explicit "not yet measured" status, is more honest than fabricating a number or silently omitting the requirement.
- **Mandatory fallback support**: the platform must never be fully dependent on the ML model working. The rule engine has zero dependency on the model artifact and can always be called directly.

## Process

1. Every new model version must be evaluated with scripts/compare_models.py against the current rule-engine baseline before being considered for promotion.
2. reports/model_vs_rules.md must be regenerated and attached as evidence in the PR proposing promotion.
3. If any hard-gate criterion fails (NaN predictions, out-of-range predictions, any carrier segment degrading), the model must not be promoted regardless of overall MAE improvement.
4. Model artifacts are versioned (models/{model_name}/{version}/) -- a new version never overwrites an old one, so a promoted model can always be rolled back to the previous version.