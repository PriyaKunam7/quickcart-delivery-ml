"""
Prediction service layer.

Owns all business logic for producing a delivery-time estimate:
loading configuration, loading (or failing to load) the ML model,
deciding whether to use ML or the deterministic rule engine, and
returning a normalized domain result.

Shadow mode: when shadow.enabled is true, the returned decision is
ALWAYS rules, regardless of inference.primary_source. ML still runs
for sampled requests, but purely to produce a comparison event -- it
never influences the response. This is what "shadow" means: validating
a model's behavior against real traffic before trusting it for real
decisions.

Route handlers in api/main.py should never contain this logic -- they
only translate HTTP <-> this service's plain domain objects.
"""

import json
import os
import time
from dataclasses import dataclass

import joblib
import pandas as pd

from quickcart_ml.config import get_inference_config, get_shadow_config
from quickcart_ml.inference.shadow import (
    ShadowConfig,
    build_comparison_event,
    emit_comparison_event,
    is_sampled,
)
from quickcart_ml.rules.delivery_estimator import (
    estimate_delivery_days,
    load_rule_engine_config,
)
from quickcart_ml.validation.schemas import PredictionRequest

RULE_ENGINE_MODEL_VERSION = "rule-engine-v1"


@dataclass
class PredictionResult:
    estimated_delivery_days: float
    decision_source: str  # "RULES" or "ML"
    model_version: str


class PredictionService:
    """
    Loads its dependencies once (at construction / app startup) and
    serves predictions from then on. Never raises out of predict() due
    to the ML model being unavailable -- it always falls back to the
    rule engine, which has no dependency on the model artifact at all.
    """

    def __init__(self, env: str | None = None):
        self.env = env
        self.inference_config = get_inference_config(env)
        self.rule_config = load_rule_engine_config(env)
        self.shadow_config = ShadowConfig.from_dict(get_shadow_config(env))

        self.model = None
        self.model_metadata = None
        self.model_load_error: str | None = None

        # The model is loaded whenever ML might be needed for the live
        # decision OR for shadow comparisons -- both cases need it.
        if (
            self.inference_config.get("primary_source", "RULES") == "ML"
            or self.shadow_config.enabled
        ):
            self._try_load_model()

    def _model_paths(self) -> tuple[str, str]:
        model_name = self.inference_config["model_name"]
        model_version = self.inference_config["model_version"]
        base = os.path.join("models", model_name, model_version)
        return os.path.join(base, "model.joblib"), os.path.join(base, "metadata.json")

    def _try_load_model(self) -> None:
        model_path, metadata_path = self._model_paths()
        try:
            self.model = joblib.load(model_path)
            with open(metadata_path) as f:
                self.model_metadata = json.load(f)
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- any load failure must not crash startup
            self.model = None
            self.model_metadata = None
            self.model_load_error = f"{type(exc).__name__}: {exc}"

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def is_ready(self) -> tuple[bool, dict]:
        require_model = self.inference_config.get("require_model", False)
        details = {
            "rule_engine_loaded": True,
            "model_required": require_model,
            "model_loaded": self.model_loaded,
            "shadow_enabled": self.shadow_config.enabled,
        }
        if self.model_load_error:
            details["model_load_error"] = self.model_load_error

        ready = (not require_model) or self.model_loaded
        return ready, details

    # ------------------------------------------------------------------
    # Live decision
    # ------------------------------------------------------------------
    def predict(self, request: PredictionRequest) -> PredictionResult:
        # Shadow mode is a hard override: while it's active, the
        # returned decision is always rules, no matter what
        # primary_source says. Shadow's entire purpose is validating ML
        # before it's trusted for real decisions -- primary_source=ML
        # silently winning here would defeat that.
        if self.shadow_config.enabled:
            return self._predict_rules(request)

        use_ml = (
            self.inference_config.get("primary_source", "RULES") == "ML"
            and self.model_loaded
        )
        if use_ml:
            return self._predict_ml(request)
        return self._predict_rules(request)

    def _predict_ml(self, request: PredictionRequest) -> PredictionResult:
        prediction, _latency_ms = self._run_ml_inference(request)
        model_version = self._model_version_string()
        return PredictionResult(
            estimated_delivery_days=round(prediction, 2),
            decision_source="ML",
            model_version=model_version,
        )

    def _predict_rules(self, request: PredictionRequest) -> PredictionResult:
        estimate = estimate_delivery_days(
            distance_miles=request.distance_miles,
            warehouse_processing_hours=request.warehouse_processing_hours,
            weather_score=request.weather_score,
            holiday_flag=request.holiday_flag,
            carrier_code=request.carrier_code,
            config=self.rule_config,
        )
        return PredictionResult(
            estimated_delivery_days=float(estimate),
            decision_source="RULES",
            model_version=RULE_ENGINE_MODEL_VERSION,
        )

    def _model_version_string(self) -> str:
        if not self.model_metadata:
            return "unknown"
        return f"{self.model_metadata['model_name']}:{self.model_metadata['model_version']}"

    def _run_ml_inference(self, request: PredictionRequest) -> tuple[float, float]:
        """Returns (prediction, latency_ms). Shared by live ML and shadow ML."""
        row = request.model_dump(exclude={"order_id"})
        X = pd.DataFrame([row])
        start = time.perf_counter()
        prediction = float(self.model.predict(X)[0])
        latency_ms = (time.perf_counter() - start) * 1000
        return prediction, latency_ms

    # ------------------------------------------------------------------
    # Shadow mode
    # ------------------------------------------------------------------
    def _shadow_sampling_key(self, request: PredictionRequest, request_id: str) -> str:
        """
        Which value gets hashed for bucketing. If sampling_key is
        "order_id" but the request didn't supply one (order_id is
        optional on PredictionRequest), this falls back to request_id.

        This is a documented, deliberate degradation, not a bug: the
        stability guarantee ("same key -> same bucket across repeated
        calls") only has meaning when there's a stable identity to
        hash. Without a supplied order_id, each call is a logically
        distinct request anyway, so request-level determinism (this
        exact request always hashes to this exact bucket) is the
        strongest guarantee actually available.
        """
        if (
            self.shadow_config.sampling_key == "order_id"
            and request.order_id is not None
        ):
            return str(request.order_id)
        return request_id

    def should_shadow_sample(self, request: PredictionRequest, request_id: str) -> bool:
        if not self.shadow_config.enabled or not self.model_loaded:
            return False
        key = self._shadow_sampling_key(request, request_id)
        return is_sampled(key, self.shadow_config.percentage)

    def run_shadow_comparison(
        self, request: PredictionRequest, request_id: str, rule_result: PredictionResult
    ) -> None:
        """
        Runs ML for a sampled request and emits a comparison event.
        Never raises: a failure here must never be allowed to affect
        anything, since by the time this runs the live response has
        already been sent to the caller.
        """
        try:
            ml_prediction, ml_latency_ms = self._run_ml_inference(request)
            sampling_key = self._shadow_sampling_key(request, request_id)
            event = build_comparison_event(
                request_id=request_id,
                sampling_key_used=sampling_key,
                rule_prediction=rule_result.estimated_delivery_days,
                ml_prediction=ml_prediction,
                model_version=self._model_version_string(),
                ml_latency_ms=ml_latency_ms,
            )
            emit_comparison_event(event)
        except Exception:  # noqa: BLE001 -- shadow failures must never propagate
            import logging

            logging.getLogger("quickcart_ml.shadow").exception(
                "Shadow comparison failed for request_id=%s", request_id
            )
