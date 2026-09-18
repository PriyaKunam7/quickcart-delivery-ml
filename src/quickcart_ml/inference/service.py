"""
Prediction service layer.

Owns all business logic for producing a delivery-time estimate:
loading configuration, loading (or failing to load) the ML model,
deciding whether to use ML or the deterministic rule engine, and
returning a normalized domain result.

Route handlers in api/main.py should never contain this logic --
they only translate HTTP <-> this service's plain domain objects.
"""

import json
import os
from dataclasses import dataclass

import joblib
import pandas as pd

from quickcart_ml.config import get_inference_config
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


class ModelLoadError(Exception):
    """Raised internally when the configured model artifact can't be loaded."""


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

        self.model = None
        self.model_metadata = None
        self.model_load_error: str | None = None

        if self.inference_config.get("primary_source", "RULES") == "ML":
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
        ) as exc:  # noqa: BLE001 -- deliberately broad: any load failure must not crash startup
            self.model = None
            self.model_metadata = None
            self.model_load_error = f"{type(exc).__name__}: {exc}"

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def is_ready(self) -> tuple[bool, dict]:
        """
        Returns (ready, details). Readiness fails if the active config
        requires the ML model and it isn't loaded -- this makes a
        misconfigured or missing model artifact visible to operators,
        even though predict() itself will still serve traffic via the
        rule-engine fallback either way.
        """
        require_model = self.inference_config.get("require_model", False)
        details = {
            "rule_engine_loaded": True,  # rule config is loaded eagerly in __init__; failure would raise there
            "model_required": require_model,
            "model_loaded": self.model_loaded,
        }
        if self.model_load_error:
            details["model_load_error"] = self.model_load_error

        ready = (not require_model) or self.model_loaded
        return ready, details

    def predict(self, request: PredictionRequest) -> PredictionResult:
        use_ml = (
            self.inference_config.get("primary_source", "RULES") == "ML"
            and self.model_loaded
        )

        if use_ml:
            return self._predict_ml(request)
        return self._predict_rules(request)

    def _predict_ml(self, request: PredictionRequest) -> PredictionResult:
        row = request.model_dump(exclude={"order_id"})
        X = pd.DataFrame([row])
        prediction = float(self.model.predict(X)[0])

        model_version = (
            f"{self.model_metadata['model_name']}:{self.model_metadata['model_version']}"
            if self.model_metadata
            else "unknown"
        )
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
