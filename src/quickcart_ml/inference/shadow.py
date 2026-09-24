"""
Shadow mode: runs ML inference alongside the rule engine for a sampled
percentage of traffic, purely for comparison logging -- never affecting
the response returned to the caller.

Design decision worth being explicit about: when shadow.enabled is
true, the customer-facing decision is ALWAYS rules, regardless of
whatever inference.primary_source says. Shadow mode exists specifically
to validate an ML model BEFORE trusting it for real decisions -- it
would defeat the purpose if primary_source=ML could silently override
that safety guarantee. This is enforced in PredictionService.predict(),
not here; this module only handles sampling and comparison-event logic.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

logger = logging.getLogger("quickcart_ml.shadow")


@dataclass
class ShadowConfig:
    enabled: bool
    percentage: int  # 0-100
    sampling_key: str  # "order_id" or "request_id"

    @classmethod
    def from_dict(cls, raw: dict) -> "ShadowConfig":
        return cls(
            enabled=raw.get("enabled", False),
            percentage=raw.get("percentage", 0),
            sampling_key=raw.get("sampling_key", "request_id"),
        )


@dataclass
class ComparisonEvent:
    request_id: str
    sampling_key_used: str
    sampled_bucket: int
    rule_prediction: float
    ml_prediction: float
    difference: float
    model_version: str
    ml_latency_ms: float
    timestamp: str


def deterministic_bucket(key: str) -> int:
    """
    Maps a string key to a stable bucket in [0, 99], using a
    cryptographic hash rather than Python's built-in hash().

    Python's hash() is deliberately randomized per-process (for
    security, to prevent hash-flooding attacks) via PYTHONHASHSEED --
    the same string produces a DIFFERENT hash() value every time the
    process restarts. That would break the "same key always lands in
    the same bucket" requirement the moment the service restarts.
    hashlib.sha256 has no such randomization: the same input always
    produces the same digest, in this process or any other.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def is_sampled(key: str, percentage: int) -> bool:
    """
    True if `key` falls within the sampled percentage.

    Buckets 0..(percentage-1) are sampled. This makes sampling
    monotonic in percentage: a key sampled at 25% is guaranteed to
    still be sampled if the percentage is later raised to 50%, since
    its bucket never changes and 25% only ever grows the sampled
    range. This matters for a real gradual shadow-mode rollout -- keys
    don't randomly flip in and out of the sample as the percentage changes.
    """
    if percentage <= 0:
        return False
    if percentage >= 100:
        return True
    return deterministic_bucket(key) < percentage


def build_comparison_event(
    request_id: str,
    sampling_key_used: str,
    rule_prediction: float,
    ml_prediction: float,
    model_version: str,
    ml_latency_ms: float,
) -> ComparisonEvent:
    return ComparisonEvent(
        request_id=request_id,
        sampling_key_used=sampling_key_used,
        sampled_bucket=deterministic_bucket(sampling_key_used),
        rule_prediction=rule_prediction,
        ml_prediction=ml_prediction,
        difference=round(ml_prediction - rule_prediction, 4),
        model_version=model_version,
        ml_latency_ms=round(ml_latency_ms, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def emit_comparison_event(event: ComparisonEvent) -> None:
    """
    Logs the comparison event as a single structured JSON line.

    A real production system would likely publish this to a message
    queue or event stream for downstream analysis; a structured log
    line is a deliberately simple, dependency-free choice for local
    development that's still trivially parseable (one JSON object per
    line) if piped into a real event pipeline later.
    """
    logger.info(json.dumps(asdict(event)))
