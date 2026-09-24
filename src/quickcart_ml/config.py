"""
Centralized configuration loader for quickcart_ml.

Loads the YAML config file matching the current environment (dev/qa/prod,
controlled by the APP_ENV environment variable), so nothing in the
codebase reads a config file directly -- everything goes through
load_config().
"""

import os
from functools import lru_cache

import yaml

CONFIG_DIR = "configs"


@lru_cache(maxsize=None)
def load_config(env: str | None = None) -> dict:
    """
    Load configs/{env}.yaml. If env is not given, uses the APP_ENV
    environment variable, defaulting to "dev".

    Cached per env value -- config files are treated as static for the
    lifetime of the process. If you need to pick up a config change
    without restarting, call load_config.cache_clear() first.
    """
    resolved_env = env or os.environ.get("APP_ENV", "dev")
    path = os.path.join(CONFIG_DIR, f"{resolved_env}.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No config file found for environment '{resolved_env}' at {path}"
        )

    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_rule_engine_config(env: str | None = None) -> dict:
    """Convenience accessor for just the rule_engine section."""
    config = load_config(env)
    if "rule_engine" not in config:
        raise KeyError(f"'rule_engine' section missing from config")
    return config["rule_engine"]


def get_inference_config(env: str | None = None) -> dict:
    """Convenience accessor for just the inference section."""
    config = load_config(env)
    if "inference" not in config:
        raise KeyError("'inference' section missing from config")
    return config["inference"]


def get_shadow_config(env: str | None = None) -> dict:
    config = load_config(env)
    if "shadow" not in config:
        raise KeyError("'shadow' section missing from config")
    return config["shadow"]
