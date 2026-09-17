"""
Trains the first delivery-time regression model, combining feature
engineering and the model itself into a single sklearn Pipeline that
gets serialized as one artifact -- so there is never a risk of the
preprocessing and the model drifting apart (e.g. someone updating
build_features.py without retraining, or loading a model with
mismatched preprocessing).

Usage:
    python -m quickcart_ml.training.train
"""

import json
import os
import subprocess

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from quickcart_ml.features.build_features import build_preprocessing_pipeline

DATA_PATH = "data/raw/orders.csv"
MODEL_NAME = "delivery_regressor"
MODEL_VERSION = "1.0.0"
MODEL_DIR = os.path.join("models", MODEL_NAME, MODEL_VERSION)
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_git_commit() -> str:
    """Returns the current git SHA, or 'unknown' outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"origin_zip": str, "destination_zip": str},
        parse_dates=["order_timestamp"],
    )


def build_model_pipeline() -> Pipeline:
    """The full pipeline: feature engineering + model, one object."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train() -> dict:
    df = load_dataset()

    feature_columns = [c for c in df.columns if c != "actual_delivery_days"]
    X = df[feature_columns]
    y = df["actual_delivery_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "algorithm": "RandomForestRegressor",
        "training_dataset": "synthetic_orders_v1",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        },
        "git_commit": get_git_commit(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main() -> None:
    metadata = train()
    print(f"Model trained: {metadata['model_name']} v{metadata['model_version']}")
    print(f"MAE: {metadata['metrics']['mae']}  RMSE: {metadata['metrics']['rmse']}")
    print(f"Artifact: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Git commit: {metadata['git_commit']}")


if __name__ == "__main__":
    main()
