"""Train and export a shirt-size predictor model.

This project originally shipped with an Iris classifier. This script replaces it with a
shirt sizing model that predicts one of: S, M, L, XL, XXL.

The dataset is synthetic (generated) but designed to be plausible by deriving a
"chest circumference" proxy from height/weight/gender/build and mapping it to sizes.

Run:
  python train_model.py

Output:
  app/model.joblib
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SIZES: Tuple[str, ...] = ("S", "M", "L", "XL", "XXL")

from app.feature_utils import FEATURE_ORDER, rows_to_matrix


@dataclass(frozen=True)
class ModelArtifact:
    version: str
    trained_at: str
    labels: List[str]
    feature_schema: Dict[str, Any]
    pipeline: Any
    metrics: Dict[str, Any]


def _clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(a, lo), hi)


def _size_from_chest_cm(chest_cm: np.ndarray) -> np.ndarray:
    """Map a chest measurement to a size bucket.

    These thresholds are deliberately simple and should be adjusted once you have
    real-world sizing data from your merch provider.
    """
    # Rough unisex tee thresholds
    bins = np.array([0.0, 96.0, 104.0, 112.0, 122.0, 10_000.0])
    idx = np.digitize(chest_cm, bins=bins, right=False) - 1
    idx = _clip(idx, 0, len(SIZES) - 1).astype(int)
    return np.array(SIZES, dtype=object)[idx]


def generate_synthetic_dataset(n: int = 25_000, seed: int = 42) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    rng = np.random.default_rng(seed)

    # Discrete attributes
    genders = np.array(["female", "male", "other"], dtype=object)
    gender = rng.choice(genders, size=n, p=[0.48, 0.48, 0.04])

    fits = np.array(["slim", "regular", "oversized"], dtype=object)
    fit_preference = rng.choice(fits, size=n, p=[0.25, 0.60, 0.15])

    builds = np.array(["lean", "average", "athletic", "curvy"], dtype=object)
    build = rng.choice(builds, size=n, p=[0.22, 0.50, 0.18, 0.10])

    # Continuous attributes
    age = rng.integers(13, 75, size=n)

    # Height distribution (cm) with slight gender conditioning
    base_height = rng.normal(loc=170.0, scale=10.0, size=n)
    base_height = base_height + np.where(gender == "male", 5.0, 0.0) - np.where(gender == "female", 3.0, 0.0)
    height_cm = _clip(base_height, 145.0, 205.0)

    # Weight distribution (kg) conditioned on height and build
    bmi = rng.normal(loc=24.0, scale=4.0, size=n)
    bmi = bmi + np.where(build == "lean", -2.0, 0.0) + np.where(build == "athletic", 1.5, 0.0) + np.where(build == "curvy", 2.5, 0.0)
    bmi = _clip(bmi, 16.0, 45.0)

    height_m = height_cm / 100.0
    weight_kg = bmi * (height_m**2)
    weight_kg = _clip(weight_kg + rng.normal(0, 2.5, size=n), 40.0, 150.0)

    # Chest proxy model
    # Start with a baseline proportional to height + weight, then adjust for gender/build.
    chest_cm = 0.38 * height_cm + 0.55 * weight_kg
    chest_cm += np.where(gender == "male", 4.0, 0.0) + np.where(gender == "female", -2.0, 0.0)
    chest_cm += np.where(build == "athletic", 3.5, 0.0) + np.where(build == "lean", -2.0, 0.0)
    chest_cm += rng.normal(0, 3.0, size=n)

    # Fit preference adjustment (what the user would feel comfortable ordering)
    # Slim: bias slightly smaller; Oversized: bias larger.
    chest_cm += np.where(fit_preference == "slim", -3.0, 0.0)
    chest_cm += np.where(fit_preference == "oversized", 4.0, 0.0)

    y = _size_from_chest_cm(chest_cm)

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "height_cm": float(height_cm[i]),
                "weight_kg": float(weight_kg[i]),
                "age": int(age[i]),
                "gender": str(gender[i]),
                "fit_preference": str(fit_preference[i]),
                "build": str(build[i]),
            }
        )

    return rows, y


def train_model(rows: List[Dict[str, Any]], y: np.ndarray, seed: int = 42) -> Tuple[Pipeline, Dict[str, Any]]:
    # Train/test split
    indices = np.arange(len(rows))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y)
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_train_rows = [rows[i] for i in train_idx]
    X_test_rows = [rows[i] for i in test_idx]

    to_matrix_tf = FunctionTransformer(rows_to_matrix, validate=False)

    numeric_idx = [0, 1, 2]
    categorical_idx = [3, 4, 5]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_idx),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_idx,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        n_jobs=None,
        class_weight=None,
        random_state=seed,
    )

    pipeline = Pipeline(steps=[("to_matrix", to_matrix_tf), ("preprocess", preprocessor), ("clf", clf)])

    pipeline.fit(X_train_rows, y_train)

    y_pred = pipeline.predict(X_test_rows)
    acc = float(accuracy_score(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "classification_report": report,
        "n_samples": int(len(rows)),
        "seed": int(seed),
    }

    return pipeline, metrics


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    out_path = repo_root / "app" / "model.joblib"

    rows, y = generate_synthetic_dataset(n=30_000, seed=42)
    pipeline, metrics = train_model(rows, y, seed=42)

    artifact = ModelArtifact(
        version="shirt-size-v1",
        trained_at=datetime.now(timezone.utc).isoformat(),
        labels=list(SIZES),
        feature_schema={
            "type": "object",
            "required": ["height_cm", "weight_kg", "age", "gender", "fit_preference", "build"],
            "properties": {
                "height_cm": {"type": "number", "minimum": 120, "maximum": 230},
                "weight_kg": {"type": "number", "minimum": 30, "maximum": 250},
                "age": {"type": "integer", "minimum": 10, "maximum": 100},
                "gender": {"type": "string", "enum": ["female", "male", "other"]},
                "fit_preference": {"type": "string", "enum": ["slim", "regular", "oversized"]},
                "build": {"type": "string", "enum": ["lean", "average", "athletic", "curvy"]},
            },
        },
        pipeline=pipeline,
        metrics=metrics,
    )

    joblib.dump(asdict(artifact), out_path)

    print(f"Wrote model artifact to: {out_path}")
    print(f"Accuracy (synthetic holdout): {metrics['accuracy']:.4f}")
    # Keep stdout readable: print a compact summary
    per_class = metrics["classification_report"]
    summary = {k: per_class[k] for k in SIZES if k in per_class}
    print(json.dumps({"per_class": summary}, indent=2))


if __name__ == "__main__":
    main()
