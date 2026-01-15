from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "app" / "model.joblib"


def _load_artifact() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` from the repo root to generate it."
        )
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid model artifact format. Expected a dict with a 'pipeline' key.")
    return artifact


_ARTIFACT = _load_artifact()
_PIPELINE = _ARTIFACT["pipeline"]

SizeLabel = Literal["S", "M", "L", "XL", "XXL"]
Gender = Literal["female", "male", "other"]
FitPreference = Literal["slim", "regular", "oversized"]
Build = Literal["lean", "average", "athletic", "curvy"]


class PredictRequest(BaseModel):
    height_cm: float = Field(..., ge=120, le=230)
    weight_kg: float = Field(..., ge=30, le=250)
    age: int = Field(..., ge=10, le=100)
    gender: Gender
    fit_preference: FitPreference = "regular"
    build: Build = "average"


class PredictResponse(BaseModel):
    recommended_size: SizeLabel
    confidence: float
    probabilities: Dict[SizeLabel, float]
    model_version: str


app = FastAPI(title="MEM Shirt Size Predictor", version=str(_ARTIFACT.get("version", "unknown")))


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Shirt size prediction API", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": _ARTIFACT.get("version", "unknown"),
        "trained_at": _ARTIFACT.get("trained_at"),
    }


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    return {
        "model_version": _ARTIFACT.get("version", "unknown"),
        "trained_at": _ARTIFACT.get("trained_at"),
        "labels": _ARTIFACT.get("labels", ["S", "M", "L", "XL", "XXL"]),
        "feature_schema": _ARTIFACT.get("feature_schema"),
        "metrics": _ARTIFACT.get("metrics"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: Dict[str, Any]) -> PredictResponse:
    """Predict a user's shirt size.

    Preferred payload:
      {
        "height_cm": 175,
        "weight_kg": 75,
        "age": 24,
        "gender": "male",
        "fit_preference": "regular",
        "build": "average"
      }

    Backward-compatible payload (legacy demo clients):
      {"features": [height_cm, weight_kg, age, gender, fit_preference, build]}
    """

    req: Optional[PredictRequest] = None

    if "features" in payload:
        features = payload.get("features")
        if not isinstance(features, list) or len(features) != 6:
            raise HTTPException(status_code=400, detail="'features' must be a list of 6 values")
        try:
            req = PredictRequest(
                height_cm=float(features[0]),
                weight_kg=float(features[1]),
                age=int(features[2]),
                gender=str(features[3]),
                fit_preference=str(features[4]),
                build=str(features[5]),
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid 'features' payload: {e}")
    else:
        try:
            req = PredictRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    row = req.model_dump() if hasattr(req, "model_dump") else req.dict()

    try:
        proba = _PIPELINE.predict_proba([row])[0]
        labels = list(_PIPELINE.classes_)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    probs: Dict[SizeLabel, float] = {}
    for label, p in zip(labels, proba):
        # Enforce the known label space
        if label in ("S", "M", "L", "XL", "XXL"):
            probs[label] = float(p)

    if not probs:
        raise HTTPException(status_code=500, detail="Model returned unexpected label set")

    recommended = max(probs.items(), key=lambda kv: kv[1])[0]
    confidence = float(probs[recommended])

    return PredictResponse(
        recommended_size=recommended,  # type: ignore[arg-type]
        confidence=confidence,
        probabilities=probs,  # type: ignore[arg-type]
        model_version=str(_ARTIFACT.get("version", "unknown")),
    )

