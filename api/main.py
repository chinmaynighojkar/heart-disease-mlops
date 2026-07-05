"""FastAPI serving layer for the heart-disease risk model.

Exposes health, model-info, and predict endpoints. Every prediction returns
per-feature SHAP contributions and is appended to a structured log that the
Evidently monitoring layer consumes.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.preprocessing import apply_imputation

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOG_PATH = ROOT / "monitoring" / "logs" / "predictions.jsonl"

model = joblib.load(MODELS / "classifier.joblib")
FEATURES = joblib.load(MODELS / "feature_names.joblib")
IMPUTE_VALUES = joblib.load(MODELS / "impute_values.joblib")
TRAINING_METRICS = json.loads((MODELS / "training_metrics.json").read_text())
explainer = shap.TreeExplainer(model)

app = FastAPI(title="Heart Disease Risk Model", version=TRAINING_METRICS["model_version"])


class PatientFeatures(BaseModel):
    """Clinical inputs for a single patient (snake_case matches the model)."""

    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    chest_pain_type: int = Field(..., ge=1, le=4)
    resting_bp_s: float = Field(..., ge=0)
    cholesterol: float = Field(..., ge=0)
    fasting_blood_sugar: int = Field(..., ge=0, le=1)
    resting_ecg: int = Field(..., ge=0, le=2)
    max_heart_rate: float = Field(..., gt=0)
    exercise_angina: int = Field(..., ge=0, le=1)
    oldpeak: float
    st_slope: int = Field(..., ge=0, le=3)


def _positive_class_shap(row: pd.DataFrame) -> np.ndarray:
    """Return a 1-D array of per-feature SHAP values for the positive class."""
    values = explainer.shap_values(row)
    values = np.asarray(values)
    if values.ndim == 3:  # (samples, features, classes)
        return values[0, :, -1]
    if values.ndim == 2:  # (samples, features), single output
        return values[0]
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def _log_prediction(features: dict, risk_score: float, prediction: int) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "risk_score": risk_score,
        "prediction": prediction,
    }
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": TRAINING_METRICS["model_version"]}


@app.get("/model-info")
def model_info():
    return TRAINING_METRICS


@app.get("/monitoring/run-drift-report")
def run_drift_report():
    from src.monitor import run_drift_report as _run

    return _run()


@app.post("/predict")
def predict(patient: PatientFeatures):
    # Apply the same zero-as-missing imputation the model was trained with.
    row = apply_imputation(pd.DataFrame([patient.model_dump()])[FEATURES], IMPUTE_VALUES)
    features = row.iloc[0].to_dict()

    risk_score = float(model.predict_proba(row)[0, 1])
    prediction = int(risk_score >= 0.5)

    shap_vec = _positive_class_shap(row)
    shap_values = {feat: round(float(val), 4) for feat, val in zip(FEATURES, shap_vec)}
    top_risk_factors = [
        feat for feat, _ in sorted(
            shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:3]
    ]

    _log_prediction(features, round(risk_score, 4), prediction)

    return {
        "risk_score": round(risk_score, 4),
        "prediction": prediction,
        "shap_values": shap_values,
        "top_risk_factors": top_risk_factors,
    }
