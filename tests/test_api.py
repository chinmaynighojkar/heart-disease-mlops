"""API tests: health, a valid prediction, and input validation."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PATIENT = {
    "age": 54,
    "sex": 1,
    "chest_pain_type": 3,
    "resting_bp_s": 150,
    "cholesterol": 195,
    "fasting_blood_sugar": 0,
    "resting_ecg": 0,
    "max_heart_rate": 122,
    "exercise_angina": 0,
    "oldpeak": 0.0,
    "st_slope": 1,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_returns_expected_keys():
    resp = client.post("/predict", json=VALID_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"risk_score", "prediction", "shap_values", "top_risk_factors"}
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert len(body["shap_values"]) == 11
    assert len(body["top_risk_factors"]) == 3


def test_predict_missing_field_returns_422():
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "cholesterol"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422
