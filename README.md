# Heart Disease Risk Model

![CI](https://github.com/chinmaynighojkar/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps project that predicts heart-disease risk from clinical
features and runs it as a production-style ML system: model training,
API serving with per-prediction explainability, data drift monitoring, a
monitoring dashboard, containerization, and automated tests in CI.

The pipeline trains and tunes a classifier, then serves it behind a FastAPI
endpoint that returns a risk score plus SHAP feature attributions for every
prediction and logs each request. Evidently AI compares those live requests
against the training baseline to detect input drift, and a Streamlit dashboard
surfaces model health, prediction trends, and drift status.

**Model:** a tuned RandomForest (selected over GradientBoosting and XGBoost by
5-fold cross-validation) reaches test ROC-AUC **0.93** and F1 **0.91** on the
combined UCI heart-disease dataset (11 clinical features, binary target).

## Features

- **Training pipeline** with model selection (RandomForest / GradientBoosting /
  XGBoost) by cross-validated ROC-AUC and GridSearch hyperparameter tuning.
- **Data-quality handling** for zero-coded missing values (a known leakage trap
  in this dataset), imputed consistently at train and serve time. See the
  [model card](docs/model_card.md) for details and a fairness analysis.
- **FastAPI serving** with Pydantic input validation, calibrated risk scores,
  and per-prediction SHAP explanations returning the top risk factors.
- **Prediction logging** to a structured store that feeds downstream monitoring.
- **Probability calibration** (Platt scaling) with a reliability diagram and
  Brier score, so `risk_score` is a calibrated probability rather than a raw
  tree vote.
- **Drift monitoring** with Evidently AI, comparing live inputs against the
  training baseline and producing HTML reports, plus a `--simulate` mode that
  demonstrates the report firing on a deliberately shifted population.
- **Streamlit dashboard** for model metrics, feature importance, prediction
  trends, and drift status.
- **Dockerized** service and **GitHub Actions CI** that trains the model and
  runs the test suite on every push.

## Architecture

```
Training (src/train.py)   → RF / GB / XGBoost, 5-fold CV + GridSearch
                            → models/classifier.joblib, training_metrics.json,
                              data/processed/train_reference.csv (drift baseline)

FastAPI (api/main.py)     → /predict: risk score + per-feature SHAP + top risk factors
                            → logs every request to monitoring/logs/predictions.jsonl

Evidently (src/monitor.py)→ drift report: live prediction inputs vs training baseline
                            → monitoring/reports/drift_report_*.html

Streamlit (dashboard/app.py) → model metrics, SHAP importance, prediction trends, drift

GitHub Actions            → trains the model and runs the test suite on every push
```

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                     # train + save artifacts
uvicorn api.main:app --reload           # serve the API at http://localhost:8000/docs
streamlit run dashboard/app.py          # launch the monitoring dashboard
python src/monitor.py                   # drift report from logged predictions
python src/monitor.py --simulate        # demo: drift report on a shifted population
```

### With Docker

```bash
docker build -t heart-disease-mlops .
docker run -p 8000:8000 heart-disease-mlops
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + model version |
| `/model-info` | GET | Training metrics and hyperparameters |
| `/predict` | POST | Risk score, prediction, SHAP values, top risk factors |
| `/monitoring/run-drift-report` | GET | Triggers an Evidently drift report |

Example prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"chest_pain_type":4,"resting_bp_s":150,"cholesterol":195,
       "fasting_blood_sugar":0,"resting_ecg":0,"max_heart_rate":122,
       "exercise_angina":1,"oldpeak":2.0,"st_slope":2}'
```

```json
{
  "risk_score": 0.96,
  "prediction": 1,
  "shap_values": {"st_slope": 0.1476, "chest_pain_type": 0.0881, "...": "..."},
  "top_risk_factors": ["st_slope", "chest_pain_type", "exercise_angina"]
}
```

## Tests

```bash
pytest tests/ -v
```

Covers the health check, a valid prediction (response schema), and input
validation (422 on missing fields). CI runs training + tests on every push.

## Dataset

Combined UCI heart-disease dataset (`data/raw/heart.csv`, 1190 rows,
11 features, binary target). Initial exploratory data analysis and modeling
experiments are kept in `reference/baseline_classifier.ipynb`.

## Dashboard

![Monitoring dashboard](docs/dashboard.png)
