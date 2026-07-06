# Project Plan: Heart Disease Risk Model

**Status:** Delivered (model v1.2). This is a living document, updated as the
system evolves; each milestone records its acceptance criteria and current state.

## 1. Objective

Deliver a heart-disease risk classifier as a small but complete production-style
ML system, not a notebook. The goal is to exercise the full model lifecycle:
training and selection, explainable serving, drift monitoring, a health
dashboard, containerization, and continuous integration, with reproducibility
and honest documentation of limitations throughout.

## 2. Scope and non-goals

**In scope**
- A trained, tuned, calibrated classifier with saved, reproducible artifacts.
- An HTTP serving layer with input validation and per-prediction explanations.
- Structured prediction logging that feeds monitoring.
- Input-drift monitoring with reports and a dashboard.
- Containerization and CI that reproduce the pipeline from a clean checkout.

**Non-goals (explicit)**
- Clinical validity. This is an educational system and is not a medical device.
- State-of-the-art accuracy. Effort is spent on lifecycle engineering, not on
  squeezing the last points of a leaderboard metric.
- Production-grade infrastructure (managed model registry, durable event store,
  authn/z, autoscaling). These are named in Future Work, not built here.

## 3. Architecture

```
Training (src/train.py)      select RF / GB / XGBoost by CV ROC-AUC, tune,
                             calibrate, and persist artifacts + metrics
        |
        v
Artifacts (models/)          classifier (calibrated), base tree (for SHAP),
                             feature names, imputation stats, metrics JSON
        |
        v
Serving (api/main.py)        FastAPI /predict: calibrated risk score,
                             per-feature SHAP, top risk factors; logs each request
        |
        v
Logs (monitoring/logs/)      predictions.jsonl (one record per request)
        |
        v
Monitoring (src/monitor.py)  Evidently input-drift report vs training baseline
                             (+ a simulate mode that deliberately triggers drift)
        |
        v
Dashboard (dashboard/app.py) model health, predictions, drift status
```

Cross-cutting: `src/preprocessing.py` is shared by training and serving to
guarantee identical input handling; Docker and GitHub Actions reproduce the
whole pipeline from source.

## 4. Milestones

Each milestone is considered done only when its acceptance criteria hold.

### M1. Foundation and data. [Done]
- Reproducible environment (pinned dependencies), project structure, dataset in
  place.
- **Acceptance:** clean dependency install; dataset loads to the expected schema.
- **Note:** the working dataset is the combined UCI set (11 features, binary
  target), which differs from the classic 13-feature Cleveland schema. The
  pipeline was built to the actual data.

### M2. Training and model selection. [Done]
- Compare RandomForest / GradientBoosting / XGBoost by 5-fold cross-validated
  ROC-AUC; tune the winner with grid search; evaluate once on a held-out set.
- **Acceptance:** artifacts and a metrics file are produced; test ROC-AUC is
  reported from an untouched test split. Current: RandomForest, test ROC-AUC ~0.93.

### M3. Data-quality correction. [Done]
- Handle zero-coded missing values (`cholesterol`, `resting_bp_s`) that otherwise
  leak a spurious signal; leave legitimately-zero fields (`oldpeak`) untouched.
- **Acceptance:** imputation is fit on training data only and applied identically
  at serve time; performance impact is measured and documented.

### M4. Serving with explainability. [Done]
- FastAPI endpoints with Pydantic validation; each prediction returns a
  calibrated risk score, per-feature SHAP values, and top risk factors, and is
  logged.
- **Acceptance:** valid input returns the documented schema; invalid input
  returns 422; predictions are logged.

### M5. Probability calibration. [Done]
- Calibrate probabilities so the risk score is a probability, not a raw tree vote.
- **Acceptance:** calibration is measured (Brier score, reliability diagram) and
  ROC-AUC is preserved (monotonic transform).

### M6. Drift monitoring. [Done]
- Compare live prediction inputs against the training baseline; produce reports;
  provide a way to demonstrate the detector firing.
- **Acceptance:** a report generates from logged predictions; a simulate mode
  produces dataset-level drift. Scope limit (input drift only) is documented.

### M7. Dashboard. [Done]
- Model overview (metrics, SHAP importance, calibration), recent predictions,
  and drift monitoring, in one app.
- **Acceptance:** all three views render from live artifacts and logs.

### M8. Containerization and CI. [Done]
- Dockerized service; GitHub Actions trains and tests on every push.
- **Acceptance:** container serves requests; CI is green from a clean checkout.

### M9. Documentation. [Done]
- README (architecture, quickstart, API, dashboard) and a model card (intended
  use, metrics, fairness analysis, limitations).
- **Acceptance:** a reader can run the system and understand its limits from the
  docs alone.

## 5. Key engineering decisions

- **Tree models, unscaled** so SHAP attributes importance in real clinical units.
- **Selection on cross-validated ROC-AUC** (threshold-free, imbalance-robust)
  rather than accuracy; only the winner is tuned.
- **Imputation shared between train and serve** to avoid leakage and train/serve
  skew.
- **Calibrated model for scoring, base tree kept for SHAP**, since calibration is
  monotonic and does not change the ranking SHAP explains.
- **Pinned dependencies and a fixed seed** so artifacts regenerate exactly;
  generated binaries are not committed.

## 6. Quality gates

- **Tests:** API health, prediction-response schema, input validation (422), and
  that simulated drift fires.
- **CI:** installs pinned deps, trains the model, and runs the test suite on
  every push, proving the pipeline reproduces from source.

## 7. Known limitations

- Prediction logging is a local JSONL file: not concurrency-safe and not durable
  in an ephemeral container.
- Monitoring covers input drift only; performance drift needs ground-truth
  labels, which the system does not collect.
- The dataset is small and heavily male-skewed; subgroup metrics (see the model
  card) are not reliable for the female subgroup.
- Training runs inside the Docker build and inside CI: convenient for a small
  project, but not how training and serving should be separated at scale.

## 8. Future work

- Replace the JSONL sink with a database or event stream behind the existing
  logging interface.
- Introduce a model registry and versioned artifacts; separate the training
  pipeline from the serving image.
- Add a labelling/feedback loop to enable performance (not just input) drift
  monitoring.
- Add authentication and rate limiting to the API.
- Tune the decision threshold to reflect the real cost of a false negative rather
  than defaulting to 0.5.
