"""Train the heart-disease risk classifier and save serving artifacts.

Compares RandomForest / GradientBoosting / XGBoost by 5-fold ROC-AUC,
tunes the winner with GridSearchCV, and saves the fitted model plus the
metadata the API and monitoring layers depend on.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python src/train.py`

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.preprocessing import apply_imputation, fit_impute_values

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "heart.csv"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

# Raw CSV headers -> clean snake_case names used everywhere downstream
# (Pydantic fields, SHAP, Evidently). Order defines the model's feature order.
COLUMN_MAP = {
    "age": "age",
    "sex": "sex",
    "chest pain type": "chest_pain_type",
    "resting bp s": "resting_bp_s",
    "cholesterol": "cholesterol",
    "fasting blood sugar": "fasting_blood_sugar",
    "resting ecg": "resting_ecg",
    "max heart rate": "max_heart_rate",
    "exercise angina": "exercise_angina",
    "oldpeak": "oldpeak",
    "ST slope": "st_slope",
    "target": "target",
}
FEATURES = [v for k, v in COLUMN_MAP.items() if v != "target"]

RANDOM_STATE = 42

# Candidate models and the grids used to tune whichever wins CV.
CANDIDATES = {
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "xgboost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_estimators=200,
    ),
}
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 6, 12],
        "min_samples_leaf": [1, 2],
    },
    "gradient_boosting": {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    },
    "xgboost": {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV).rename(columns=COLUMN_MAP)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def save_reliability_plot(y_true, proba_uncal, proba_cal) -> None:
    """Reliability diagram comparing raw vs calibrated probabilities."""
    DOCS.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for proba, label in [(proba_uncal, "Uncalibrated"), (proba_cal, "Calibrated (sigmoid)")]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, "o-", label=label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram (test set)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(DOCS / "reliability.png", dpi=110)
    plt.close()


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    PROCESSED.mkdir(exist_ok=True)

    df = load_data()
    X, y = df[FEATURES], df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Fit zero-as-missing imputation on the training split only, then apply to
    # both splits (and save it so the API can reproduce it at serve time).
    impute_values = fit_impute_values(X_train)
    X_train = apply_imputation(X_train, impute_values)
    X_test = apply_imputation(X_test, impute_values)

    # 5-fold ROC-AUC on the training split; pick the strongest baseline.
    cv_scores = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        cv_scores[name] = float(scores.mean())
        print(f"{name:18s} 5-fold ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest baseline: {best_name} (ROC-AUC {cv_scores[best_name]:.4f}) -> tuning")

    grid = GridSearchCV(
        CANDIDATES[best_name],
        PARAM_GRIDS[best_name],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    base_model = grid.best_estimator_  # tuned tree model, used for SHAP explanations
    print(f"Best params: {grid.best_params_}")

    # Calibrate the probabilities (Platt/sigmoid, suited to this small sample).
    # RandomForest votes are not well-calibrated, so the raw predict_proba is a
    # ranking rather than a true probability. The calibrated model is what we
    # serve for `risk_score`; the base tree model is kept only for SHAP.
    calibrated = CalibratedClassifierCV(clone(base_model), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluate the calibrated model on the held-out test set.
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }

    # Quantify the calibration improvement (lower Brier score is better).
    y_proba_uncal = base_model.predict_proba(X_test)[:, 1]
    calibration = {
        "method": "sigmoid",
        "brier_uncalibrated": round(brier_score_loss(y_test, y_proba_uncal), 4),
        "brier_calibrated": round(brier_score_loss(y_test, y_proba), 4),
    }
    save_reliability_plot(y_test, y_proba_uncal, y_proba)

    # Per-sex subgroup metrics on the test set (fairness signal for the model card).
    subgroup_metrics = {}
    for label, code in [("female", 0), ("male", 1)]:
        mask = (X_test["sex"] == code).to_numpy()
        if mask.sum() > 0 and y_test[mask].nunique() > 1:
            subgroup_metrics[label] = {
                "n": int(mask.sum()),
                "accuracy": round(accuracy_score(y_test[mask], y_pred[mask]), 4),
                "roc_auc": round(roc_auc_score(y_test[mask], y_proba[mask]), 4),
                "recall": round(recall_score(y_test[mask], y_pred[mask]), 4),
                "positive_rate": round(float(y_test[mask].mean()), 4),
            }

    joblib.dump(calibrated, MODELS / "classifier.joblib")
    joblib.dump(base_model, MODELS / "shap_model.joblib")
    joblib.dump(FEATURES, MODELS / "feature_names.joblib")
    joblib.dump(impute_values, MODELS / "impute_values.joblib")

    training_metrics = {
        "model_version": "1.2",
        "model_type": best_name,
        "best_params": grid.best_params_,
        "cv_roc_auc": {k: round(v, 4) for k, v in cv_scores.items()},
        "test_metrics": metrics,
        "calibration": calibration,
        "subgroup_metrics_by_sex": subgroup_metrics,
        "impute_values": impute_values,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
    }
    (MODELS / "training_metrics.json").write_text(json.dumps(training_metrics, indent=2))

    # Reference distribution (training data) for Evidently drift detection.
    reference = X_train.copy()
    reference["target"] = y_train.values
    reference.to_csv(PROCESSED / "train_reference.csv", index=False)

    print("\nSaved artifacts to models/ and data/processed/train_reference.csv")
    print("Test metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
