"""Evidently AI drift monitoring.

Compares the feature distribution of recent live predictions against the
training baseline and writes an HTML report plus a short console summary.

Scope note: this detects *input* (data) drift only. Detecting *performance*
drift (a drop in accuracy) requires ground-truth outcomes for past
predictions, which this system does not collect. See docs/model_card.md.

Run `python src/monitor.py` to report on logged predictions, or
`python src/monitor.py --simulate` to demonstrate the report firing on a
deliberately shifted ("drifted") population.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CSV = ROOT / "data" / "processed" / "train_reference.csv"
LOG_PATH = ROOT / "monitoring" / "logs" / "predictions.jsonl"
REPORTS_DIR = ROOT / "monitoring" / "reports"
FEATURES = joblib.load(ROOT / "models" / "feature_names.joblib")


def load_reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_CSV)[FEATURES]


def load_current(n_recent: int | None = None) -> pd.DataFrame:
    """Load recent prediction inputs from the log into a feature DataFrame."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No prediction log at {LOG_PATH}")
    rows = [json.loads(line)["features"] for line in LOG_PATH.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Prediction log is empty. Make some /predict calls first.")
    df = pd.DataFrame(rows)[FEATURES]
    return df.tail(n_recent) if n_recent else df


def make_drifted_sample(reference: pd.DataFrame) -> pd.DataFrame:
    """Shift the reference distribution to emulate a changed patient population."""
    drifted = reference.sample(min(200, len(reference)), replace=True, random_state=1).copy()
    drifted["age"] = drifted["age"] + 18
    drifted["cholesterol"] = drifted["cholesterol"] + 80
    drifted["max_heart_rate"] = drifted["max_heart_rate"] - 35
    drifted["resting_bp_s"] = drifted["resting_bp_s"] + 25
    drifted["oldpeak"] = drifted["oldpeak"] + 2.0
    drifted["chest_pain_type"] = 4  # shift toward asymptomatic
    drifted["exercise_angina"] = 1  # shift toward angina present
    return drifted


def run_drift_report(current: pd.DataFrame | None = None, n_recent: int | None = None,
                     label: str = "drift") -> dict:
    reference = load_reference()
    if current is None:
        current = load_current(n_recent)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{label}_report_{timestamp}.html"
    report.save_html(str(report_path))

    drift = report.as_dict()["metrics"][0]["result"]
    return {
        "report": report_path.name,
        "n_current": int(len(current)),
        "drifted_features": int(drift["number_of_drifted_columns"]),
        "total_features": int(drift["number_of_columns"]),
        "dataset_drift": bool(drift["dataset_drift"]),
    }


def run_simulated_drift() -> dict:
    """Generate a deliberately drifted population and report on it (demo)."""
    return run_drift_report(current=make_drifted_sample(load_reference()), label="drift_sim")


def _print(summary: dict) -> None:
    print(f"Report saved: monitoring/reports/{summary['report']}")
    print(
        f"Drift: {summary['drifted_features']}/{summary['total_features']} features drifted "
        f"| dataset_drift={summary['dataset_drift']} (n_current={summary['n_current']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidently drift monitoring")
    parser.add_argument("--simulate", action="store_true",
                        help="Report on a synthetic drifted population instead of the log")
    args = parser.parse_args()
    _print(run_simulated_drift() if args.simulate else run_drift_report())


if __name__ == "__main__":
    main()
