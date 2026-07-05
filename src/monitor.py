"""Evidently AI drift monitoring.

Compares the feature distribution of recent live predictions against the
training baseline and writes an HTML report plus a short console summary.
"""

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


def load_current(n_recent: int | None = None) -> pd.DataFrame:
    """Load recent prediction inputs from the log into a feature DataFrame."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No prediction log at {LOG_PATH}")
    rows = [json.loads(line)["features"] for line in LOG_PATH.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Prediction log is empty — make some /predict calls first.")
    df = pd.DataFrame(rows)[FEATURES]
    return df.tail(n_recent) if n_recent else df


def run_drift_report(n_recent: int | None = None) -> dict:
    reference = pd.read_csv(REFERENCE_CSV)[FEATURES]
    current = load_current(n_recent)

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))

    drift = report.as_dict()["metrics"][0]["result"]
    summary = {
        "report": report_path.name,
        "n_current": int(len(current)),
        "drifted_features": int(drift["number_of_drifted_columns"]),
        "total_features": int(drift["number_of_columns"]),
        "dataset_drift": bool(drift["dataset_drift"]),
    }
    return summary


def main() -> None:
    summary = run_drift_report()
    print(f"Report saved: monitoring/reports/{summary['report']}")
    print(
        f"Drift: {summary['drifted_features']}/{summary['total_features']} features drifted "
        f"| dataset_drift={summary['dataset_drift']} (n_current={summary['n_current']})"
    )


if __name__ == "__main__":
    main()
