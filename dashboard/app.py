"""Streamlit monitoring dashboard: model health, predictions, and drift."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
LOG_PATH = ROOT / "monitoring" / "logs" / "predictions.jsonl"
REPORTS_DIR = ROOT / "monitoring" / "reports"
SHAP_IMG = ROOT / "dashboard" / "shap_summary.png"

st.set_page_config(page_title="Heart Disease Risk Monitoring", layout="wide")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((MODELS / "training_metrics.json").read_text())


@st.cache_resource
def load_model_bits():
    model = joblib.load(MODELS / "classifier.joblib")
    features = joblib.load(MODELS / "feature_names.joblib")
    return model, features


@st.cache_data
def shap_summary_image() -> str:
    """Generate (once) a SHAP mean-importance bar chart from the training data."""
    if SHAP_IMG.exists():
        return str(SHAP_IMG)
    model, features = load_model_bits()
    ref = pd.read_csv(ROOT / "data" / "processed" / "train_reference.csv")[features]
    explainer = shap.TreeExplainer(model)
    values = explainer(ref)
    if values.values.ndim == 3:  # (samples, features, classes) -> positive class
        values = values[:, :, -1]
    plt.figure()
    shap.summary_plot(values, ref, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_IMG, dpi=110, bbox_inches="tight")
    plt.close()
    return str(SHAP_IMG)


def load_predictions() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


st.title("❤️ Heart Disease Risk Model Monitoring")
tab1, tab2, tab3 = st.tabs(["Model Overview", "Recent Predictions", "Drift Monitoring"])

# --- Tab 1: Model Overview ---
with tab1:
    metrics = load_metrics()
    tm = metrics["test_metrics"]
    st.subheader(f"Model: {metrics['model_type']} (v{metrics['model_version']})")
    cols = st.columns(5)
    for col, (label, key) in zip(
        cols, [("Accuracy", "accuracy"), ("ROC-AUC", "roc_auc"),
               ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    ):
        col.metric(label, f"{tm[key]:.3f}")

    st.markdown("### Feature importance (mean |SHAP|)")
    st.image(shap_summary_image(), use_container_width=False, width=700)

# --- Tab 2: Recent Predictions ---
with tab2:
    preds = load_predictions()
    if preds.empty:
        st.info("No predictions logged yet. Make some /predict calls.")
    else:
        st.markdown(f"**{len(preds)} predictions logged.** Showing the last 20:")
        recent = preds[["timestamp", "risk_score", "prediction"]].tail(20).iloc[::-1]
        st.dataframe(recent, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Risk score distribution**")
            st.bar_chart(preds["risk_score"].value_counts(bins=10, sort=False))
        with c2:
            st.markdown("**Prediction class balance**")
            st.bar_chart(preds["prediction"].value_counts().rename({0: "No disease", 1: "Disease"}))

# --- Tab 3: Drift Monitoring ---
with tab3:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(REPORTS_DIR.glob("drift_report_*.html"), reverse=True)

    if st.button("Run new drift report"):
        from src.monitor import run_drift_report
        with st.spinner("Running Evidently drift analysis..."):
            try:
                summary = run_drift_report()
                st.success(
                    f"{summary['drifted_features']}/{summary['total_features']} features drifted "
                    f"| dataset_drift={summary['dataset_drift']} (n={summary['n_current']})"
                )
                reports = sorted(REPORTS_DIR.glob("drift_report_*.html"), reverse=True)
            except Exception as exc:  # noqa: BLE001, surface any pipeline error to the UI
                st.error(f"Could not run drift report: {exc}")

    if reports:
        st.markdown(f"**{len(reports)} report(s) available.** Most recent: `{reports[0].name}`")
        st.download_button(
            "Download most recent report (HTML)",
            data=reports[0].read_bytes(),
            file_name=reports[0].name,
            mime="text/html",
        )
        with st.expander("Available reports"):
            st.write([r.name for r in reports])
    else:
        st.info("No drift reports yet. Click the button above to generate one.")
