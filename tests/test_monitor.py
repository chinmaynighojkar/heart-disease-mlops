"""Monitoring tests: a deliberately drifted population must trigger drift."""

from src.monitor import run_simulated_drift


def test_simulated_drift_fires():
    summary = run_simulated_drift()
    assert summary["dataset_drift"] is True
    assert summary["drifted_features"] >= summary["total_features"] // 2
    assert summary["report"].endswith(".html")
