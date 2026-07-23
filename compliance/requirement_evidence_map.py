"""Step 1 -- Requirement -> evidence mapping.

Maps each operative requirement of the EU AI Act (Chapter III, Section 2:
Articles 9-15, plus Art. 72 post-market monitoring) to a CONCRETE artifact that
already exists in this repository, with the exact metric or file that evidences
it. Where a control genuinely does not exist, it is recorded as a GAP rather
than fabricated -- an honest map is the whole point.

`verify_evidence_map()` walks every cited artifact path and confirms it exists
on disk, so the table cannot silently rot or reference invented files.

Run:  python -m compliance.requirement_evidence_map        # prints + verifies
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from compliance.ai_act_reference import ARTICLES, REGULATION

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "models" / "training_metrics.json"

# Status vocabulary -------------------------------------------------------
MET = "MET"          # a real control fully evidences the requirement
PARTIAL = "PARTIAL"  # a real control exists but only partially covers it
GAP = "GAP"          # no control exists; recorded honestly, not invented
NA = "N/A"           # requirement is out of scope for a demonstrator/non-provider


@dataclass
class Evidence:
    article: str            # e.g. "Article 15"
    requirement: str        # short paraphrase of the obligation
    control: str            # what we actually built
    artifacts: list[str]    # repo-relative paths that back the claim
    metric: str             # the concrete number / fact, or ""
    status: str             # MET | PARTIAL | GAP | N/A
    note: str = ""          # honest caveat / gap explanation


def _metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def build_evidence_map() -> list[Evidence]:
    """Construct the map, pulling live numbers from training_metrics.json."""
    m = _metrics()
    tm = m["test_metrics"]
    cal = m["calibration"]
    sub = m["subgroup_metrics_by_sex"]

    roc = f"{tm['roc_auc']:.4f}"
    acc = f"{tm['accuracy']:.4f}"
    brier_c = f"{cal['brier_calibrated']:.4f}"
    brier_u = f"{cal['brier_uncalibrated']:.4f}"
    f_auc = f"{sub['female']['roc_auc']:.4f}"
    male_auc = f"{sub['male']['roc_auc']:.4f}"

    return [
        Evidence(
            article="Article 9",
            requirement="Continuous, documented risk-management system across the lifecycle.",
            control="Decision log recording every design decision, the rejected alternatives, "
                    "and how to defend each; documented data-leakage risk and its mitigation.",
            artifacts=["docs/decision_log.md", "src/preprocessing.py"],
            metric="cholesterol==0 (18.7% of rows) treated as missing and imputed to median 238.0 "
                   "to remove a spurious 'missing-field' signal (88% vs 55% disease rate).",
            status=PARTIAL,
            note="Risk identification and mitigation are documented and reproducible, but retraining "
                 "is manual (src/train.py); there is no automated, drift-triggered risk-response loop.",
        ),
        Evidence(
            article="Article 10",
            requirement="Data governance: representative data, bias examination, handling of gaps.",
            control="Stratified train/test split; documented data-quality handling; subgroup "
                    "performance measured by sex to surface disparate accuracy.",
            artifacts=["models/training_metrics.json", "src/preprocessing.py", "docs/model_card.md"],
            metric=f"734 train / 184 test (stratified, seed 42); subgroup ROC-AUC female {f_auc} "
                   f"vs male {male_auc}; median imputation for cholesterol & resting_bp_s.",
            status=MET,
            note="Bias is measured on the sex attribute only; the public dataset carries no other "
                 "protected attributes, which is itself a documented representativeness limitation.",
        ),
        Evidence(
            article="Article 11",
            requirement="Technical documentation per Annex IV, drawn up and kept up to date.",
            control="Annex IV technical-documentation generator populating a Pydantic schema from "
                    "live model metadata and rendering a versioned PDF.",
            artifacts=["compliance/doc_generator.py", "compliance/schemas.py",
                       "models/training_metrics.json"],
            metric="9 Annex IV sections generated from real metadata; model version "
                   f"{m['model_version']}.",
            status=MET,
        ),
        Evidence(
            article="Article 12",
            requirement="Automatic recording of events (logs) ensuring traceability of functioning.",
            control="Append-only prediction log capturing timestamp, full feature vector, risk score "
                    "and decision for every /predict call; extended with a tamper-evident hash chain.",
            artifacts=["monitoring/logs/predictions.jsonl", "api/main.py",
                       "compliance/audit_trail.py"],
            metric=f"{_count_log_lines()} logged predictions; each chained entry stores a SHA-256 "
                   "hash of the previous entry so any retroactive edit is detectable.",
            status=MET,
        ),
        Evidence(
            article="Article 13",
            requirement="Transparency: interpretable output and instructions for use.",
            control="Per-prediction SHAP attributions served with each prediction; global SHAP "
                    "summary; model card documenting capabilities, limitations and intended use.",
            artifacts=["models/shap_model.joblib", "dashboard/shap_summary.png",
                       "docs/model_card.md", "api/main.py"],
            metric="Local SHAP values returned per prediction; global SHAP summary over 11 features.",
            status=MET,
        ),
        Evidence(
            article="Article 14",
            requirement="Human oversight: enable a person to interpret, override or not use the output.",
            control="Model card frames the system as decision-support only, out of scope for "
                    "autonomous clinical use; SHAP gives an overseer per-case reasoning to judge output.",
            artifacts=["docs/model_card.md", "dashboard/shap_summary.png"],
            metric="Documented intended-use boundary + per-case explanations to support a human reviewer.",
            status=PARTIAL,
            note="Oversight is supported by explanation and documentation, but there is no formal "
                 "override workflow or logged human-in-the-loop sign-off, since the system is a "
                 "portfolio demonstrator, not a deployed clinical tool.",
        ),
        Evidence(
            article="Article 15",
            requirement="Appropriate and consistent accuracy, robustness and cybersecurity.",
            control="Held-out evaluation with calibrated probabilities; calibration quality measured "
                    "by Brier score; Evidently data-drift monitoring for input robustness.",
            artifacts=["models/training_metrics.json", "docs/reliability.png",
                       "src/monitor.py", "monitoring/reports"],
            metric=f"Test ROC-AUC {roc}, accuracy {acc}; Brier {brier_c} calibrated vs {brier_u} "
                   "uncalibrated (Platt/sigmoid); Evidently PSI/drift reports generated.",
            status=PARTIAL,
            note="Monitoring detects INPUT (data) drift only. Performance/concept drift is not "
                 "detected because ground-truth outcomes for past predictions are not collected -- "
                 "a documented Article 15 / Article 72 limitation, not a solved control.",
        ),
        Evidence(
            article="Article 72",
            requirement="Documented post-market monitoring system and plan.",
            control="Scheduled/on-demand Evidently drift reporting over the live prediction log; "
                    "drift-simulation mode demonstrating the report firing on a shifted population.",
            artifacts=["src/monitor.py", "monitoring/reports"],
            metric="Data-drift reports generated on logged traffic; --simulate mode proves detection.",
            status=PARTIAL,
            note="Covers input-distribution monitoring; a full Art. 72 plan would add outcome "
                 "collection, performance tracking and a documented response procedure.",
        ),
    ]


def _count_log_lines() -> int:
    log = REPO_ROOT / "monitoring" / "logs" / "predictions.jsonl"
    if not log.exists():
        return 0
    return sum(1 for line in log.read_text(encoding="utf-8").splitlines() if line.strip())


def verify_evidence_map() -> tuple[bool, list[str]]:
    """Confirm every cited artifact actually exists. Returns (ok, problems)."""
    problems: list[str] = []
    for ev in build_evidence_map():
        if ev.article not in ARTICLES:
            problems.append(f"{ev.article}: not found in ai_act_reference.ARTICLES")
        for rel in ev.artifacts:
            if not (REPO_ROOT / rel).exists():
                problems.append(f"{ev.article}: cited artifact missing -> {rel}")
    return (len(problems) == 0, problems)


def to_markdown() -> str:
    rows = build_evidence_map()
    out = [
        f"# Requirement -> Evidence Map ({REGULATION['id']})",
        "",
        f"Source: {REGULATION['id']}, {REGULATION['oj_version']} ({REGULATION['eli']}).",
        "",
        "Every row cites a real artifact in this repository. Rows marked **GAP** or "
        "**PARTIAL** are honest limitations, deliberately not papered over.",
        "",
        "| Article | Requirement | Control (what we built) | Evidence | Status |",
        "|---|---|---|---|---|",
    ]
    for ev in rows:
        arts = "<br>".join(f"`{a}`" for a in ev.artifacts)
        evidence = f"{arts}<br>{ev.metric}" if ev.metric else arts
        req = f"**{ev.article}: {ARTICLES[ev.article]['title']}**<br>{ev.requirement}"
        out.append(f"| {ev.article} | {req} | {ev.control} | {evidence} | **{ev.status}** |")
    out.append("")
    out.append("## Notes and limitations")
    out.append("")
    for ev in rows:
        if ev.note:
            out.append(f"- **{ev.article} ({ev.status}).** {ev.note}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ok, problems = verify_evidence_map()
    print(to_markdown())
    print("\n" + "=" * 70)
    if ok:
        print("VERIFICATION PASSED: every cited artifact exists on disk.")
    else:
        print("VERIFICATION FAILED:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
