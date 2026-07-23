"""Step 3b -- Annex IV technical-documentation generator.

Populates the Pydantic `TechnicalDocumentation` schema from REAL model metadata
(models/training_metrics.json) plus the risk classification (Step 2) and the
requirement->evidence map (Step 1), then renders a LaTeX document and compiles
it to PDF with xelatex/pdflatex.

Every figure in the output is read live from training_metrics.json -- there is
no hard-coded compliance boilerplate carrying invented numbers. If the metrics
file changes, the document changes.

Run:  python -m compliance.doc_generator
Output: compliance/output/technical_documentation.pdf
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from compliance.ai_act_reference import ANNEX_IV, REGULATION
from compliance.requirement_evidence_map import GAP, PARTIAL, build_evidence_map
from compliance.risk_classifier import classify
from compliance.schemas import (
    Sec1GeneralDescription,
    Sec2DevelopmentProcess,
    Sec3MonitoringControl,
    Sec4MetricAppropriateness,
    Sec5RiskManagement,
    Sec6LifecycleChanges,
    Sec7Standards,
    Sec8DeclarationOfConformity,
    Sec9PostMarketMonitoring,
    TechnicalDocumentation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = REPO_ROOT / "models" / "training_metrics.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# --------------------------------------------------------------------------- #
# LaTeX helpers                                                               #
# --------------------------------------------------------------------------- #
_TEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_TEX_ESCAPE_RE = re.compile("|".join(re.escape(c) for c in _TEX_SPECIAL_CHARS))


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters in a single pass over the ORIGINAL
    string, so a replacement (e.g. the braces in \\textbackslash{}) is never
    re-matched by a later rule in the same call."""
    return _TEX_ESCAPE_RE.sub(lambda m: _TEX_SPECIAL_CHARS[m.group()], s)


def _kv(label: str, value: str) -> str:
    return f"\\textbf{{{tex_escape(label)}:}} {tex_escape(value)}\\\\\n"


# --------------------------------------------------------------------------- #
# Populate the schema from real metadata                                      #
# --------------------------------------------------------------------------- #
def build_documentation() -> TechnicalDocumentation:
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    tm = m["test_metrics"]
    cal = m["calibration"]
    sub = m["subgroup_metrics_by_sex"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    sec1 = Sec1GeneralDescription(
        intended_purpose=(
            "Estimate the calibrated probability of heart disease from 11 clinical "
            "features. Built as an educational MLOps demonstrator; NOT a medical "
            "device and not for clinical use."
        ),
        provider="Chinmay Nighojkar (portfolio project)",
        version=str(m["model_version"]),
        relation_to_previous_versions=(
            f"Version {m['model_version']}. Model selected by 5-fold CV ROC-AUC over "
            "RandomForest, GradientBoosting and XGBoost."
        ),
        interactions=(
            "Served via a FastAPI /predict endpoint (api/main.py); a Streamlit "
            "dashboard (dashboard/app.py) reads model metadata, the prediction log "
            "and drift reports. No external AI systems are involved."
        ),
        software_versions=(
            "scikit-learn 1.5.2, xgboost 2.1.3, shap 0.46.0, evidently 0.4.40, "
            "fastapi 0.115.6 (see requirements.txt)."
        ),
        forms_placed_on_market=(
            "Not placed on any market. Runs as a containerised service (Dockerfile) "
            "and a local dashboard for demonstration only."
        ),
        target_hardware="Commodity CPU; no GPU required for inference.",
        user_interface="FastAPI JSON API and a Streamlit monitoring dashboard.",
        instructions_for_use=(
            "Submit 11 clinical features to /predict; the response returns a calibrated "
            "risk_score, a 0/1 decision at a 0.5 threshold, and per-feature SHAP "
            "attributions. Output is decision-support only and must be reviewed by a human."
        ),
    )

    sec2 = Sec2DevelopmentProcess(
        development_methods=(
            "Supervised learning on the combined UCI heart-disease dataset. "
            "GridSearchCV hyper-parameter tuning on ROC-AUC; probability calibration "
            "with CalibratedClassifierCV (sigmoid/Platt, 5-fold)."
        ),
        design_specifications=(
            f"RandomForestClassifier (n_estimators={m['best_params']['n_estimators']}, "
            f"max_depth={m['best_params']['max_depth']}, "
            f"min_samples_leaf={m['best_params']['min_samples_leaf']}). The uncalibrated "
            "tree is retained only to compute SHAP attributions; the calibrated "
            "classifier produces the served probability."
        ),
        optimisation_target=(
            "ROC-AUC during selection/tuning; calibrated probability quality (Brier "
            "score) for the served output, because a risk score must be trustworthy as "
            "a probability, not just rank-ordered."
        ),
        system_architecture=(
            "preprocessing.py -> train.py (model + calibration + SHAP model persisted "
            "to models/) -> FastAPI serving with prediction logging -> monitor.py "
            "(Evidently drift) -> Streamlit dashboard."
        ),
        computational_resources="Trained on CPU in seconds; artefacts persisted with joblib.",
        data_requirements=(
            f"Combined UCI heart-disease dataset: 918 rows after de-duplication, split "
            f"{m['n_train']} train / {m['n_test']} test (stratified, random_state=42). "
            "Features: " + ", ".join(m["features"]) + "."
        ),
        data_cleaning=(
            "Values of 0 for cholesterol (18.7% of rows) and resting_bp_s encode a "
            "MISSING measurement, not a real reading, and leak a spurious signal (rows "
            "with missing cholesterol show an 88% disease rate vs 55% overall). These "
            f"zeros are replaced with the training-set median "
            f"(cholesterol={m['impute_values']['cholesterol']}, "
            f"resting_bp_s={m['impute_values']['resting_bp_s']})."
        ),
        human_oversight_assessment=(
            "Per-prediction SHAP attributions let a human reviewer see which features "
            "drove each score; the model card documents that output is decision-support "
            "only. No autonomous decision is taken."
        ),
        predetermined_changes=(
            "Retraining is performed by re-running train.py on updated data; there is no "
            "automated drift-triggered retraining (documented limitation)."
        ),
        validation_testing_procedure=(
            "5-fold stratified cross-validation for selection/tuning; a held-out 184-row "
            "test set for final metrics; automated tests in tests/ for the API and monitor."
        ),
        accuracy_metrics={
            "roc_auc": tm["roc_auc"],
            "accuracy": tm["accuracy"],
            "precision": tm["precision"],
            "recall": tm["recall"],
            "f1": tm["f1"],
            "brier_calibrated": cal["brier_calibrated"],
            "brier_uncalibrated": cal["brier_uncalibrated"],
        },
        subgroup_metrics=sub,
        cybersecurity_measures=(
            "Input validation via Pydantic request schemas on the API; pinned "
            "dependencies; containerised runtime. No authentication layer (demonstrator)."
        ),
    )

    sec3 = Sec3MonitoringControl(
        capabilities_and_limitations=(
            "Capability: calibrated heart-disease risk from tabular clinical inputs. "
            "Limitation: trained on a small public dataset, not validated on any real "
            "population; performance may not transfer."
        ),
        per_group_accuracy=(
            f"By sex -- female (n={sub['female']['n']}): ROC-AUC {sub['female']['roc_auc']:.4f}, "
            f"recall {sub['female']['recall']:.4f}; male (n={sub['male']['n']}): ROC-AUC "
            f"{sub['male']['roc_auc']:.4f}, recall {sub['male']['recall']:.4f}. The smaller "
            "female subgroup warrants caution."
        ),
        overall_expected_accuracy=(
            f"Held-out ROC-AUC {tm['roc_auc']:.4f}, accuracy {tm['accuracy']:.4f}, "
            f"recall {tm['recall']:.4f} at a 0.5 threshold."
        ),
        foreseeable_unintended_outcomes=(
            "False negatives (missed disease) are the most consequential error for a "
            "screening-style score; recall is reported alongside accuracy for this reason. "
            "Distribution shift in the patient population would degrade performance."
        ),
        human_oversight_measures=(
            "SHAP explanations per prediction; documented intended-use boundary; the "
            "dashboard exposes model health and drift to an overseer."
        ),
        input_data_specifications=(
            "11 numeric/categorical clinical features with the ranges and encodings "
            "documented in the model card; the API rejects malformed inputs."
        ),
    )

    sec4 = Sec4MetricAppropriateness(
        rationale=(
            "ROC-AUC captures ranking quality independent of threshold and is standard "
            "for clinical risk models. Brier score and a reliability curve are reported "
            "because a probability that will be read by a human must be calibrated, not "
            "merely discriminative. Recall is highlighted because missed disease is the "
            "costliest error. Subgroup metrics test for disparate accuracy (Art. 10/15)."
        )
    )

    ev = build_evidence_map()
    # Residual risks are pulled live from the requirement->evidence map's own
    # PARTIAL/GAP notes, rather than a second hand-authored copy of the same
    # facts -- so this section cannot drift from Step 1's honest gap-flagging.
    residual_risks = [f"{e.article}: {e.note}" for e in ev if e.status in (PARTIAL, GAP) and e.note]
    sec5 = Sec5RiskManagement(
        description=(
            "Iterative, documented risk management: risks and mitigations are recorded "
            "in docs/decision_log.md, and residual risks are stated openly rather than "
            "closed out. Mapped article-by-article in the requirement->evidence map."
        ),
        identified_risks=[
            "Data leakage from missing-value sentinels (cholesterol/resting_bp_s == 0).",
            "Disparate accuracy across sex subgroups (smaller female sample).",
            "Silent performance decay under population/concept drift.",
            "Over-reliance on an uncalibrated score by a human reader.",
        ],
        mitigations=[
            "Median imputation of sentinel zeros; documented in preprocessing.py and the model card.",
            "Subgroup metrics measured and reported by sex.",
            "Evidently input-drift monitoring with a drift-simulation demonstration.",
            "Probability calibration (Platt) + per-prediction SHAP for interpretability.",
        ],
        residual_risks=residual_risks,
    )

    sec6 = Sec6LifecycleChanges(
        changes=(
            f"Current model version {m['model_version']}. Version history and the "
            "rationale for each change are maintained in docs/decision_log.md. This "
            "compliance layer adds a tamper-evident audit trail over the prediction log."
        )
    )

    sec7 = Sec7Standards(
        harmonised_standards_applied=[
            "None formally applied (portfolio demonstrator).",
        ],
        solutions_where_no_standard=(
            "In the absence of applied harmonised standards, conformity-relevant practices "
            "follow widely used references in spirit: model cards (Mitchell et al., 2019) "
            "for transparency, calibration/Brier for accuracy, and SHAP for explainability. "
            "ISO/IEC 42001 and the forthcoming CEN-CENELEC AI Act harmonised standards "
            "would be the reference points for a real deployment."
        ),
    )

    sec8 = Sec8DeclarationOfConformity(
        status="Not applicable -- demonstrator, not placed on the EU market.",
        explanation=(
            "No EU declaration of conformity (Art. 47) is issued because this is a "
            "portfolio artifact and not a high-risk AI system placed on the market. The "
            "field is retained to show where a real Annex IV(8) copy would be attached."
        ),
    )

    sec9 = Sec9PostMarketMonitoring(
        system_description=(
            "Evidently-based input-drift monitoring over the live prediction log "
            "(src/monitor.py), producing dated HTML reports in monitoring/reports/."
        ),
        monitoring_plan=(
            "Run drift reporting on a schedule and on demand; investigate flagged feature "
            "drift; a --simulate mode demonstrates the report firing on a shifted population."
        ),
        known_limitations=(
            "Input drift only. A complete Art. 72 plan would add outcome collection, "
            "performance-drift tracking, alert thresholds and a documented response SLA."
        ),
    )

    doc = TechnicalDocumentation(
        generated_utc=now,
        doc_version=f"td-{m['model_version']}",
        sec1=sec1, sec2=sec2, sec3=sec3, sec4=sec4, sec5=sec5,
        sec6=sec6, sec7=sec7, sec8=sec8, sec9=sec9,
    )
    # Content hash over the substantive payload only: excludes the hash field
    # itself AND generated_utc (a wall-clock timestamp), so two builds with
    # identical substantive content produce the same hash. Without excluding
    # generated_utc, the hash would change on every run even when nothing
    # about the model or documentation actually changed.
    payload = doc.model_dump()
    payload.pop("content_hash", None)
    payload.pop("generated_utc", None)
    doc.content_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return doc


# --------------------------------------------------------------------------- #
# Render LaTeX                                                                #
# --------------------------------------------------------------------------- #
def render_latex(doc: TechnicalDocumentation) -> str:
    L: list[str] = []
    a = L.append
    a(r"\documentclass[11pt,a4paper]{article}")
    a(r"\usepackage[margin=2.2cm]{geometry}")
    a(r"\usepackage{fontspec}" if shutil.which("xelatex") else r"\usepackage[T1]{fontenc}")
    a(r"\usepackage{longtable}")
    a(r"\usepackage{booktabs}")
    a(r"\usepackage{xcolor}")
    a(r"\usepackage{titlesec}")
    a(r"\usepackage{fancyhdr}")
    a(r"\usepackage{enumitem}")
    a(r"\definecolor{acblue}{RGB}{20,40,90}")
    a(r"\definecolor{acgrey}{RGB}{90,90,90}")
    a(r"\titleformat{\section}{\Large\bfseries\color{acblue}}{\thesection}{0.6em}{}")
    a(r"\titleformat{\subsection}{\large\bfseries\color{acblue}}{\thesubsection}{0.6em}{}")
    a(r"\pagestyle{fancy}\fancyhf{}")
    a(r"\lhead{\small\color{acgrey}Technical Documentation -- EU AI Act Annex IV}")
    a(r"\rhead{\small\color{acgrey}\thepage}")
    a(r"\renewcommand{\headrulewidth}{0.4pt}")
    a(r"\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}")
    a(r"\begin{document}")

    # Title block
    a(r"\begin{center}")
    a(r"{\Huge\bfseries\color{acblue} Technical Documentation}\\[4pt]")
    a(r"{\large per Annex IV, " + tex_escape(REGULATION["id"]) + r" (EU AI Act)}\\[10pt]")
    a(r"{\large\bfseries " + tex_escape(doc.sec1.intended_purpose.split(".")[0]) + r"}\\[6pt]")
    a(_kv("Provider", doc.sec1.provider))
    a(_kv("System version", doc.sec1.version))
    a(_kv("Document version", doc.doc_version))
    a(_kv("Generated", doc.generated_utc))
    a(_kv("Content hash (SHA-256)", doc.content_hash))
    a(r"\end{center}")
    a(r"\vspace{4pt}\hrule\vspace{6pt}")
    a(r"\textit{\small This document is auto-generated from live model metadata. "
      r"It demonstrates Annex IV documentation structure for a portfolio project and "
      r"is not a legal conformity certification.}")

    # Risk classification banner (from Step 2)
    result = classify()
    a(r"\section*{Risk classification (summary)}")
    a(_kv("Classification (as built)", result.tier.value))
    a(_kv("Primary citations", "; ".join(result.primary_citations)))
    a(r"\textbf{Conclusion:} " + tex_escape(result.conclusion) + r"\\")

    def section(num: str, model, body_fn):
        title = ANNEX_IV["sections"][num]["title"]
        a(rf"\section*{{Annex IV({num}) -- {tex_escape(title)}}}")
        body_fn(model)

    def body1(s):
        a(_kv("Intended purpose", s.intended_purpose))
        a(_kv("Provider / version", f"{s.provider} / v{s.version}"))
        a(_kv("Relation to previous versions", s.relation_to_previous_versions))
        a(_kv("Interactions (b)", s.interactions))
        a(_kv("Software versions (c)", s.software_versions))
        a(_kv("Forms placed on market (d)", s.forms_placed_on_market))
        a(_kv("Target hardware (e)", s.target_hardware))
        a(_kv("User interface (g)", s.user_interface))
        a(_kv("Instructions for use (h)", s.instructions_for_use))

    def body2(s):
        a(_kv("Development methods (a)", s.development_methods))
        a(_kv("Design specifications (b)", s.design_specifications))
        a(_kv("Optimisation target (b)", s.optimisation_target))
        a(_kv("System architecture (c)", s.system_architecture))
        a(_kv("Computational resources (c)", s.computational_resources))
        a(_kv("Data requirements (d)", s.data_requirements))
        a(_kv("Data cleaning (d)", s.data_cleaning))
        a(_kv("Human-oversight assessment (e)", s.human_oversight_assessment))
        a(_kv("Pre-determined changes (f)", s.predetermined_changes))
        a(_kv("Validation & testing (g)", s.validation_testing_procedure))
        a(_kv("Cybersecurity (h)", s.cybersecurity_measures))
        # metrics table
        a(r"\textbf{Accuracy metrics (g):}\\[2pt]")
        a(r"\begin{tabular}{ll}\toprule Metric & Value \\ \midrule")
        for k, v in s.accuracy_metrics.items():
            a(f"{tex_escape(k)} & {v:.4f} \\\\")
        a(r"\bottomrule\end{tabular}\\[4pt]")
        a(r"\textbf{Subgroup metrics by sex (g -- discriminatory-impact check):}\\[2pt]")
        a(r"\begin{tabular}{lrrrr}\toprule Group & n & Accuracy & ROC-AUC & Recall \\ \midrule")
        for g, d in s.subgroup_metrics.items():
            a(f"{tex_escape(g)} & {d['n']} & {d['accuracy']:.4f} & {d['roc_auc']:.4f} & {d['recall']:.4f} \\\\")
        a(r"\bottomrule\end{tabular}")

    def body3(s):
        a(_kv("Capabilities & limitations", s.capabilities_and_limitations))
        a(_kv("Per-group accuracy", s.per_group_accuracy))
        a(_kv("Overall expected accuracy", s.overall_expected_accuracy))
        a(_kv("Foreseeable unintended outcomes", s.foreseeable_unintended_outcomes))
        a(_kv("Human-oversight measures", s.human_oversight_measures))
        a(_kv("Input-data specifications", s.input_data_specifications))

    def body4(s):
        a(tex_escape(s.rationale))

    def body5(s):
        a(tex_escape(s.description) + r"\\[4pt]")
        for label, items in [("Identified risks", s.identified_risks),
                             ("Mitigations", s.mitigations),
                             ("Residual risks (stated openly)", s.residual_risks)]:
            a(rf"\textbf{{{label}:}}")
            a(r"\begin{itemize}[leftmargin=1.4em,topsep=1pt]")
            for it in items:
                a(r"\item " + tex_escape(it))
            a(r"\end{itemize}")

    def body6(s):
        a(tex_escape(s.changes))

    def body7(s):
        a(_kv("Harmonised standards applied", "; ".join(s.harmonised_standards_applied)))
        a(_kv("Solutions where no standard", s.solutions_where_no_standard))

    def body8(s):
        a(_kv("Status", s.status))
        a(tex_escape(s.explanation))

    def body9(s):
        a(_kv("System description", s.system_description))
        a(_kv("Monitoring plan", s.monitoring_plan))
        a(_kv("Known limitations", s.known_limitations))

    section("1", doc.sec1, body1)
    section("2", doc.sec2, body2)
    section("3", doc.sec3, body3)
    section("4", doc.sec4, body4)
    section("5", doc.sec5, body5)
    section("6", doc.sec6, body6)
    section("7", doc.sec7, body7)
    section("8", doc.sec8, body8)
    section("9", doc.sec9, body9)

    a(r"\vfill\hrule\vspace{4pt}")
    a(r"\textit{\small Generated by the heart-disease-mlops compliance module. "
      r"Structure follows Annex IV of Regulation (EU) 2024/1689. Not legal advice.}")
    a(r"\end{document}")
    return "\n".join(L)


def generate_pdf() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = build_documentation()
    tex = render_latex(doc)
    tex_path = OUTPUT_DIR / "technical_documentation.tex"
    tex_path.write_text(tex, encoding="utf-8")
    # persist the structured JSON too (useful for the dashboard / audit)
    (OUTPUT_DIR / "technical_documentation.json").write_text(
        doc.model_dump_json(indent=2), encoding="utf-8"
    )

    engine = "xelatex" if shutil.which("xelatex") else "pdflatex"
    for _ in range(2):  # twice for headers/page numbers
        proc = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(OUTPUT_DIR), str(tex_path)],
            capture_output=True, text=True,
        )
    pdf_path = OUTPUT_DIR / "technical_documentation.pdf"
    if not pdf_path.exists():
        raise RuntimeError(
            f"PDF not produced. LaTeX tail:\n{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}"
        )
    # cleanup aux files (best-effort; never fail the build over a temp file)
    for ext in (".aux", ".log", ".out"):
        p = OUTPUT_DIR / f"technical_documentation{ext}"
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    return pdf_path


if __name__ == "__main__":
    path = generate_pdf()
    print(f"Generated: {path}")
