# EU AI Act Compliance & Model-Risk Documentation Layer

A documentation and audit-tooling layer built **on top of** the
`heart-disease-mlops` project. It maps the existing MLOps controls to the
requirements of the **EU AI Act (Regulation (EU) 2024/1689)** and generates the
kinds of artifacts a provider of a high-risk AI system has to produce: a
requirement-to-evidence map, a risk classification, Annex IV technical
documentation, and a tamper-evident audit trail.

> **Scope and honesty note.** This is a demonstration of *conformity patterns
> and documentation/audit tooling*, not a legal conformity assessment and not
> legal advice. The heart-disease model itself is an educational demonstrator,
> not a medical device. Where a control is incomplete, this layer records it as
> a **gap** rather than claiming coverage - that honesty is the point.

## Problem

The AI Act's obligations for high-risk AI systems (Chapter III, Section 2 - 
Articles 9-15, plus Annex III classification, Annex IV documentation, and
Article 72 post-market monitoring) are being phased into enforcement through
2026. Most ML portfolios show a model and a metric; very few show the
*evidence, documentation and traceability* a regulated deployment actually
requires. The gap between "I trained a classifier" and "I can hand an auditor
the file" is exactly what this layer fills.

## Solution

Six components, each with a verification gate so nothing is decorative:

| # | Component | File | What it produces |
|---|-----------|------|------------------|
| 1 | Requirement → evidence map | `requirement_evidence_map.py` | Every Art. 9-15/72 obligation mapped to a **real artifact** and metric; self-verifies that each cited file exists |
| 2 | Risk classification | `risk_classifier.py` | Walks Art. 5 → 6(1) → Annex III → 6(3) with a **cited decision trace** and surfaced ambiguities |
| 3 | Technical documentation | `schemas.py`, `doc_generator.py` | Pydantic schema per Annex IV section, populated from live metadata, rendered to a **LaTeX PDF** |
| 4 | Audit trail | `audit_trail.py` | **SHA-256 hash chain** over the prediction log; any retro edit breaks it |
| 5 | Compliance dashboard | `dashboard_compliance.py` | A 4th Streamlit tab where every indicator runs a **live check** |
| 6 | This write-up | `README.md`, `RESUME_BULLETS.md` | Interview-defensible framing |

Single source of truth for all citations: `ai_act_reference.py` (verbatim
Annex III & IV, Article 9-15 obligations).

## Outcome

Running against the real model artifacts:

- **7 obligations mapped**, each to a file that exists (verified programmatically):
  4 met, 3 partial - with the partials (input-drift-only monitoring, manual
  retraining, informal human oversight) documented as honest limitations.
- **Risk classification:** minimal-risk *as built* (portfolio demonstrator), but
  the trace shows it becomes **high-risk under Annex III(5) or Article 6(1)** the
  moment it is used for healthcare-benefit eligibility, insurance pricing,
  triage, or as a medical device. Classification is a property of the *use*.
- **Annex IV PDF** generated from live metadata - cites the real figures
  (ROC-AUC **0.9273**, accuracy **0.8913**, Brier **0.098** calibrated vs
  **0.1023** uncalibrated, subgroup metrics by sex). No placeholder text.
- **Audit trail** chains all **58** logged predictions; a test suite mutates,
  deletes and reorders past entries and confirms the validator catches each.
- **15/15 tests pass.**

## Usage

```bash
# from the repo root
pip install -r requirements.txt          # adds pydantic (already used by the API)

# Step 1 - evidence map (prints + verifies every cited artifact exists)
python -m compliance.requirement_evidence_map

# Step 2 - risk classification (demonstrator + a deployed counterfactual)
python -m compliance.risk_classifier

# Step 3 - generate the Annex IV technical documentation PDF
python -m compliance.doc_generator
#   -> compliance/output/technical_documentation.pdf (+ .json)

# Step 4 - build and verify the tamper-evident audit trail
python -m compliance.audit_trail build
python -m compliance.audit_trail verify

# Step 5 - the compliance tab appears in the existing dashboard
streamlit run dashboard/app.py           # 4th tab: "AI Act Compliance"

# Tests
python -m pytest compliance/tests/ -v
```

## How each control maps to the Act

- **Article 9 (risk management)** → `docs/decision_log.md` documents every design
  decision, rejected alternative, and the data-leakage risk + mitigation
  (median-imputing the `cholesterol==0` sentinel). *Partial:* retraining is
  manual, not drift-triggered.
- **Article 10 (data governance)** → stratified split, documented data-quality
  handling, and subgroup accuracy measured by sex.
- **Article 11 + Annex IV (technical documentation)** → the generator here.
- **Article 12 (record-keeping)** → append-only prediction log + hash chain.
- **Article 13 (transparency)** → per-prediction SHAP + model card.
- **Article 14 (human oversight)** → documented decision-support-only boundary +
  per-case explanations. *Partial:* no formal override/sign-off workflow.
- **Article 15 (accuracy/robustness)** → calibrated probabilities, Brier score,
  Evidently drift monitoring. *Partial:* input drift only; no concept-drift
  detection because ground-truth outcomes are not collected.
- **Article 72 (post-market monitoring)** → Evidently reporting + drift
  simulation. *Partial:* no outcome collection or response SLA.

## Design choices worth defending in an interview

- **Honesty over coverage.** Partial/gap statuses are first-class. An auditor
  trusts a document that names its own limitations far more than one that claims
  green across the board.
- **Intended-purpose-driven classification.** The classifier refuses to give a
  context-free label; it shows *why*, cites the paragraph, and flags that the
  same weights are minimal- or high-risk depending on deployment.
- **Everything traces to a real file.** The evidence map self-verifies file
  existence; the doc reads live metadata; the audit trail recomputes real
  hashes. There is no invented compliance prose.

## What this is **not**

It is not a legal conformity assessment, not a CE marking, not an EU declaration
of conformity, and the underlying model is not a medical device. It is tooling
that demonstrates the documentation and audit *patterns* the AI Act requires.
