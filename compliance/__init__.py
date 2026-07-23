"""EU AI Act compliance & model-risk documentation tooling for the
heart-disease-mlops project.

Modules:
  ai_act_reference        -- verbatim Annex III/IV + Article 9-15 citations (single source of truth)
  requirement_evidence_map-- Step 1: requirement -> real-artifact evidence map (self-verifying)
  risk_classifier         -- Step 2: Annex III / Art. 6 risk classification with cited reasoning
  schemas                 -- Step 3: Pydantic schemas per Annex IV section
  doc_generator           -- Step 3: populate schemas from real metadata -> LaTeX -> PDF
  audit_trail             -- Step 4: tamper-evident hash chain over the prediction log

This is documentation/audit tooling, not a legal conformity certification.
"""

__all__ = [
    "ai_act_reference",
    "requirement_evidence_map",
    "risk_classifier",
    "schemas",
    "doc_generator",
    "audit_trail",
]
