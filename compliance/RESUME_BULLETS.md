# Resume / interview material - EU AI Act compliance layer

All bullets are backed by code in `compliance/` and verified by 15 passing
tests. Numbers are real (read live from `models/training_metrics.json`).

## Resume bullets (pick 1-2)

**Concise (one line):**
- Built an EU AI Act (Regulation 2024/1689) compliance layer over a heart-disease
  MLOps system: auto-generated Annex IV technical documentation, Annex III risk
  classification with cited reasoning, and a SHA-256 hash-chained audit trail for
  Article 12 record-keeping (15/15 tests).

**Two-line / detailed:**
- Designed a model-risk documentation toolkit mapping an ML system's controls to
  EU AI Act Articles 9-15 - self-verifying requirement-to-evidence traceability,
  a Pydantic-schema-driven Annex IV technical-documentation PDF generated from
  live model metadata, and an Annex III risk classifier that outputs a cited,
  auditable decision trace instead of an opaque label.
- Implemented a tamper-evident prediction audit trail (per-record SHA-256 hash
  chain) satisfying Article 12 record-keeping, with tests proving mutation,
  deletion and reordering of historical records are all detected.

**For a regulated/Irish-EU employer (emphasis on governance):**
- Translated EU AI Act high-risk obligations (Annex III classification, Annex IV
  documentation, Art. 9 risk management, Art. 12 record-keeping, Art. 15
  accuracy/robustness, Art. 72 post-market monitoring) into working tooling over
  a real ML system - including honest gap-flagging of controls not yet met.

## STAR story (behavioural interview)

**Situation.** EU AI Act high-risk obligations are entering enforcement through
2026, but ML portfolios rarely show the documentation and traceability a
regulated deployment requires.

**Task.** Turn an existing heart-disease MLOps project into something that
demonstrates AI Act conformity *patterns* end to end, without overclaiming.

**Action.** Read the primary regulation (Annex III/IV, Arts 9-15, 72) and built
six components: a self-verifying requirement→evidence map, an intended-purpose
risk classifier with a cited decision trace, a Pydantic→LaTeX Annex IV document
generator fed by live metrics, a SHA-256 hash-chained audit trail, a live
compliance dashboard tab, and honest documentation. Every claim is pinned to a
real artifact; incomplete controls are recorded as gaps.

**Result.** 4 of 7 obligations fully evidenced and 3 partial (with documented
limitations), a technical-documentation PDF citing the real ROC-AUC 0.9273 and
Brier 0.098, an audit chain over 58 predictions with tamper-detection tests, and
15/15 passing tests.

## Likely interview questions - and defensible answers

**"Is this legally compliant?"**
No, and I'm careful about that claim. It's a demonstration of the documentation
and audit *tooling patterns* the Act requires - not a legal conformity
assessment, CE marking or declaration of conformity, and the model isn't a
medical device. Conflating the two would be the mistake; naming the boundary is
part of the deliverable.

**"Why is the model 'minimal-risk' if it's health-related?"**
Because AI Act risk is driven by *intended purpose*, not subject matter. As a
demonstrator not placed on the market, it isn't high-risk. But my classifier's
trace shows it flips to high-risk under Annex III(5)(a)/(c)/(d) or Article 6(1)
the moment it's used for benefit eligibility, insurance pricing, triage, or as a
medical device. I encoded that as a counterfactual so the reasoning is explicit.

**"What are the weakest parts?"**
Three documented gaps: monitoring detects input drift only (no concept-drift
detection, because I don't collect ground-truth outcomes); retraining is manual,
not drift-triggered; and human oversight is documentation-plus-explanations, not
a formal override workflow. I'd rather surface those than hide them - an auditor
trusts a self-critical document more.

**"How does the audit trail actually work?"**
Each prediction record stores a SHA-256 hash of the previous record's contents - 
the same chaining a blockchain uses for block headers. Recomputing the chain
detects any edit, deletion or reordering of past records, and I pinpoint the
sequence index where it breaks. There's a test that mutates a historical entry
and asserts the validator catches it.

**"Why LaTeX/Pydantic for the docs?"**
Pydantic gives the Annex IV structure typed validation - a missing required
field fails the build instead of shipping a document with holes. LaTeX gives a
clean, reproducible PDF, and the whole thing is populated from live metadata, so
the document can't drift from the model's real numbers.
