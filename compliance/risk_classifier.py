"""Step 2 -- EU AI Act risk classification for the heart-disease risk model.

Walks the AI Act classification logic in the order the Regulation itself applies
it, and returns a structured result with *cited reasoning at every branch* --
never an opaque yes/no. The point is that a human can read the trace and defend
the conclusion.

Decision order implemented (mirrors Arts 5, 6 and Annex III):
  1. Article 5   -- is it a prohibited practice?            -> PROHIBITED
  2. Article 6(1)-- product safety component under Annex I
                    harmonisation law (e.g. medical devices)? -> HIGH_RISK
  3. Article 6(2)+ Annex III -- does the intended use fall in
                    a listed high-risk area?                  -> HIGH_RISK (candidate)
  4. Article 6(3)-- derogation: no significant risk / purely
                    preparatory, and no profiling?            -> may drop to LIMITED
  5. Otherwise                                                -> MINIMAL / LIMITED

Because the model here is a *health risk score*, the interesting analysis is at
steps 2-4, and the module makes the Annex III(5) reasoning and the Article 6(1)
medical-device question explicit, including the ambiguity.

Run:  python -m compliance.risk_classifier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from compliance.ai_act_reference import (
    ANNEX_III,
    ARTICLE_6_1,
    ARTICLE_6_3,
    REGULATION,
)


class RiskTier(str, Enum):
    PROHIBITED = "Prohibited (Article 5)"
    HIGH_RISK = "High-risk (Chapter III)"
    LIMITED = "Limited-risk / transparency obligations (Article 50)"
    MINIMAL = "Minimal-risk (no specific obligations)"
    UNDETERMINED = "Undetermined -- requires human/legal judgement"


@dataclass
class SystemProfile:
    """Facts about the AI system that drive classification. Defaults describe
    THIS project: an educational heart-disease risk-scoring demonstrator."""
    name: str = "Heart Disease Risk Model"
    intended_purpose: str = (
        "Estimate an individual's probability of heart disease from 11 clinical "
        "features, as an educational MLOps demonstrator."
    )
    # Does it emulate a use listed in Annex III? For a clinical risk score the
    # nearest listed uses are in Area 5 (essential services incl. healthcare).
    performs_health_risk_scoring: bool = True
    used_by_public_authority_for_benefit_eligibility: bool = False  # Annex III 5(a)
    used_for_insurance_pricing: bool = False                        # Annex III 5(c)
    used_for_emergency_triage: bool = False                         # Annex III 5(d)
    # Article 6(1): is it a medical device (or safety component of one) under
    # Regulation (EU) 2017/745 requiring third-party conformity assessment?
    is_regulated_medical_device: bool = False
    # Article 6(3): does it perform profiling of natural persons? If yes, the
    # Art. 6(3) derogation can never apply.
    performs_profiling: bool = False
    # Is it actually placed on the market / put into service in the EU, or a
    # portfolio artifact? Drives whether obligations bite in practice.
    deployed_in_eu: bool = False
    # Article 5 prohibited-practice flags (all false for this system).
    uses_subliminal_or_manipulative_techniques: bool = False
    exploits_vulnerabilities: bool = False
    social_scoring: bool = False


@dataclass
class ClassificationStep:
    stage: str
    citation: str
    question: str
    answer: str
    outcome: str


@dataclass
class ClassificationResult:
    system: str
    tier: RiskTier
    steps: list[ClassificationStep] = field(default_factory=list)
    primary_citations: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    conclusion: str = ""

    def add(self, **kw) -> None:
        self.steps.append(ClassificationStep(**kw))


def classify(profile: SystemProfile | None = None) -> ClassificationResult:
    p = profile or SystemProfile()
    r = ClassificationResult(system=p.name, tier=RiskTier.UNDETERMINED)

    # --- Step 1: Article 5 prohibited practices ---------------------------
    prohibited = (
        p.uses_subliminal_or_manipulative_techniques
        or p.exploits_vulnerabilities
        or p.social_scoring
    )
    r.add(
        stage="1. Prohibited practice",
        citation="Article 5",
        question="Does the system use manipulative/subliminal techniques, exploit "
                 "vulnerabilities, or perform social scoring?",
        answer="No" if not prohibited else "Yes",
        outcome="Not prohibited; continue." if not prohibited
                else "PROHIBITED under Article 5.",
    )
    if prohibited:
        r.tier = RiskTier.PROHIBITED
        r.primary_citations = ["Article 5"]
        r.conclusion = "Classified as a prohibited practice under Article 5."
        return r

    # --- Step 2: Article 6(1) product-safety / medical device -------------
    r.add(
        stage="2. Product safety component (Annex I harmonisation law)",
        citation=ARTICLE_6_1["reference"],
        question="Is the system a medical device / safety component under Regulation "
                 "(EU) 2017/745 requiring third-party conformity assessment?",
        answer="No (educational demonstrator; not CE-marked, not a certified device)"
               if not p.is_regulated_medical_device else "Yes",
        outcome="Article 6(1) not triggered; continue to Annex III."
                if not p.is_regulated_medical_device
                else "HIGH-RISK under Article 6(1).",
    )
    if p.is_regulated_medical_device:
        r.tier = RiskTier.HIGH_RISK
        r.primary_citations = [ARTICLE_6_1["reference"], "Regulation (EU) 2017/745"]
        r.conclusion = (
            "High-risk under Article 6(1): a clinical risk-scoring tool deployed as a "
            "medical device would fall under the MDR and require conformity assessment."
        )
        r.ambiguities.append(
            "Whether a heart-disease risk score is a 'medical device' depends on its "
            "claimed intended purpose under MDR Art. 2 / Rule 11 -- a diagnostic or "
            "decision-support claim would likely pull it in. This demonstrator makes no "
            "such claim, so 6(1) is treated as not triggered."
        )

    # --- Step 3: Article 6(2) + Annex III listed use ----------------------
    area5 = ANNEX_III["areas"]["5"]
    listed_hits: list[str] = []
    if p.used_by_public_authority_for_benefit_eligibility:
        listed_hits.append("5(a)")
    if p.used_for_insurance_pricing:
        listed_hits.append("5(c)")
    if p.used_for_emergency_triage:
        listed_hits.append("5(d)")

    # The *nearest* Annex III hooks for a clinical risk score, even when not
    # currently wired to any of those deployments, are 5(a)/5(c)/5(d).
    nearest = "5(a) (healthcare-benefit eligibility), 5(c) (health-insurance pricing), " \
              "5(d) (emergency healthcare triage)"
    r.add(
        stage="3. Annex III listed high-risk use (Article 6(2))",
        citation="Article 6(2), Annex III area 5",
        question="Is the system's *intended purpose* one of the listed uses in "
                 "Annex III area 5 (essential services, incl. healthcare)?",
        answer=(f"Yes -- matches {', '.join(listed_hits)}" if listed_hits
                else f"Not as currently scoped. Nearest listed uses: {nearest}."),
        outcome=("Candidate HIGH-RISK under Annex III area 5." if listed_hits
                 else "As a pure demonstrator it is not wired to a listed deployment, "
                      "but the risk-scoring *capability* sits directly adjacent to area 5."),
    )

    if listed_hits:
        # --- Step 4: Article 6(3) derogation ------------------------------
        derogation_available = (not p.performs_profiling)
        r.add(
            stage="4. Article 6(3) derogation",
            citation=ARTICLE_6_3["reference"],
            question="Could the derogation apply (no significant risk / preparatory "
                     "task) -- and critically, does it avoid profiling?",
            answer=("Derogation unavailable: the system performs profiling of natural persons."
                    if p.performs_profiling else
                    "Profiling not performed, but a health risk score materially influences "
                    "decisions, so it is unlikely to be 'no significant risk'."),
            outcome=("Remains HIGH-RISK." if p.performs_profiling else
                     "Derogation unlikely to rescue it; treat as HIGH-RISK."),
        )
        r.tier = RiskTier.HIGH_RISK
        r.primary_citations = [
            "Article 6(2)",
            f"Annex III(5): {', '.join(listed_hits)}",
        ]
        r.conclusion = (
            "High-risk under Article 6(2) and Annex III area 5 "
            f"({', '.join(listed_hits)}). A health risk score materially influences "
            "decisions about a person's access to care or pricing, so the Article 6(3) "
            "derogation is unlikely to apply."
        )
    elif not p.is_regulated_medical_device:
        # Not a medical device, not wired to a listed deployment -> demonstrator.
        r.tier = RiskTier.MINIMAL
        r.primary_citations = ["Article 6(2)", "Annex III area 5 (adjacent, not triggered)"]
        r.conclusion = (
            "As built -- an educational demonstrator not placed on the EU market and not "
            "connected to any Annex III deployment -- the system is not high-risk today. "
            "HOWEVER, its intended-purpose class (clinical risk scoring) is exactly what "
            "Annex III area 5 and Article 6(1) capture: the moment it were used for "
            "benefit eligibility, insurance pricing, triage, or as a medical device, it "
            "would become high-risk and the full Chapter III obligations would attach."
        )

    # --- Ambiguities the module deliberately surfaces ---------------------
    r.ambiguities.append(
        "Classification is intended-purpose-driven (Art. 6): the SAME model is minimal-risk "
        "as a demonstrator and high-risk once deployed for healthcare eligibility, insurance "
        "pricing or triage. The tier is a property of the use, not the code."
    )
    r.ambiguities.append(
        "Annex III(5)(a) is worded for use 'by public authorities'; private-sector clinical "
        "risk scoring may instead engage Art. 6(1) (MDR) rather than Annex III -- the two "
        "routes to high-risk should both be checked."
    )
    if not r.conclusion:
        r.conclusion = "Undetermined on the given facts; escalate for human/legal judgement."
        r.tier = RiskTier.UNDETERMINED
    return r


def format_report(result: ClassificationResult) -> str:
    lines = [
        f"EU AI Act Risk Classification -- {result.system}",
        f"Basis: {REGULATION['id']} ({REGULATION['eli']})",
        "=" * 72,
        f"RESULT: {result.tier.value}",
        "",
        "Decision trace:",
    ]
    for s in result.steps:
        lines += [
            f"  [{s.stage}]  ({s.citation})",
            f"     Q: {s.question}",
            f"     A: {s.answer}",
            f"     -> {s.outcome}",
            "",
        ]
    lines.append("Primary citations: " + "; ".join(result.primary_citations))
    lines.append("")
    lines.append("Conclusion:")
    lines.append("  " + result.conclusion)
    lines.append("")
    lines.append("Ambiguities deliberately surfaced (not hidden):")
    for a in result.ambiguities:
        lines.append(f"  - {a}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Default profile = this project's demonstrator.
    print(format_report(classify()))
    print("\n\n" + "#" * 72)
    print("# Counterfactual: same model deployed for health-insurance pricing")
    print("#" * 72)
    deployed = SystemProfile(
        name="Heart Disease Risk Model (deployed for insurance pricing)",
        used_for_insurance_pricing=True,
        deployed_in_eu=True,
    )
    print(format_report(classify(deployed)))
