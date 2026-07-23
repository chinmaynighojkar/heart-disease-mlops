"""Tests for the Annex III risk classifier (Step 2).

Confirms the classifier produces defensible, explainable results with cited
reasoning -- not an opaque label -- and that the tier tracks the *intended use*.
"""

from compliance.risk_classifier import RiskTier, SystemProfile, classify


def test_default_demonstrator_is_not_high_risk_but_explains_why():
    r = classify()
    assert r.tier == RiskTier.MINIMAL
    # every result must carry a decision trace and cited reasoning
    assert len(r.steps) >= 3
    assert r.conclusion
    assert r.primary_citations
    # it must flag the intended-purpose sensitivity, not just say "minimal"
    assert any("intended-purpose" in a.lower() or "intended purpose" in a.lower()
               for a in r.ambiguities)


def test_insurance_pricing_deployment_is_high_risk():
    p = SystemProfile(used_for_insurance_pricing=True, deployed_in_eu=True)
    r = classify(p)
    assert r.tier == RiskTier.HIGH_RISK
    assert any("5(c)" in c for c in r.primary_citations)


def test_benefit_eligibility_deployment_is_high_risk():
    p = SystemProfile(used_by_public_authority_for_benefit_eligibility=True)
    r = classify(p)
    assert r.tier == RiskTier.HIGH_RISK
    assert any("5(a)" in c for c in r.primary_citations)


def test_medical_device_route_is_high_risk_via_article_6_1():
    p = SystemProfile(is_regulated_medical_device=True)
    r = classify(p)
    assert r.tier == RiskTier.HIGH_RISK
    assert any("6(1)" in c for c in r.primary_citations)


def test_prohibited_practice_short_circuits():
    p = SystemProfile(social_scoring=True)
    r = classify(p)
    assert r.tier == RiskTier.PROHIBITED
    assert r.primary_citations == ["Article 5"]


def test_every_step_has_a_citation():
    r = classify(SystemProfile(used_for_emergency_triage=True))
    for step in r.steps:
        assert step.citation, f"step {step.stage} has no citation"
