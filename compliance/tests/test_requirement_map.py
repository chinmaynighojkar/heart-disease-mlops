"""Tests for the requirement->evidence map (Step 1).

The load-bearing guarantee: every cited artifact must actually exist, and the
map must not silently drop the honest gaps.
"""

from compliance.requirement_evidence_map import (
    GAP,
    MET,
    PARTIAL,
    build_evidence_map,
    verify_evidence_map,
)


def test_every_cited_artifact_exists():
    ok, problems = verify_evidence_map()
    assert ok, "Cited artifacts missing:\n" + "\n".join(problems)


def test_core_articles_are_covered():
    covered = {e.article for e in build_evidence_map()}
    for art in ["Article 9", "Article 10", "Article 11", "Article 12",
                "Article 13", "Article 14", "Article 15"]:
        assert art in covered, f"{art} not mapped"


def test_honest_gaps_are_preserved():
    """The map must keep at least the documented PARTIAL limitations
    (input-drift-only, manual retraining) rather than overclaim full coverage."""
    statuses = {e.article: e.status for e in build_evidence_map()}
    assert statuses["Article 15"] == PARTIAL   # input drift only
    assert statuses["Article 9"] == PARTIAL    # manual retraining
    # and the notes must actually explain the limitation
    notes = {e.article: e.note for e in build_evidence_map()}
    assert "concept" in notes["Article 15"].lower() or "performance" in notes["Article 15"].lower()


def test_metrics_are_real_not_placeholder():
    """Spot-check that the map pulled the true ROC-AUC from metadata."""
    ev = {e.article: e for e in build_evidence_map()}
    assert "0.9273" in ev["Article 15"].metric   # real test ROC-AUC
