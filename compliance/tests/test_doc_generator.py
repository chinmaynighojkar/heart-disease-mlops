"""Tests for the Annex IV technical-documentation generator (Step 3b).

Covers the two areas the module had zero coverage on before this review:
tex_escape (the function defending the LaTeX render path) and end-to-end
LaTeX generation from real model metadata.
"""

from compliance import doc_generator as dg


def test_tex_escape_handles_all_special_characters():
    assert dg.tex_escape("a_b") == r"a\_b"
    assert dg.tex_escape("50% & $5 #1") == r"50\% \& \$5 \#1"
    assert dg.tex_escape("{x}") == r"\{x\}"
    assert dg.tex_escape("a~b^c") == r"a\textasciitilde{}b\textasciicircum{}c"


def test_tex_escape_backslash_is_not_reescaped():
    """Regression test: a naive sequential-replace implementation escapes
    the backslash first, then re-matches the braces it just inserted,
    producing a doubly-escaped, garbled result."""
    assert dg.tex_escape("a\\_b") == r"a\textbackslash{}\_b"


def test_tex_escape_is_single_pass_over_the_original_string():
    """No replacement output should ever be re-scanned for further escaping."""
    out = dg.tex_escape("\\{}")
    assert out == r"\textbackslash{}\{\}"


def test_build_documentation_and_render_latex_end_to_end():
    doc = dg.build_documentation()
    tex = dg.render_latex(doc)
    assert tex.startswith(r"\documentclass")
    assert r"\end{document}" in tex
    # Real, non-placeholder numbers must reach the rendered output.
    assert f"{doc.content_hash}" in tex
    assert doc.content_hash != ""


def test_content_hash_is_stable_across_builds_with_identical_metrics():
    """Two builds against the same metrics file must hash identically --
    otherwise the hash can't be used to detect real content changes,
    because it would also change on every rebuild from generated_utc alone."""
    doc1 = dg.build_documentation()
    doc2 = dg.build_documentation()
    assert doc1.generated_utc <= doc2.generated_utc  # built moments apart
    assert doc1.content_hash == doc2.content_hash


def test_residual_risks_come_from_the_live_evidence_map():
    """Sec5 residual_risks must reflect the requirement->evidence map's own
    PARTIAL/GAP notes, not a second hand-authored list that can drift from
    Step 1's honest gap-flagging."""
    from compliance.requirement_evidence_map import GAP, PARTIAL, build_evidence_map

    doc = dg.build_documentation()
    expected = [f"{e.article}: {e.note}" for e in build_evidence_map()
                if e.status in (PARTIAL, GAP) and e.note]
    assert doc.sec5.residual_risks == expected
    assert len(doc.sec5.residual_risks) > 0
