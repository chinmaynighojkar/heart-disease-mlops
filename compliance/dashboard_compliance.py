"""Step 5 -- Compliance tab for the Streamlit dashboard.

`render_compliance_tab()` is imported by dashboard/app.py. Every indicator on
this tab is backed by a LIVE check performed when the tab renders -- nothing is
a hard-coded green light:

  * Risk classification      -> compliance.risk_classifier.classify()
  * Evidence-artifact status -> verify_evidence_map() walks the filesystem
  * Audit-trail integrity    -> verify_chain() recomputes every SHA-256 hash
  * Technical documentation  -> checks the PDF exists and shows its content hash

If a cited file is deleted or a log entry is tampered with, the corresponding
indicator turns red on the next render.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from compliance.audit_trail import CHAIN_LOG, build_chain, verify_chain
from compliance.requirement_evidence_map import (
    GAP,
    MET,
    NA,
    PARTIAL,
    build_evidence_map,
    verify_evidence_map,
)
from compliance.risk_classifier import classify

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PDF_PATH = OUTPUT_DIR / "technical_documentation.pdf"

_STATUS_ICON = {MET: "🟢", PARTIAL: "🟡", GAP: "🔴", NA: "⚪"}


def render_compliance_tab() -> None:
    st.subheader("EU AI Act conformity (Regulation (EU) 2024/1689)")
    st.caption(
        "Documentation & audit tooling mapped to the AI Act - a portfolio "
        "demonstration of conformity patterns, **not** a legal certification."
    )

    # ---- Live checks -----------------------------------------------------
    result = classify()
    ev_ok, ev_problems = verify_evidence_map()
    rows = build_evidence_map()
    chain_res = verify_chain() if CHAIN_LOG.exists() else None

    counts = {MET: 0, PARTIAL: 0, GAP: 0, NA: 0}
    for e in rows:
        counts[e.status] = counts.get(e.status, 0) + 1

    # ---- Top-line indicators (each tied to a real check) -----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk classification", result.tier.value.split("(")[0].strip())
    c2.metric("Evidence artifacts", "All present" if ev_ok else f"{len(ev_problems)} missing",
              delta=None if ev_ok else "check failed",
              delta_color="normal" if ev_ok else "inverse")
    if chain_res is not None:
        c3.metric("Audit trail", "Intact" if chain_res.valid else f"BROKEN @ seq {chain_res.broken_at}",
                  delta=f"{chain_res.n_entries} entries",
                  delta_color="normal" if chain_res.valid else "inverse")
    else:
        c3.metric("Audit trail", "Not built", delta="run build", delta_color="off")
    c4.metric("Technical doc", "Generated" if PDF_PATH.exists() else "Not generated")

    st.markdown(
        f"**Conformity coverage:** {_STATUS_ICON[MET]} {counts[MET]} met · "
        f"{_STATUS_ICON[PARTIAL]} {counts[PARTIAL]} partial · "
        f"{_STATUS_ICON[GAP]} {counts.get(GAP,0)} gap · "
        f"{_STATUS_ICON[NA]} {counts.get(NA,0)} n/a"
    )

    if not ev_ok:
        st.error("Evidence verification FAILED:\n" + "\n".join(ev_problems))

    st.divider()

    # ---- Risk classification detail --------------------------------------
    with st.expander("① Risk classification - decision trace", expanded=True):
        st.write(f"**Result:** {result.tier.value}")
        st.write(f"**Primary citations:** {'; '.join(result.primary_citations)}")
        st.info(result.conclusion)
        for s in result.steps:
            st.markdown(f"- **{s.stage}** ({s.citation}) - {s.answer} → _{s.outcome}_")
        st.markdown("**Ambiguities surfaced (not hidden):**")
        for a in result.ambiguities:
            st.markdown(f"- {a}")

    # ---- Requirement -> evidence map -------------------------------------
    with st.expander("② Requirement → evidence map (Articles 9-15, 72)", expanded=True):
        table = []
        for e in rows:
            table.append({
                "": _STATUS_ICON[e.status],
                "Article": e.article,
                "Requirement": e.requirement,
                "Evidence": ", ".join(e.artifacts),
                "Metric": e.metric,
                "Status": e.status,
            })
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("Rows marked partial/gap are honest limitations, deliberately not overclaimed.")

    # ---- Audit trail integrity -------------------------------------------
    with st.expander("③ Audit-trail integrity (Article 12 record-keeping)", expanded=True):
        col_a, col_b = st.columns([1, 2])
        if col_a.button("Rebuild + verify chain"):
            n = build_chain()
            chain_res = verify_chain()
            st.success(f"Rebuilt {n} entries.")
        if chain_res is not None:
            if chain_res.valid:
                st.success(
                    f"✅ Chain intact - {chain_res.n_entries} prediction records verified. "
                    f"Head hash `{chain_res.head_hash[:24]}…`"
                )
                st.caption(
                    "Each record stores a SHA-256 hash of the previous one. Editing, deleting, "
                    "or reordering any past prediction breaks the chain and is flagged here."
                )
            else:
                st.error(f"❌ Chain BROKEN at seq {chain_res.broken_at}: {chain_res.reason}")
        else:
            st.info("Chain not built yet - click 'Rebuild + verify chain'.")

    # ---- Technical documentation -----------------------------------------
    with st.expander("④ Annex IV technical documentation", expanded=True):
        if PDF_PATH.exists():
            st.success("Technical documentation PDF generated from live model metadata.")
            st.download_button(
                "Download technical_documentation.pdf",
                data=PDF_PATH.read_bytes(),
                file_name="technical_documentation.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Not generated yet. Run `python -m compliance.doc_generator`.")
        json_path = OUTPUT_DIR / "technical_documentation.json"
        if json_path.exists():
            import json as _json
            data = _json.loads(json_path.read_text())
            st.caption(f"Doc version `{data.get('doc_version')}` · "
                       f"content hash `{data.get('content_hash','')[:24]}…` · "
                       f"generated {data.get('generated_utc')}")
