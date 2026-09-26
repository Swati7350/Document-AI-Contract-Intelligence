"""
OCR Results page — summary cards + extracted fields + raw JSON.
Text Blocks, Layout Map, Bounding Boxes removed per UX spec.
"""
import json
import streamlit as st


def render() -> None:
    ocr    = st.session_state.get("ocr_result")
    ext    = st.session_state.get("extraction_result")

    if not ocr:
        st.warning("No results yet. Please upload a document and run OCR first.")
        if st.button("← Upload a document"):
            st.session_state.page = "upload"
            st.rerun()
        return

    fields = ext.get("fields", {}) if ext else {}

    # ── Page title ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">🔍 OCR Results</div>'
        '<div class="sec-sub">Structured extraction from your document.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Summary cards — Parties, Tables, Value, Dates ─────────────────────────
    parties = fields.get("parties", [])
    if isinstance(parties, list):
        parties_str = ", ".join(
            p.get("name", str(p)) if isinstance(p, dict) else str(p)
            for p in parties
        ) or "—"
    else:
        parties_str = str(parties) or "—"

    tables_count  = len(ocr.get("tables", []))
    contract_val  = fields.get("contract_value", "—") or "—"
    eff_date      = fields.get("effective_date", "—") or "—"
    exp_date      = fields.get("expiry_date",    "—") or "—"
    dates_str     = f"{eff_date} → {exp_date}"

    st.markdown(f"""
<div class="kpi-grid" style="grid-template-columns:repeat(4,1fr);">
  <div class="kpi">
    <div class="kpi-lbl">Parties</div>
    <div style="font-size:.9rem;font-weight:600;color:#0f172a;margin-top:4px;line-height:1.4;">{parties_str}</div>
  </div>
  <div class="kpi">
    <div class="kpi-val blue">{tables_count}</div>
    <div class="kpi-lbl">Tables Detected</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Contract Value</div>
    <div style="font-size:.95rem;font-weight:700;color:#16a34a;margin-top:4px;">{contract_val}</div>
  </div>
  <div class="kpi">
    <div class="kpi-lbl">Term</div>
    <div style="font-size:.8rem;font-weight:600;color:#0f172a;margin-top:4px;line-height:1.4;">{dates_str}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Extracted fields ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-weight:700;color:#0f172a;font-size:1rem;margin-bottom:12px;">'
        '📋 Extracted Fields</div>',
        unsafe_allow_html=True,
    )

    FIELD_META = [
        ("parties",            "👥", "Parties"),
        ("contract_number",    "🔢", "Contract Number"),
        ("effective_date",     "📅", "Effective Date"),
        ("expiry_date",        "📅", "Expiry Date"),
        ("contract_value",     "💰", "Contract Value"),
        ("payment_terms",      "💳", "Payment Terms"),
        ("termination_clause", "⚠️", "Termination Clause"),
        ("governing_law",      "⚖️", "Governing Law"),
        ("signatories",        "✍️", "Signatories"),
    ]

    for key, icon, label in FIELD_META:
        val = fields.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            if val and isinstance(val[0], dict):
                if key == "parties":
                    val_str = " · ".join(
                        f"{p.get('name','')} ({p.get('role','')})" for p in val
                    )
                elif key == "signatories":
                    val_str = " · ".join(
                        f"{s.get('name','')} — {s.get('title','')} @ {s.get('company','')}"
                        for s in val
                    )
                else:
                    val_str = ", ".join(str(v) for v in val)
            else:
                val_str = ", ".join(str(v) for v in val)
        else:
            val_str = str(val)

        st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:12px;
            padding:11px 0;border-bottom:1px solid #f1f5f9;">
  <span style="font-size:1.05rem;flex-shrink:0;margin-top:1px;">{icon}</span>
  <div>
    <div style="font-size:.68rem;font-weight:700;color:#94a3b8;
                text-transform:uppercase;letter-spacing:.05em;">{label}</div>
    <div style="font-size:.875rem;color:#0f172a;margin-top:3px;line-height:1.5;">
      {val_str}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Raw JSON (collapsible) ────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.expander("{ }  Raw JSON output", expanded=False):
        st.code(json.dumps(fields, indent=2), language="json")

    # ── Action ────────────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬  Switch to RAG Chat", type="secondary", use_container_width=True):
            # Mark as embedded — page_rag.py will build the index on first visit
            # using document_text / document_pages already in session state.
            # Sync those keys from ocr_result if they're missing.
            if not st.session_state.get("document_text"):
                st.session_state.document_text  = ocr.get("full_text", "")
                st.session_state.document_pages = ocr.get("pages", [])
                st.session_state.doc_name       = st.session_state.get(
                    "uploaded_filename", "doc"
                )
            st.session_state.doc_embedded  = True
            st.session_state.vector_store  = None   # force re-embed on RAG page
            st.session_state.chat_history  = []
            st.session_state._rag_is_demo  = False
            st.session_state.page          = "rag"
            st.rerun()
    with col2:
        if st.button("📤  Upload another document", type="secondary", use_container_width=True):
            for k in ["ocr_result", "extraction_result", "doc_embedded", "chat_history"]:
                st.session_state.pop(k, None)
            st.session_state.page = "upload"
            st.rerun()
