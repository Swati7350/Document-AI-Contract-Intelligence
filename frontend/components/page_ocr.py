"""
OCR Results page — full extracted text + raw JSON output.
No fixed schema cards (Parties, Contract Value, Term, etc.).
Table contents are shown when detected, with headers read from the document.
"""
import json
import streamlit as st


def render() -> None:
    ocr = st.session_state.get("ocr_result")

    if not ocr:
        st.warning("No results yet. Please upload a document and run OCR first.")
        if st.button("← Upload a document"):
            st.session_state.page = "upload"
            st.rerun()
        return

    full_text = ocr.get("full_text", "").strip()
    tables    = ocr.get("tables", [])

    # ── Page title ────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">🔍 OCR Results</div>'
        '<div class="sec-sub">Full extracted text from your document.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Full extracted text ───────────────────────────────────────────────────
    st.markdown(
        '<div style="font-weight:700;color:#0f172a;font-size:1rem;margin-bottom:8px;">'
        '📄 Full extracted text</div>',
        unsafe_allow_html=True,
    )

    if full_text:
        st.text_area(
            label="full_text",
            value=full_text,
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("No text could be extracted from this document.")

    # ── Detected tables ───────────────────────────────────────────────────────
    if tables:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-weight:700;color:#0f172a;font-size:1rem;margin-bottom:8px;">'
            f'📊 Tables detected ({len(tables)})</div>',
            unsafe_allow_html=True,
        )
        for tbl in tables:
            headers = tbl.get("headers", [])
            rows    = tbl.get("rows", [])
            if not headers and not rows:
                continue

            import pandas as pd

            # Pad / trim rows to match header length
            n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
            padded = []
            for row in rows:
                if len(row) < n_cols:
                    row = row + [""] * (n_cols - len(row))
                padded.append(row[:n_cols])

            df = pd.DataFrame(padded, columns=headers if headers else None)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Raw JSON output ───────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    raw_json = {"full_text": full_text}
    if tables:
        raw_json["tables"] = tables

    with st.expander("{ }  Raw JSON output", expanded=False):
        st.code(json.dumps(raw_json, indent=2), language="json")

    # ── Actions ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💬  Switch to RAG Chat", type="secondary", use_container_width=True):
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
