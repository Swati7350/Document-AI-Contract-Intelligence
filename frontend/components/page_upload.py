"""
Upload page — store file, then show two action buttons.
No processing happens until the user clicks Start OCR or Start RAG.
"""
import io
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render() -> None:
    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">📤 Upload Document</div>'
        '<div class="sec-sub">Drop a PDF or image, then choose your workflow.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sample PDF download ───────────────────────────────────────────────────
    sample_path = Path(__file__).parent.parent.parent / "assets" / "sample_beverage_invoice.pdf"
    if sample_path.exists():
        col_dl, _ = st.columns([2, 3])
        with col_dl:
            st.download_button(
                label="⬇️  Download sample PDF to try",
                data=sample_path.read_bytes(),
                file_name="sample_beverage_invoice.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    uploaded = st.file_uploader(
        "file",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    # ── Nothing uploaded yet ──────────────────────────────────────────────────
    if not uploaded:
        # Clear any stale state from a previous run
        for k in ["uploaded_filename", "uploaded_bytes", "ocr_result",
                  "extraction_result", "doc_embedded", "chat_history"]:
            st.session_state.pop(k, None)

        st.markdown("""
<div class="upload-hint">
  <span class="icon">📂</span>
  Drag and drop a <strong>PDF</strong> or <strong>Image</strong> (PNG / JPG)<br>
  <span style="font-size:.8rem;color:#94a3b8;margin-top:6px;display:block;">
    Scanned contracts · invoices · multi-page documents
  </span>
</div>""", unsafe_allow_html=True)
        return

    # ── File received — store in session, do NOT process yet ─────────────────
    file_bytes = uploaded.read()
    st.session_state.uploaded_filename = uploaded.name
    st.session_state.uploaded_bytes    = file_bytes

    is_pdf = uploaded.name.lower().endswith(".pdf")
    size_kb = len(file_bytes) / 1024

    st.markdown(f"""
<div class="card" style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
  <span style="font-size:2rem">{'📄' if is_pdf else '🖼️'}</span>
  <div>
    <div style="font-weight:700;color:#0f172a;">{uploaded.name}</div>
    <div style="font-size:.78rem;color:#64748b;">
      {size_kb:.1f} KB &nbsp;·&nbsp; {'PDF document' if is_pdf else 'Image file'}
    </div>
  </div>
  <span class="badge badge-green" style="margin-left:auto;">✓ Ready</span>
</div>""", unsafe_allow_html=True)

    # Image preview
    if not is_pdf:
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        st.image(img, caption="Preview", use_column_width=True)
    else:
        st.info("PDF ready — preview not shown. Click a workflow button below to begin.")

    # ── Two large workflow buttons ─────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:.78rem;font-weight:600;color:#64748b;
            text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px;">
  Choose a workflow
</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
<div class="lp-wf-card" style="margin-bottom:10px;">
  <div class="lp-wf-icon">🔍</div>
  <div class="lp-wf-title">OCR &amp; Extract</div>
  <div class="lp-wf-desc">
    Extract text, tables, and structured JSON fields from the document.
  </div>
</div>""", unsafe_allow_html=True)
        if st.button("Start OCR →", type="primary", use_container_width=True, key="do_ocr"):
            _run_ocr()

    with col2:
        st.markdown("""
<div class="lp-wf-card" style="margin-bottom:10px;">
  <div class="lp-wf-icon">💬</div>
  <div class="lp-wf-title">Chat with Document</div>
  <div class="lp-wf-desc">
    Ask questions in plain English and get answers with source citations.
  </div>
</div>""", unsafe_allow_html=True)
        if st.button("Start RAG →", type="primary", use_container_width=True, key="do_rag"):
            _run_rag()


# ── Workflow runners ──────────────────────────────────────────────────────────

def _run_ocr() -> None:
    """OCR + extraction pipeline — runs on button click."""
    from processors import ocr as ocr_proc
    from processors import extractor

    fname = st.session_state.uploaded_filename
    fbytes = st.session_state.uploaded_bytes

    with st.spinner("Running OCR…"):
        ocr_result = ocr_proc.process_document(fbytes, fname)
    st.session_state.ocr_result = ocr_result

    with st.spinner("Extracting structured fields…"):
        ext_result = extractor.extract(ocr_result["full_text"])
    st.session_state.extraction_result = ext_result

    st.session_state.page = "ocr"
    st.rerun()


def _run_rag() -> None:
    """OCR → chunk → embed pipeline — silent background steps."""
    from processors import ocr as ocr_proc
    from processors import rag as rag_proc

    fname  = st.session_state.uploaded_filename
    fbytes = st.session_state.uploaded_bytes

    with st.spinner("Processing document for RAG…"):
        ocr_result = ocr_proc.process_document(fbytes, fname)
        rag_proc.embed_document(ocr_result["full_text"], fname)

    st.session_state.ocr_result   = ocr_result
    st.session_state.doc_embedded = True
    st.session_state.chat_history = []
    st.session_state.page         = "rag"
    st.rerun()
