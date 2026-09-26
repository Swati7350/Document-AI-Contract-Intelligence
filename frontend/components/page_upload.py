"""
Upload page.

On file upload:
  1. Save bytes in st.session_state.uploaded_file
  2. Extract OCR text immediately
  3. Store: document_text, document_pages, doc_name
  4. Reset vector store: vector_store=None, doc_embedded=False

No sample documents. No fallback content. The uploaded file is the
only source of truth for both OCR and RAG workflows.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def render() -> None:
    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">📤 Upload Document</div>'
        '<div class="sec-sub">Drop a PDF or image, then choose your workflow.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sample PDF download (demo asset only — not loaded into RAG) ───────────
    sample_path = ROOT / "assets" / "sample_beverage_invoice.pdf"
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

    # ── Nothing uploaded ──────────────────────────────────────────────────────
    if not uploaded:
        _clear_document_state()
        st.markdown("""
<div class="upload-hint">
  <span class="icon">📂</span>
  Drag and drop a <strong>PDF</strong> or <strong>Image</strong> (PNG / JPG)<br>
  <span style="font-size:.8rem;color:#94a3b8;margin-top:6px;display:block;">
    Contracts · invoices · scanned documents
  </span>
</div>""", unsafe_allow_html=True)
        return

    # ── File received ─────────────────────────────────────────────────────────
    # Only re-process if a new file is uploaded (different name or size)
    prev_name = st.session_state.get("doc_name", "")
    prev_size = st.session_state.get("_doc_size", 0)
    file_bytes = uploaded.read()
    file_size  = len(file_bytes)

    if uploaded.name != prev_name or file_size != prev_size:
        # New document — extract text and reset vector store
        _ingest_document(file_bytes, uploaded.name, file_size)

    # ── Display file info ─────────────────────────────────────────────────────
    is_pdf  = uploaded.name.lower().endswith(".pdf")
    size_kb = file_size / 1024
    has_text = bool(st.session_state.get("document_text", "").strip())

    st.markdown(f"""
<div class="card" style="display:flex;align-items:center;gap:14px;margin-bottom:16px;">
  <span style="font-size:2rem">{'📄' if is_pdf else '🖼️'}</span>
  <div>
    <div style="font-weight:700;color:#0f172a;">{uploaded.name}</div>
    <div style="font-size:.78rem;color:#64748b;">
      {size_kb:.1f} KB &nbsp;·&nbsp;
      {'PDF' if is_pdf else 'Image'} &nbsp;·&nbsp;
      {len(st.session_state.get('document_pages', []))} page(s)
    </div>
  </div>
  <span class="badge {'badge-green' if has_text else 'badge-orange'}" style="margin-left:auto;">
    {'✓ Text extracted' if has_text else '⚠ No text found'}
  </span>
</div>""", unsafe_allow_html=True)

    if not has_text:
        st.warning(
            "No text could be extracted from this file. "
            "Scanned images require a real OCR engine. "
            "Try uploading a digitally-created PDF."
        )
        return

    # Image preview
    if not is_pdf:
        img = __import__("PIL.Image", fromlist=["Image"]).Image.open(
            io.BytesIO(file_bytes)
        )
        st.image(img, caption="Preview", use_column_width=True)
    else:
        with st.expander("👁  Preview extracted text", expanded=False):
            st.text(st.session_state["document_text"][:1500] + "…"
                    if len(st.session_state["document_text"]) > 1500
                    else st.session_state["document_text"])

    # ── Workflow buttons ──────────────────────────────────────────────────────
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
  <div class="lp-wf-desc">Extract structured fields from the document.</div>
</div>""", unsafe_allow_html=True)
        if st.button("Start OCR →", type="primary",
                     use_container_width=True, key="do_ocr"):
            _run_ocr(file_bytes, uploaded.name)

    with col2:
        st.markdown("""
<div class="lp-wf-card" style="margin-bottom:10px;">
  <div class="lp-wf-icon">💬</div>
  <div class="lp-wf-title">Chat with Document</div>
  <div class="lp-wf-desc">Ask questions and get grounded answers with citations.</div>
</div>""", unsafe_allow_html=True)
        if st.button("Start RAG →", type="primary",
                     use_container_width=True, key="do_rag"):
            _run_rag()


# ══════════════════════════════════════════════════════════════════
#  DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════════

def _ingest_document(file_bytes: bytes, filename: str, file_size: int) -> None:
    """
    Extract text from the uploaded file and store in session state.
    Resets the vector store so any previous embedding is gone.
    """
    from processors.ocr import extract_text, extract_pages

    with st.spinner("Extracting text from document…"):
        full_text  = extract_text(file_bytes, filename)
        page_strs  = extract_pages(file_bytes) if filename.lower().endswith(".pdf") else []

    # Build structured page list
    if page_strs:
        document_pages = [
            {"page": i + 1, "text": t.strip()}
            for i, t in enumerate(page_strs)
            if t.strip()
        ]
    elif full_text:
        document_pages = [{"page": 1, "text": full_text}]
    else:
        document_pages = []

    # Persist in session state
    st.session_state.uploaded_file    = file_bytes
    st.session_state.uploaded_bytes   = file_bytes          # legacy compat
    st.session_state.uploaded_filename = filename           # legacy compat
    st.session_state.doc_name         = filename
    st.session_state.document_text    = full_text
    st.session_state.document_pages   = document_pages
    st.session_state._doc_size        = file_size

    # Reset vector store — new document invalidates any previous embedding
    st.session_state.vector_store     = None
    st.session_state.doc_embedded     = False
    st.session_state.chat_history     = []
    st.session_state._rag_is_demo     = False


def _clear_document_state() -> None:
    """Clear all document-related session state when no file is uploaded."""
    for k in [
        "uploaded_file", "uploaded_bytes", "uploaded_filename",
        "doc_name", "document_text", "document_pages", "_doc_size",
        "ocr_result", "extraction_result",
        "vector_store", "doc_embedded", "chat_history", "_rag_is_demo",
    ]:
        st.session_state.pop(k, None)


# ══════════════════════════════════════════════════════════════════
#  WORKFLOW RUNNERS
# ══════════════════════════════════════════════════════════════════

def _run_ocr(file_bytes: bytes, filename: str) -> None:
    """Run OCR + structured extraction and navigate to OCR results page."""
    from processors import ocr as ocr_proc
    from processors import extractor

    with st.spinner("Running OCR…"):
        ocr_result = ocr_proc.process_document(file_bytes, filename)
    st.session_state.ocr_result = ocr_result

    # Sync document_text / document_pages in case they weren't set
    if not st.session_state.get("document_text"):
        st.session_state.document_text  = ocr_result["full_text"]
        st.session_state.document_pages = ocr_result.get("pages", [])

    with st.spinner("Extracting structured fields…"):
        ext_result = extractor.extract(ocr_result["full_text"])
    st.session_state.extraction_result = ext_result

    st.session_state.page = "ocr"
    st.rerun()


def _run_rag() -> None:
    """
    Embed document_text into the vector store and navigate to RAG chat.
    Uses the text already extracted by _ingest_document — no second OCR pass.
    """
    from processors import rag as rag_proc

    doc_text  = st.session_state.get("document_text", "")
    doc_name  = st.session_state.get("doc_name", "uploaded_doc")
    doc_pages = st.session_state.get("document_pages", [])

    if not doc_text.strip():
        st.error("No text available to embed. Please upload a readable PDF.")
        return

    page_texts = [p["text"] for p in doc_pages] if doc_pages else [doc_text]

    with st.spinner("Building vector index…"):
        result = rag_proc.embed_document(
            text=doc_text,
            doc_id=doc_name,
            pages=page_texts,
        )
    st.session_state.vector_store = result
    st.session_state.doc_embedded = True
    st.session_state.chat_history = []
    st.session_state._rag_is_demo = False
    st.session_state.page         = "rag"
    st.rerun()
