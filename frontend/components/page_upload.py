import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from processors import ocr as ocr_proc


def render() -> None:
    st.markdown('<div class="sec-header"><div class="sec-title">📤 Document Upload</div><div class="sec-sub">Upload a scanned PDF, image, or multi-page contract to begin processing.</div></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your file here",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown("""
<div class="upload-hint">
  <span class="icon">📂</span>
  Drag and drop a <strong>PDF</strong> or <strong>Image</strong> (PNG / JPG) here<br>
  <span style="font-size:.8rem;color:#94a3b8;margin-top:6px;display:block;">Supports scanned contracts, invoices, and multi-page documents</span>
</div>""", unsafe_allow_html=True)
        return

    # Store file in session
    st.session_state.uploaded_filename = uploaded.name
    st.session_state.uploaded_bytes    = uploaded.read()

    st.markdown(f"""
<div class="card" style="display:flex;align-items:center;gap:14px;">
  <span style="font-size:2rem">{'📄' if uploaded.name.endswith('.pdf') else '🖼️'}</span>
  <div>
    <div style="font-weight:700;color:#0f172a;">{uploaded.name}</div>
    <div style="font-size:.78rem;color:#64748b;">{len(st.session_state.uploaded_bytes)/1024:.1f} KB &nbsp;·&nbsp; {'PDF document' if uploaded.name.endswith('.pdf') else 'Image file'}</div>
  </div>
  <span class="badge badge-green" style="margin-left:auto;">✓ Ready</span>
</div>""", unsafe_allow_html=True)

    # Preview for images
    if not uploaded.name.lower().endswith(".pdf"):
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(st.session_state.uploaded_bytes))
        st.image(img, caption="Document preview", use_column_width=True)
    else:
        st.info("PDF uploaded — page preview available after OCR processing.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔍  Run OCR & Layout Analysis", type="primary", use_container_width=True):
            with st.spinner("Processing document…"):
                result = ocr_proc.process_document(
                    st.session_state.uploaded_bytes,
                    st.session_state.uploaded_filename,
                )
            st.session_state.ocr_result = result
            # Auto-embed for RAG
            from processors import rag as rag_proc
            rag_proc.embed_document(result["full_text"], uploaded.name)
            st.session_state.doc_embedded = True
            st.session_state.page = "ocr"
            st.rerun()
    with c2:
        if st.button("↺  Clear", type="secondary", use_container_width=True):
            for k in ["uploaded_filename","uploaded_bytes","ocr_result","extraction_result","doc_embedded"]:
                st.session_state.pop(k, None)
            st.rerun()
