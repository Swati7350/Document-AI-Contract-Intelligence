import streamlit as st


def render() -> None:

      # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="lp-hero">
      <div class="lp-hero-badge">📄 &nbsp; AI-Powered Document Intelligence</div>

      <h1 class="lp-hero-title">
        Document AI &amp;<br>Contract Intelligence
      </h1>

      <p class="lp-hero-sub">
        Upload a PDF or scanned image and choose one workflow.
      </p>

      <div style="margin-top:10px;font-size:0.82rem;color:rgba(255,255,255,.82);
                  line-height:1.5;max-width:560px;margin-left:auto;margin-right:auto;">
        <strong>Portfolio Demo</strong> — A streamlined showcase of core OCR, RAG,
        and Document Intelligence capabilities. The production version includes
        additional enterprise features and integrations.
      </div>
    </div>
    """, unsafe_allow_html=True)
    # ── Two workflow cards ─────────────────────────────────────────────────────
    st.markdown('<div class="lp-cards-row">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
<div class="lp-wf-card">
  <div class="lp-wf-icon">🔍</div>
  <div class="lp-wf-title">OCR &amp; Structured Extraction</div>
  <div class="lp-wf-desc">
    Upload a scanned PDF or image. The system extracts text, detects tables,
    identifies stamps and signatures, then uses an LLM to pull out structured
    fields — parties, dates, values, clauses — as clean JSON.
  </div>
  <ul class="lp-wf-bullets">
    <ul class="lp-wf-bullets">
      <li>🧩 Structured JSON extraction</li>
      <li>📑 Contract &amp; invoice field extraction</li>
    </ul>
        
  </ul>
</div>
""", unsafe_allow_html=True)
        if st.button("Start OCR →", type="primary", use_container_width=True, key="btn_ocr"):
            st.session_state.page = "upload"
            st.session_state.workflow = "ocr"
            st.rerun()

    with col2:
        st.markdown("""
<div class="lp-wf-card">
  <div class="lp-wf-icon">💬</div>
  <div class="lp-wf-title">Chat with Documents (RAG)</div>
  <div class="lp-wf-desc">
    Upload a contract or any document, then ask questions in plain English.
    The system retrieves the most relevant chunks and generates a grounded
    answer with citations — no hallucinations.
  </div>
  <ul class="lp-wf-bullets">
    <li>📎 Chunk-level retrieval</li>
    <li>💬 Natural language Q&amp;A</li>
    <li>🔗 Source citations shown</li>
    <li>❓ Example questions included</li>
  </ul>
</div>
""", unsafe_allow_html=True)
        if st.button("Try RAG →", type="primary", use_container_width=True, key="btn_rag"):
            st.session_state.page = "upload"
            st.session_state.workflow = "rag"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Demo notice ───────────────────────────────────────────────────────────
    st.markdown("""
<div class="lp-notice">
  💡 <strong>Live Demo</strong> — Runs in mock mode with realistic contract data.
  Swap in real OCR / LLM engines via environment variables when ready.
</div>
""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="footer">
  <span class="footer-brand">📄 Document AI &amp; Contract Intelligence</span>
  <span class="footer-copy">Portfolio Demo · Swati Gupta · 2026</span>
</div>
""", unsafe_allow_html=True)
