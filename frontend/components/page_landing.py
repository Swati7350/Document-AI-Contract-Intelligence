import streamlit as st


def render() -> None:
    st.markdown("""
<div class="hero">
  <div class="hero-badge">📄 &nbsp; AI-Powered Document Intelligence</div>
  <h1 class="hero-title">Document AI &amp;<br>Contract Intelligence</h1>
  <p class="hero-sub">
    Upload any contract, invoice, or scanned document — get OCR, structured extraction,
    RAG-powered Q&amp;A, and model evaluation in seconds.
  </p>
  <div class="hero-chips">
    <span class="hero-chip">🔍 OCR &amp; Layout</span>
    <span class="hero-chip">🧩 LLM Extraction</span>
    <span class="hero-chip">💬 Contract RAG</span>
    <span class="hero-chip">📊 Evaluation</span>
    <span class="hero-chip">🎛 Prompt Studio</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-banner">
  💡 <div><strong>Live Demo —</strong>
  This platform runs fully in mock mode with realistic demo data.
  Upload any PDF or image to explore all features.
  Swap in real OCR/LLM engines via environment variables.</div>
</div>
""", unsafe_allow_html=True)

    features = [
        ("🔍", "OCR & Layout Parsing",    "Extract text blocks, detect tables, stamps and seals. Multi-column reconstruction with bounding-box visualisation."),
        ("🧩", "Structured Extraction",   "LLM-powered JSON extraction of contract parties, dates, values, signatories, and clauses."),
        ("💬", "Contract RAG",            "Chat over uploaded contracts. Ask about termination clauses, payment schedules, or any clause — with source chunks shown."),
        ("🎛", "Prompt Studio",           "View and edit system + extraction prompts. Re-run extraction with custom templates to tune accuracy."),
        ("📊", "Evaluation Dashboard",    "Verified test dataset with precision, recall, F1 and field-level accuracy. Ground-truth vs. predicted comparison."),
        ("⚙️", "Pluggable Architecture",  "OCR, LLM, and vector DB layers are fully swappable. Replace mock processors without touching the frontend."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
<div class="feat-card">
  <span class="feat-icon">{icon}</span>
  <div class="feat-title">{title}</div>
  <div class="feat-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("📤  Upload a Document", type="primary", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

    st.markdown("""
<div class="footer">
  <span class="footer-brand">📄 Document AI & Contract Intelligence</span>
  <span class="footer-copy">Portfolio Demo · Swati Gupta · 2026</span>
</div>
""", unsafe_allow_html=True)
