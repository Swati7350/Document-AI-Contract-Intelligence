"""
Document AI & Contract Intelligence
=====================================
Run:  streamlit run frontend/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Document AI & Contract Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "styles" / "main.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
for key, val in {"page": "landing", "chat_history": [], "doc_embedded": False}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
from frontend.components.sidebar import render as render_sidebar
render_sidebar(st.session_state.page)

# ── Router ────────────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "landing":
    from frontend.components.page_landing import render; render()
elif page == "upload":
    from frontend.components.page_upload import render; render()
elif page == "ocr":
    from frontend.components.page_ocr import render; render()
elif page == "extraction":
    from frontend.components.page_extraction import render; render()
elif page == "rag":
    from frontend.components.page_rag import render; render()
elif page == "evaluation":
    from frontend.components.page_evaluation import render; render()
else:
    st.session_state.page = "landing"
    st.rerun()
