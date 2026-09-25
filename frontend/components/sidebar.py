"""Sidebar — Home and Upload only. Clean, no internal pages exposed."""
import streamlit as st


def render(current_page: str) -> None:
    with st.sidebar:
        st.markdown("""
<style>
.sb-profile{text-align:center;padding:24px 16px 18px;border-bottom:1px solid #f1f5f9;}
.sb-avatar{width:68px;height:68px;border-radius:50%;object-fit:cover;display:block;margin:0 auto 10px;border:3px solid #e2e8f0;box-shadow:0 2px 8px rgba(15,23,42,.1);}
.sb-name{font-size:.92rem;font-weight:700;color:#0f172a;}
.sb-role{font-size:.72rem;color:#64748b;margin-top:2px;}
.sb-section{padding:14px 14px 0;}
.sb-sec-title{font-size:.65rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;}
.sb-social{display:flex;gap:7px;padding:14px 14px 18px;}
.sb-social-btn{flex:1;display:flex;align-items:center;justify-content:center;gap:5px;padding:7px 0;border-radius:8px;font-size:.73rem;font-weight:600;text-decoration:none!important;border:1px solid #e2e8f0;color:#475569!important;background:#f8fafc;transition:background .15s,color .15s;}
.sb-social-btn:hover{background:#0f172a;color:#fff!important;border-color:#0f172a;}
</style>
<div class="sb-profile">
  <img class="sb-avatar" src="https://avatars.githubusercontent.com/u/100705742?v=4" alt="Swati Gupta"/>
  <div class="sb-name">Swati Gupta</div>
  <div class="sb-role">AI · ML · Agentic AI Engineer</div>
</div>
""", unsafe_allow_html=True)

        # ── Nav — Home + Upload only ──────────────────────────────────────────
        st.markdown('<div class="sb-section"><div class="sb-sec-title">Navigation</div></div>', unsafe_allow_html=True)

        for icon, label, key in [("🏠", "Home", "landing"), ("📤", "Upload", "upload")]:
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        # ── Other projects ────────────────────────────────────────────────────
        st.markdown("""
<div class="sb-section" style="margin-top:10px;">
  <div class="sb-sec-title">Other Projects</div>
</div>
<div style="padding:0 14px;">
  <a href="https://lastpass-autofill-login-agent.onrender.com" target="_blank"
     style="display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:10px;border:1px solid #e2e8f0;font-size:.8rem;font-weight:500;color:#374151!important;text-decoration:none!important;margin-bottom:6px;background:#f8fafc;">
    🔐 LastPass Login Agent
    <span style="margin-left:auto;font-size:.62rem;font-weight:700;background:#dcfce7;color:#15803d;padding:2px 7px;border-radius:999px;">LIVE</span>
  </a>
  <a href="https://travel-agent-orchestration-and-whatsapp-integration.streamlit.app/" target="_blank"
     style="display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:10px;border:1px solid #e2e8f0;font-size:.8rem;font-weight:500;color:#374151!important;text-decoration:none!important;margin-bottom:6px;background:#f8fafc;">
    ✈️ Travel Agent
    <span style="margin-left:auto;font-size:.62rem;font-weight:700;background:#dcfce7;color:#15803d;padding:2px 7px;border-radius:999px;">LIVE</span>
  </a>
  <a href="https://ai-invoice-ocr.onrender.com/" target="_blank"
     style="display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:10px;border:1px solid #e2e8f0;font-size:.8rem;font-weight:500;color:#374151!important;text-decoration:none!important;margin-bottom:6px;background:#f8fafc;">
    🧾 Invoice OCR
    <span style="margin-left:auto;font-size:.62rem;font-weight:700;background:#dcfce7;color:#15803d;padding:2px 7px;border-radius:999px;">LIVE</span>
  </a>
</div>
<div class="sb-social">
  <a class="sb-social-btn" href="https://github.com/Swati7350" target="_blank">⌥ GitHub</a>
  <a class="sb-social-btn" href="https://linkedin.com/in/swati7350" target="_blank">💼 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
