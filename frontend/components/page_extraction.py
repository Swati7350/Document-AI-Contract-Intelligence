import streamlit as st
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from processors.extractor import extract, DEFAULT_SYSTEM_PROMPT, DEFAULT_EXTRACTION_PROMPT


def render() -> None:
    st.markdown('<div class="sec-header"><div class="sec-title">🧩 Structured Extraction & Prompt Studio</div><div class="sec-sub">Extract contract fields as structured JSON. Edit prompts and re-run to tune results.</div></div>', unsafe_allow_html=True)

    ocr = st.session_state.get("ocr_result")
    text = ocr["full_text"] if ocr else _demo_text()

    # ── Prompt Studio ──────────────────────────────────────────────────────────
    with st.expander("🎛  Prompt Studio — View & Edit Prompts", expanded=False):
        st.markdown('<div style="font-weight:600;color:#0f172a;margin-bottom:6px;">System Prompt</div>', unsafe_allow_html=True)
        sys_p = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=90, label_visibility="collapsed", key="sys_prompt")

        st.markdown('<div style="font-weight:600;color:#0f172a;margin:12px 0 6px;">Extraction Prompt Template</div>', unsafe_allow_html=True)
        ext_p = st.text_area("Extraction Prompt", value=DEFAULT_EXTRACTION_PROMPT, height=200, label_visibility="collapsed", key="ext_prompt")

        c1, c2 = st.columns([1,3])
        with c1:
            run_custom = st.button("▶  Re-run with Custom Prompts", type="primary")

    if "sys_prompt" not in st.session_state:
        sys_p, ext_p, run_custom = DEFAULT_SYSTEM_PROMPT, DEFAULT_EXTRACTION_PROMPT, False

    # ── Run extraction ─────────────────────────────────────────────────────────
    if "extraction_result" not in st.session_state or run_custom:
        with st.spinner("Running LLM extraction…"):
            result = extract(text, sys_p, ext_p)
        st.session_state.extraction_result = result

    result = st.session_state.extraction_result
    fields = result.get("fields", {})

    # ── Meta bar ───────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">
  <span class="badge badge-green">✓ Extraction Complete</span>
  <span class="badge badge-blue">Engine: {result.get('engine','mock').upper()}</span>
  <span class="badge badge-blue">~{result.get('tokens_used','—')} tokens</span>
  <span class="badge badge-blue">{result.get('processing_time_ms','—')}ms</span>
</div>""", unsafe_allow_html=True)

    # ── Two columns: human readable + JSON ────────────────────────────────────
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div style="font-weight:700;color:#0f172a;margin-bottom:12px;">📋 Extracted Fields</div>', unsafe_allow_html=True)

        field_icons = {
            "parties":           ("👥", "Parties"),
            "contract_number":   ("🔢", "Contract Number"),
            "effective_date":    ("📅", "Effective Date"),
            "expiry_date":       ("📅", "Expiry Date"),
            "contract_value":    ("💰", "Contract Value"),
            "payment_terms":     ("💳", "Payment Terms"),
            "termination_clause":("⚠️", "Termination Clause"),
            "governing_law":     ("⚖️", "Governing Law"),
            "signatories":       ("✍️", "Signatories"),
        }

        for key, (icon, label) in field_icons.items():
            val = fields.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                if key == "parties":
                    val_str = ", ".join(f"{p.get('name','')} ({p.get('role','')})" for p in val) if isinstance(val[0], dict) else ", ".join(str(v) for v in val)
                elif key == "signatories":
                    val_str = " · ".join(f"{s.get('name','')} — {s.get('title','')} @ {s.get('company','')}" for s in val) if isinstance(val[0], dict) else ", ".join(str(v) for v in val)
                else:
                    val_str = ", ".join(str(v) for v in val)
            else:
                val_str = str(val)

            st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:10px;padding:11px 0;border-bottom:1px solid #f1f5f9;">
  <span style="font-size:1.1rem;flex-shrink:0;">{icon}</span>
  <div>
    <div style="font-size:.72rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;">{label}</div>
    <div style="font-size:.875rem;color:#0f172a;margin-top:2px;">{val_str}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div style="font-weight:700;color:#0f172a;margin-bottom:12px;">📦 Raw JSON Output</div>', unsafe_allow_html=True)
        display_fields = {k: v for k, v in fields.items()}
        json_str = json.dumps(display_fields, indent=2)
        st.code(json_str, language="json")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("💬  Go to Contract RAG →", type="primary"):
        st.session_state.page = "rag"; st.rerun()


def _demo_text() -> str:
    return """SERVICE AGREEMENT
This Service Agreement is entered into as of January 1, 2024, by and between
Acme Corporation (Client) and TechSolutions Ltd (Supplier). Contract Number: SVC-2024-0042.
Term: January 1, 2024 through December 31, 2025. Total value: USD 150,000.
Payment: Monthly USD 12,500 within 30 days of invoice.
Termination: 30 days written notice. Governing law: California, USA.
Signed: John Smith, CEO — Acme Corporation. Sarah Johnson, Director — TechSolutions Ltd."""
