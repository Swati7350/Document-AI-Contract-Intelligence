"""
Contract RAG chat page.
Clean ChatGPT-style interface — question input + AI answer + source chunks.
Fixed: f-string nesting that caused syntax errors in the previous version.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from processors import rag as rag_proc

EXAMPLE_QUESTIONS = [
    "What is the grand total?",
    "Who is the vendor?",
    "What items were purchased?",
    "What are the payment terms?",
    "What is the invoice number?",
]


def render() -> None:
    # ── Ensure something is embedded ─────────────────────────────────────────
    if not st.session_state.get("doc_embedded"):
        from processors.rag import embed_sample_doc
        with st.spinner("Loading sample invoice…"):
            embed_sample_doc()
        st.session_state.doc_embedded   = True
        st.session_state.chat_history   = []
        st.session_state._rag_is_demo   = True

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Header ────────────────────────────────────────────────────────────────
    fname = st.session_state.get("uploaded_filename", "Demo contract")
    is_demo = st.session_state.get("_rag_is_demo", False)

    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">💬 Chat with Document</div>'
        f'<div class="sec-sub">{"Demo contract loaded" if is_demo else fname} — ask anything about it.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if is_demo:
        st.info("📄 Sample document loaded: **ACME Industrial Supplies — Purchase Invoice (INV-2026-0147)**. Upload your own document via the sidebar to chat with it.")

    # ── Example questions ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:.75rem;font-weight:600;color:#94a3b8;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">'
        'Try these</div>',
        unsafe_allow_html=True,
    )
    eq_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with eq_cols[i]:
            if st.button(q, key=f"eq_{i}", use_container_width=True):
                _ask(q)

    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0;'>", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        _render_message(msg)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("rag_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "question",
                label_visibility="collapsed",
                placeholder="Ask a question about the document…",
            )
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        _ask(question.strip())

    # ── Clear ─────────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🗑  Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single chat message without f-string nesting."""
    role    = msg["role"]
    content = msg["content"]
    chunks  = msg.get("chunks", [])

    if role == "user":
        st.markdown(
            '<div class="chat-msg user">'
            '<div class="chat-avatar user-av">👤</div>'
            f'<div class="chat-bubble user-b">{content}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # Build chunk cards separately — no nested f-strings
        chunk_html = ""
        for j, c in enumerate(chunks):
            snippet = c["text"][:280] + ("…" if len(c["text"]) > 280 else "")
            chunk_html += (
                '<div class="chunk-card">'
                f'<div class="chunk-label">📎 Source Chunk {j + 1}</div>'
                f'{snippet}'
                '</div>'
            )

        st.markdown(
            '<div class="chat-msg">'
            '<div class="chat-avatar ai-av">🤖</div>'
            '<div style="max-width:78%;">'
            f'<div class="chat-bubble ai-b">{content}</div>'
            f'{chunk_html}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


def _ask(question: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Thinking…"):
        resp = rag_proc.query(question)
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": resp["answer"],
        "chunks":  resp.get("retrieved_chunks", []),
    })
    st.rerun()
