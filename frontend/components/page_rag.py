"""
RAG chat page.
Answers come only from the document embedded via embed_document().
No sample documents. No fallbacks. No hardcoded content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from processors import rag as rag_proc

EXAMPLE_QUESTIONS = [
    "What is the total amount?",
    "What are the payment terms?",
    "Who is the supplier?",
    "What is the invoice number?",
    "What is the late fee?",
]


def render() -> None:

    # ── Guard: must have an embedded document ─────────────────────────────────
    if not st.session_state.get("doc_embedded"):
        st.markdown(
            '<div class="sec-header">'
            '<div class="sec-title">💬 Chat with Document</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.warning(
            "No document embedded yet. "
            "Upload a document and click **Start RAG** first."
        )
        if st.button("📤  Go to Upload", type="primary"):
            st.session_state.page = "upload"
            st.rerun()
        return

    # ── Embed on first visit (if navigated here via Switch to RAG) ────────────
    # doc_embedded may be True (set by page_ocr.py Switch button) but the
    # ChromaDB collection might not be populated yet.
    if st.session_state.get("doc_embedded") and \
       not st.session_state.get("vector_store"):
        doc_text  = st.session_state.get("document_text", "")
        doc_name  = st.session_state.get("doc_name", "doc")
        doc_pages = st.session_state.get("document_pages", [])
        if doc_text.strip():
            page_texts = [p["text"] for p in doc_pages] if doc_pages else [doc_text]
            with st.spinner("Building vector index…"):
                result = rag_proc.embed_document(
                    text=doc_text,
                    doc_id=doc_name,
                    pages=page_texts,
                )
            st.session_state.vector_store = result
            st.session_state.chat_history = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Header ────────────────────────────────────────────────────────────────
    doc_name = st.session_state.get("doc_name",
               st.session_state.get("uploaded_filename", "Uploaded document"))
    chunks_n = (st.session_state.get("vector_store") or {}).get("chunks_created", "?")

    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">💬 Chat with Document</div>'
        f'<div class="sec-sub">📄 {doc_name} &nbsp;·&nbsp; {chunks_n} chunks indexed</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Example question chips ────────────────────────────────────────────────
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

    st.markdown(
        "<hr style='border:none;border-top:1px solid #e2e8f0;margin:16px 0;'>",
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        _render_message(msg)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("rag_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "q", label_visibility="collapsed",
                placeholder="Ask a question about the document…",
            )
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        _ask(question.strip())

    # ── Controls ──────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑  Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


# ── Render helpers ────────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
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
        return

    # Build chunk cards without nested f-strings
    chunk_html = ""
    for j, c in enumerate(chunks):
        page_label = f" · p.{c['page']}" if c.get("page") else ""
        snippet = c["text"][:300] + ("…" if len(c["text"]) > 300 else "")
        chunk_html += (
            '<div class="chunk-card">'
            f'<div class="chunk-label">📎 Source Chunk {j+1}{page_label}</div>'
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
    with st.spinner("Searching document…"):
        resp = rag_proc.query(question)
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": resp["answer"],
        "chunks":  resp.get("retrieved_chunks", []),
    })
    st.rerun()
