"""
Contract RAG chat page.
Uses the uploaded document as the retrieval source; never loads sample content.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from processors import rag as rag_proc

EXAMPLE_QUESTIONS = [
    "What is the late fee?",
    "What is the termination notice period?",
    "Who is the supplier?",
    "What are the payment terms?",
    "What is the contract number?",
]


def render() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    document_text = st.session_state.get("document_text", "")
    if document_text and not st.session_state.get("doc_embedded"):
        doc_name = st.session_state.get("doc_name") or st.session_state.get("uploaded_filename", "document")
        pages = st.session_state.get("document_pages") or [document_text]
        with st.spinner("Embedding uploaded document…"):
            rag_proc.embed_document(document_text, doc_name, pages=pages)
        st.session_state.doc_embedded = True
        st.session_state.chat_history = []

    if not document_text:
        st.warning("No document has been uploaded yet. Please upload a PDF and start the RAG flow.")
        if st.button("← Upload a document"):
            st.session_state.page = "upload"
            st.rerun()
        return

    fname = st.session_state.get("doc_name") or st.session_state.get("uploaded_filename", "Uploaded document")
    st.markdown(
        '<div class="sec-header">'
        '<div class="sec-title">💬 Chat with Document</div>'
        f'<div class="sec-sub">{fname} — ask anything about it.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

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

    for msg in st.session_state.chat_history:
        _render_message(msg)

    with st.form("rag_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "question",
                label_visibility="collapsed",
                placeholder="Ask a question about the uploaded document…",
            )
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        _ask(question.strip())

    if st.session_state.chat_history:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("🗑  Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


def _render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]
    chunks = msg.get("chunks", [])

    if role == "user":
        st.markdown(
            '<div class="chat-msg user">'
            '<div class="chat-avatar user-av">👤</div>'
            f'<div class="chat-bubble user-b">{content}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        chunk_html = ""
        for j, c in enumerate(chunks):
            snippet = c.get("text", "")[:280] + ("…" if len(c.get("text", "")) > 280 else "")
            page = c.get("page", 1)
            chunk_html += (
                '<div class="chunk-card">'
                f'<div class="chunk-label">📎 Source Chunk {j + 1} · Page {page}</div>'
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
        "role": "assistant",
        "content": resp["answer"],
        "chunks": resp.get("retrieved_chunks", []),
    })
    st.rerun()
