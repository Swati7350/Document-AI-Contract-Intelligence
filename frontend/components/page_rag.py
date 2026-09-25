import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from processors import rag as rag_proc


EXAMPLE_QUESTIONS = [
    "What is the termination clause?",
    "What is the payment schedule?",
    "Who is the supplier?",
    "When does the contract expire?",
    "What is the governing law?",
]


def render() -> None:
    st.markdown('<div class="sec-header"><div class="sec-title">💬 Contract RAG</div><div class="sec-sub">Ask questions about the uploaded contract. Relevant chunks are retrieved and shown alongside the answer.</div></div>', unsafe_allow_html=True)

    # Embedding status
    if st.session_state.get("doc_embedded"):
        st.markdown('<div class="info-banner">✅ <div>Document is embedded and ready for Q&A. Ask anything about the contract below.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-banner">💡 <div>No document uploaded yet — using a built-in demo contract for Q&A.</div></div>', unsafe_allow_html=True)
        from processors.ocr import _mock_process
        demo = _mock_process("demo_contract.pdf")
        rag_proc.embed_document(demo["full_text"], "demo_contract")
        st.session_state.doc_embedded = True

    # Init chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example question chips
    st.markdown('<div style="font-size:.78rem;font-weight:600;color:#64748b;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em;">Try these questions</div>', unsafe_allow_html=True)
    q_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with q_cols[i]:
            if st.button(q, key=f"eq_{i}", use_container_width=True):
                _ask(q)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
<div class="chat-msg user">
  <div class="chat-avatar user-av">👤</div>
  <div class="chat-bubble user-b">{msg['content']}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="chat-msg">
  <div class="chat-avatar ai-av">🤖</div>
  <div style="max-width:78%;">
    <div class="chat-bubble ai-b">{msg['content']}</div>
    {''.join(f"""<div class="chunk-card"><div class="chunk-label">📎 Source Chunk {j+1}</div>{c['text'][:280]}…</div>""" for j,c in enumerate(msg.get('chunks',[]))) if msg.get('chunks') else ''}
  </div>
</div>""", unsafe_allow_html=True)

    # Input
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input("Ask a question…", label_visibility="collapsed", placeholder="e.g. What is the termination clause?")
        with c2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and question.strip():
        _ask(question.strip())

    if st.session_state.chat_history:
        if st.button("🗑  Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()


def _ask(question: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Retrieving and generating answer…"):
        resp = rag_proc.query(question)
    st.session_state.chat_history.append({
        "role":    "assistant",
        "content": resp["answer"],
        "chunks":  resp.get("retrieved_chunks", []),
    })
    st.rerun()
