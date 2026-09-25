"""
Contract RAG processor.
Mock mode: simple keyword search over text chunks.
Real mode: FAISS + OpenAI embeddings.
"""
from __future__ import annotations
import os, re, time, random
from typing import List

ENABLE_REAL_LLM = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"

# ── In-memory store (replaced by FAISS in real mode) ─────────────────────────
_CHUNKS: List[dict] = []
_DOC_TEXT: str = ""


def embed_document(text: str, doc_id: str) -> dict:
    """Chunk document and store for retrieval."""
    global _CHUNKS, _DOC_TEXT
    _DOC_TEXT = text
    chunks = _chunk_text(text)
    _CHUNKS = [{"id": i, "doc_id": doc_id, "text": c} for i, c in enumerate(chunks)]
    return {"chunks_created": len(_CHUNKS), "doc_id": doc_id}


def query(question: str) -> dict:
    """Retrieve relevant chunks and generate an answer."""
    if not _CHUNKS:
        return {"answer": "No document embedded yet. Please upload and process a document first.", "chunks": []}
    if ENABLE_REAL_LLM:
        return _real_query(question)
    return _mock_query(question)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks or [text]


# ── Mock ──────────────────────────────────────────────────────────────────────

_MOCK_ANSWERS = {
    "terminat": {
        "answer": "Either party may terminate this Agreement upon **30 days written notice**. The Client may terminate immediately for cause if the Supplier materially breaches the agreement.",
        "chunks": [4],
    },
    "payment":  {
        "answer": "The Client shall pay the Supplier a **monthly fee of USD 12,500**, payable within **30 days** of invoice receipt. The total contract value is USD 150,000 over 12 months.",
        "chunks": [1],
    },
    "supplier": {
        "answer": "The Supplier is **TechSolutions Ltd**, represented by **Sarah Johnson (Director)**.",
        "chunks": [0],
    },
    "expir":    {
        "answer": "The Agreement expires on **December 31, 2025** unless earlier terminated by either party per the termination clause.",
        "chunks": [2],
    },
    "confiden": {
        "answer": "Each party agrees to keep confidential all proprietary information disclosed by the other party during the term of the Agreement and for **3 years thereafter**.",
        "chunks": [3],
    },
    "govern":   {
        "answer": "This Agreement is governed by the laws of the **State of California, USA**.",
        "chunks": [3],
    },
}


def _mock_query(question: str) -> dict:
    time.sleep(0.6)
    q_lower = question.lower()
    matched = None
    for keyword, ans in _MOCK_ANSWERS.items():
        if keyword in q_lower:
            matched = ans
            break

    if not matched:
        matched = {
            "answer": "Based on the contract, the relevant clause states that both parties have agreed to the terms as outlined in this Service Agreement dated January 1, 2024.",
            "chunks": [0, 1],
        }

    # Pull actual chunk texts
    retrieved = []
    for cid in matched["chunks"]:
        if cid < len(_CHUNKS):
            retrieved.append(_CHUNKS[cid])
        elif _CHUNKS:
            retrieved.append(_CHUNKS[0])

    return {
        "answer": matched["answer"],
        "retrieved_chunks": retrieved,
        "model": "mock",
        "tokens_used": random.randint(180, 320),
    }


# ── Real (stub) ───────────────────────────────────────────────────────────────

def _real_query(question: str) -> dict:
    raise NotImplementedError("Real RAG not wired yet.")
