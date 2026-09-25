"""
Contract RAG processor — ChromaDB vector store.
──────────────────────────────────────────────────────────────────
Vector store    : ChromaDB in-memory  (no server, no API key)
Embeddings      : ChromaDB default (ONNX all-MiniLM-L6-v2, bundled)
Chunking        : word-window 150 words / 30-word overlap
Retrieval       : Top-3 chunks by cosine distance
Generation      : sentence extraction from chunks (mock mode)
                  OpenAI GPT (set ENABLE_REAL_LLM=true + OPENAI_API_KEY)

Sample doc      : assets/sample_invoice.txt  (ACME Purchase Invoice)
                  Loaded only when no document has been uploaded.
──────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List

ENABLE_REAL_LLM = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"
SAMPLE_DOC_PATH = Path(__file__).parent.parent / "assets" / "sample_invoice.txt"

# ── In-memory ChromaDB client (lazy-init) ─────────────────────────────────────
_client     = None
_collection = None
_doc_id: str = ""
_chunks: List[dict] = []


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb
        _client = chromadb.Client()          # pure in-memory, no persistence
        _collection = _client.get_or_create_collection(
            name="doc_chunks",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def embed_document(text: str, doc_id: str) -> dict:
    """
    Chunk text, embed with ChromaDB's default model, store in-memory.
    Wipes any previously stored document first.
    """
    global _collection, _doc_id, _chunks

    # Reset collection so uploaded doc fully replaces the previous one
    if _client is not None and _collection is not None:
        try:
            _client.delete_collection("doc_chunks")
        except Exception:
            pass
        _collection = None

    col = _get_collection()
    _doc_id = doc_id

    raw = _chunk_text(text, size=150, overlap=30)
    _chunks = [{"id": i, "doc_id": doc_id, "text": c} for i, c in enumerate(raw)]

    col.add(
        documents=[c["text"] for c in _chunks],
        ids=[f"{doc_id}_{c['id']}" for c in _chunks],
        metadatas=[{"doc_id": doc_id, "chunk_id": c["id"]} for c in _chunks],
    )

    return {"chunks_created": len(_chunks), "doc_id": doc_id}


def embed_sample_doc() -> dict:
    """Load and embed the bundled ACME invoice sample document."""
    if SAMPLE_DOC_PATH.exists():
        text = SAMPLE_DOC_PATH.read_text(encoding="utf-8")
    else:
        # Minimal inline fallback
        text = (
            "PURCHASE INVOICE. ACME Industrial Supplies Pvt. Ltd. "
            "Invoice No: INV-2026-0147. Invoice Date: 24 Sep 2026. "
            "Bill To: Paras Engg Works, New Delhi. "
            "Items: Centrifugal Blower Impeller x2 ₹25,000; "
            "Industrial V-Belt B45 x5 ₹2,400; "
            "Motor Mounting Bracket x3 ₹3,450; "
            "Powder Coating Service x1 ₹2,800. "
            "Subtotal ₹33,650. GST 18% ₹6,057. Grand Total ₹39,707. "
            "Payment Terms: 30 Days. Vendor GSTIN: 07AABCU9603R1ZV. "
            "Stamp: PAID 25 Sep 2026."
        )
    return embed_document(text, "sample_invoice")


def query(question: str) -> dict:
    """Retrieve Top-3 relevant chunks and generate an answer."""
    col = _get_collection()
    if col.count() == 0:
        return {
            "answer": "No document has been processed yet. Please upload a document and click Start RAG.",
            "retrieved_chunks": [],
        }

    top_chunks = _retrieve(question, top_k=3)

    if ENABLE_REAL_LLM:
        return _llm_answer(question, top_chunks)
    return _synthesise_answer(question, top_chunks)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = 150, overlap: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return [text]
    result, i = [], 0
    while i < len(words):
        result.append(" ".join(words[i : i + size]))
        i += size - overlap
    return result


# ── ChromaDB retrieval ────────────────────────────────────────────────────────

def _retrieve(question: str, top_k: int = 3) -> List[dict]:
    col = _get_collection()
    k   = min(top_k, col.count())
    res = col.query(query_texts=[question], n_results=k)

    retrieved = []
    for i, doc_text in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        retrieved.append({
            "id":     meta.get("chunk_id", i),
            "doc_id": meta.get("doc_id", _doc_id),
            "text":   doc_text,
        })
    return retrieved if retrieved else ([_chunks[0]] if _chunks else [])


# ── Answer synthesis (no external calls) ─────────────────────────────────────

def _synthesise_answer(question: str, chunks: List[dict]) -> dict:
    """
    Extract the best-matching sentences from the retrieved chunk text.
    Answer is derived entirely from the uploaded document.
    """
    time.sleep(0.2)

    combined  = " ".join(c["text"] for c in chunks)
    sentences = re.split(r"(?<=[.!\?\n])\s+", combined)
    q_words   = set(re.findall(r"[a-z0-9₹]+", question.lower()))

    scored = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 15:
            continue
        s_words = set(re.findall(r"[a-z0-9₹]+", sent.lower()))
        overlap = len(q_words & s_words) / (len(q_words) + 1)
        scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored[:3] if s]

    if best:
        answer = " ".join(best)
        if len(answer) > 700:
            answer = answer[:700].rsplit(" ", 1)[0] + "…"
    else:
        answer = "This information is not available in the uploaded document."

    return {
        "answer":           answer,
        "retrieved_chunks": chunks,
        "model":            "chromadb + MiniLM",
        "doc_id":           _doc_id,
    }


# ── Real LLM path ─────────────────────────────────────────────────────────────

def _llm_answer(question: str, chunks: List[dict]) -> dict:
    try:
        import openai

        context    = "\n\n---\n\n".join(
            f"[Chunk {c['id'] + 1}]\n{c['text']}" for c in chunks
        )
        system_msg = (
            "You are a document analysis assistant. "
            "Answer ONLY from the context provided. "
            "If the answer is not in the context say exactly: "
            "'This information is not available in the uploaded document.' "
            "Be concise and cite the chunk number."
        )
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0,
            max_tokens=400,
        )
        return {
            "answer":           resp.choices[0].message.content.strip(),
            "retrieved_chunks": chunks,
            "model":            "openai",
            "tokens_used":      resp.usage.total_tokens,
            "doc_id":           _doc_id,
        }
    except Exception as exc:
        return {
            "answer":           f"LLM error: {exc}",
            "retrieved_chunks": chunks,
            "model":            "openai-error",
        }
