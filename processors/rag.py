"""
Contract RAG processor — ChromaDB vector store.
═══════════════════════════════════════════════════════════════════
Pipeline (called on every Try RAG click):
  1. OCR text  →  _chunk_text()       word-window chunks
  2. chunks    →  ChromaDB.add()      ONNX MiniLM embeddings (bundled)
  3. query     →  ChromaDB.query()    cosine Top-K retrieval
  4. chunks    →  _answer_from_chunks() grounded answer extraction

Source of truth: ONLY the text embedded in step 2.
No hardcoded answers. No mock contract data. No cached static responses.

Real LLM:  set ENABLE_REAL_LLM=true + OPENAI_API_KEY for GPT answers.
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Tuple

ENABLE_REAL_LLM = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"
SAMPLE_DOC_PATH = Path(__file__).parent.parent / "assets" / "sample_invoice.txt"

# ── Module-level state ────────────────────────────────────────────────────────
_client     = None
_collection = None
_doc_id: str = ""
_raw_chunks: List[str] = []          # plain text, parallel to ChromaDB store


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def embed_document(text: str, doc_id: str) -> dict:
    """
    Chunk the OCR-extracted text, compute embeddings, build the vector store.
    Always replaces the previous document — the uploaded file is the only source.
    """
    global _client, _collection, _doc_id, _raw_chunks

    # ── Reset: wipe previous document completely ──────────────────────────────
    if _client is not None:
        try:
            _client.delete_collection("rag_chunks")
        except Exception:
            pass
        _collection = None

    import chromadb
    _client     = chromadb.Client()
    _collection = _client.get_or_create_collection(
        name="rag_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    _doc_id     = doc_id
    _raw_chunks = _chunk_text(text)

    if not _raw_chunks:
        return {"chunks_created": 0, "doc_id": doc_id}

    _collection.add(
        documents=_raw_chunks,
        ids=[f"c{i}" for i in range(len(_raw_chunks))],
        metadatas=[{"chunk_idx": i} for i in range(len(_raw_chunks))],
    )
    return {"chunks_created": len(_raw_chunks), "doc_id": doc_id}


def embed_sample_doc() -> dict:
    """Embed the bundled Beverage Sales Invoice as the demo document."""
    if SAMPLE_DOC_PATH.exists():
        text = SAMPLE_DOC_PATH.read_text(encoding="utf-8")
    else:
        text = (
            "Beverage Sales Invoice. Beverage Distribution Co. "
            "123 Market Street, Bengaluru. Invoice No: INV-2026-1048. "
            "Date: 24 Sep 2026. "
            "Items: Coca-Cola 330ml qty 12 unit $1.25 amount $15.00. "
            "Pepsi 330ml qty 8 unit $1.20 amount $9.60. "
            "Sprite 330ml qty 10 unit $1.15 amount $11.50. "
            "Fanta Orange 330ml qty 6 unit $1.30 amount $7.80. "
            "Red Bull 250ml qty 5 unit $2.80 amount $14.00. "
            "Subtotal $57.90. Tax $5.79. Total $63.69. "
            "Payment Terms: Net 15 Days."
        )
    return embed_document(text, "sample_beverage_invoice")


def query(question: str) -> dict:
    """
    Embed the question, retrieve Top-3 chunks from the vector store,
    and return a grounded answer derived from those chunks only.
    """
    if _collection is None or _collection.count() == 0:
        return {
            "answer": (
                "No document has been processed yet. "
                "Please upload a document and click Start RAG."
            ),
            "retrieved_chunks": [],
        }

    top_chunks = _retrieve(question, top_k=3)

    if ENABLE_REAL_LLM:
        return _llm_answer(question, top_chunks)
    return _answer_from_chunks(question, top_chunks)


# ══════════════════════════════════════════════════════════════════
#  CHUNKING  (600–800 token equivalent in words)
# ══════════════════════════════════════════════════════════════════

def _chunk_text(text: str, size: int = 120, overlap: int = 20) -> List[str]:
    """
    Split text into overlapping word-window chunks.
    120 words ≈ 160 tokens  →  3–5 chunks per typical invoice.
    Smaller chunks = better precision for tabular / numeric data.
    """
    # Preserve line structure inside each chunk for better readability
    lines  = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)
    words  = joined.split()
    if not words:
        return [text]

    result, i = [], 0
    while i < len(words):
        result.append(" ".join(words[i : i + size]))
        i += size - overlap
    return result


# ══════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════════

def _retrieve(question: str, top_k: int = 3) -> List[dict]:
    """Query ChromaDB and return top_k chunks as dicts."""
    k   = min(top_k, _collection.count())
    res = _collection.query(query_texts=[question], n_results=k)

    chunks = []
    for i, text in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i] if res.get("distances") else 0.0
        chunks.append({
            "id":         meta.get("chunk_idx", i),
            "doc_id":     _doc_id,
            "text":       text,
            "similarity": round(1.0 - dist, 3),   # cosine distance → similarity
        })
    return chunks


# ══════════════════════════════════════════════════════════════════
#  ANSWER GENERATION (no external API)
# ══════════════════════════════════════════════════════════════════

def _answer_from_chunks(question: str, chunks: List[dict]) -> dict:
    """
    Build a grounded answer from the retrieved chunks.

    Strategy:
    1. Scan every line in the retrieved chunks.
    2. Score each line by token overlap with the question.
    3. Pick the top-scoring lines as the answer.
    4. For numeric/quantity questions, also surface lines containing digits.

    The answer is ONLY constructed from the retrieved chunk text —
    no hardcoded strings, no fallback contract data.
    """
    time.sleep(0.15)

    q_tokens = set(re.findall(r"[a-z0-9$₹%]+", question.lower()))

    # Collect all lines from retrieved chunks (preserve order)
    candidate_lines: List[Tuple[float, str, int]] = []  # (score, line, chunk_idx)
    for cidx, chunk in enumerate(chunks):
        for line in re.split(r"[\n.!?]+", chunk["text"]):
            line = line.strip()
            if len(line) < 8:
                continue
            l_tokens = set(re.findall(r"[a-z0-9$₹%]+", line.lower()))
            overlap  = len(q_tokens & l_tokens)

            # Boost lines with numbers when question asks for amounts/quantities
            numeric_boost = 0.0
            if re.search(r"\d", line):
                num_q_words = {"total", "amount", "qty", "quantity", "price",
                               "subtotal", "tax", "cost", "how many", "how much",
                               "sum", "value", "number", "count", "invoice"}
                if q_tokens & num_q_words:
                    numeric_boost = 1.5

            score = overlap + numeric_boost
            candidate_lines.append((score, line, cidx))

    # Sort by score descending, deduplicate near-identical lines
    candidate_lines.sort(key=lambda x: x[0], reverse=True)
    seen, best_lines = set(), []
    for score, line, _ in candidate_lines:
        normalised = re.sub(r"\s+", " ", line.lower())
        if normalised not in seen and score > 0:
            seen.add(normalised)
            best_lines.append(line)
        if len(best_lines) == 4:
            break

    if best_lines:
        answer = "  \n".join(best_lines)          # markdown line breaks
    else:
        # Fall back: return the first sentence of the top chunk verbatim
        first_chunk_text = chunks[0]["text"] if chunks else ""
        sentences = re.split(r"(?<=[.!?])\s+", first_chunk_text)
        non_empty = [s.strip() for s in sentences if len(s.strip()) > 10]
        answer = non_empty[0] if non_empty else \
                 "This information is not available in the uploaded document."

    return {
        "answer":           answer,
        "retrieved_chunks": chunks,
        "model":            "chromadb-MiniLM + extraction",
        "doc_id":           _doc_id,
    }


# ══════════════════════════════════════════════════════════════════
#  REAL LLM PATH (OpenAI-compatible)
# ══════════════════════════════════════════════════════════════════

def _llm_answer(question: str, chunks: List[dict]) -> dict:
    """Send retrieved chunks + question to the configured LLM."""
    try:
        import openai

        context = "\n\n---\n\n".join(
            f"[Chunk {c['id'] + 1}  similarity={c.get('similarity','?')}]\n{c['text']}"
            for c in chunks
        )
        system_msg = (
            "You are a document analysis assistant. "
            "Answer the user's question using ONLY the context below. "
            "If the answer cannot be found in the context, respond exactly: "
            "'This information is not available in the uploaded document.' "
            "Be concise, factual, and cite the chunk number when relevant."
        )
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"},
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
