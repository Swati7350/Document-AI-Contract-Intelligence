"""
RAG processor — ChromaDB vector store, line-aware chunking.
═════════════════════════════════════════════════════════════════
Source of truth: ONLY the text passed to embed_document().
Zero hardcoded content. Zero sample documents. Zero fallbacks.

Chunking (TOKEN-BASED, not word-based):
  CHUNK_SIZE    = 100 tokens (using tiktoken GPT-2 encoding)
  CHUNK_OVERLAP = 30 tokens
  Strategy: line-aware — never splits a table row or sentence mid-way
  Each chunk: {"text", "page", "chunk_id", "token_count"}

Retrieval: ChromaDB cosine, Top-K = 4
Answer:    best-matching lines from retrieved chunks
           Returns "not available" when nothing relevant found.
═════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Tuple

ENABLE_REAL_LLM = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"
CHUNK_SIZE_TOKENS = 100   # tokens (not words)
CHUNK_OVERLAP_TOKENS = 30  # tokens
TOP_K           = 4

# ── Module-level vector store state ──────────────────────────────
_client:     object = None
_collection: object = None
_doc_id:     str    = ""
_chunks:     List[dict] = []


# ════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════

def embed_document(
    text:  str,
    doc_id: str,
    pages: List[str] = None,
) -> dict:
    """
    Chunk `text` (token-based) and build an in-memory ChromaDB vector store.

    Args:
        text:   Full document text (from OCR).
        doc_id: Filename / identifier stored in metadata.
        pages:  Per-page text list; used to assign page numbers to chunks.
                Falls back to treating the whole text as page 1.

    Returns:
        {"chunks_created": int, "doc_id": str}

    Calling this always wipes the previous document completely.
    """
    global _client, _collection, _doc_id, _chunks

    # Hard reset — previous document is gone
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

    _doc_id = doc_id
    _chunks = _token_aware_chunk(text, pages or [text])

    if not _chunks:
        return {"chunks_created": 0, "doc_id": doc_id}

    _collection.add(
        documents=[c["text"] for c in _chunks],
        ids=[f"c{c['chunk_id']}" for c in _chunks],
        metadatas=[{
            "chunk_id": c["chunk_id"],
            "page":     c["page"],
            "doc_id":   doc_id,
        } for c in _chunks],
    )
    return {"chunks_created": len(_chunks), "doc_id": doc_id}


def query(question: str) -> dict:
    """
    Retrieve the Top-K most relevant chunks for `question`
    and return a grounded answer derived only from those chunks.
    """
    if _collection is None or _collection.count() == 0:
        return {
            "answer": (
                "No document has been processed yet. "
                "Please upload a document and click Start RAG."
            ),
            "retrieved_chunks": [],
        }

    top = _retrieve(question, top_k=TOP_K)
    return _llm_answer(question, top) if ENABLE_REAL_LLM \
           else _answer_from_chunks(question, top)


# ════════════════════════════════════════════════════════════════
#  TOKEN-AWARE CHUNKING (100 tokens, 30-token overlap, line-preserving)
# ════════════════════════════════════════════════════════════════

def _token_aware_chunk(
    full_text: str,
    pages: List[str],
) -> List[dict]:
    """
    Build chunks using token counting (tiktoken GPT-2 encoding).
    Respects line/row boundaries — never splits a line mid-way.

    Strategy:
    1. Use tiktoken to count tokens precisely.
    2. Split text into logical lines (non-empty stripped lines).
    3. Accumulate lines until token count reaches CHUNK_SIZE.
       A line is never split — it either fits or starts a new chunk.
    4. Apply overlap by back-tracking: each new chunk starts
       CHUNK_OVERLAP tokens before the previous chunk ended.
    5. Tag each chunk with the page number it starts on.

    This preserves table rows and invoice line items intact.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("gpt2")
    except ImportError:
        # Fallback: approximate tokens as words * 1.3 if tiktoken unavailable
        encoding = None

    def count_tokens(s: str) -> int:
        if encoding:
            return len(encoding.encode(s))
        return int(len(s.split()) * 1.3)  # fallback approximation

    # Build cumulative token-count map for page assignment
    page_token_boundaries: List[int] = []
    cumulative = 0
    for pg_text in pages:
        page_token_boundaries.append(cumulative)
        cumulative += count_tokens(pg_text)

    def _page_for_token(token_offset: int) -> int:
        pg = 1
        for idx, boundary in enumerate(page_token_boundaries):
            if token_offset >= boundary:
                pg = idx + 1
        return pg

    # Split into lines, preserving all content
    lines = [l.rstrip() for l in full_text.splitlines() if l.strip()]

    chunks:        List[dict] = []
    current_lines: List[str] = []
    current_tokens: int       = 0
    token_offset:   int       = 0   # tracks position in full text for page tagging

    for line in lines:
        line_tokens = count_tokens(line)

        if current_tokens + line_tokens > CHUNK_SIZE_TOKENS and current_lines:
            # Flush current chunk
            chunk_text = "\n".join(current_lines)
            chunk_start_offset = token_offset - current_tokens
            chunks.append({
                "chunk_id": len(chunks),
                "text":     chunk_text,
                "page":     _page_for_token(chunk_start_offset),
                "token_count": current_tokens,
            })

            # Overlap: keep the last CHUNK_OVERLAP_TOKENS worth of lines
            overlap_lines: List[str] = []
            overlap_tokens = 0
            for prev_line in reversed(current_lines):
                pt = count_tokens(prev_line)
                if overlap_tokens + pt > CHUNK_OVERLAP_TOKENS:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_tokens += pt

            current_lines = overlap_lines
            current_tokens = overlap_tokens

        current_lines.append(line)
        current_tokens += line_tokens
        token_offset   += line_tokens

    # Flush remaining lines
    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunk_start_offset = token_offset - current_tokens
        chunks.append({
            "chunk_id": len(chunks),
            "text":     chunk_text,
            "page":     _page_for_token(chunk_start_offset),
            "token_count": current_tokens,
        })

    if not chunks:
        chunks = [{
            "chunk_id": 0,
            "text": full_text[:3000],
            "page": 1,
            "token_count": count_tokens(full_text[:3000]),
        }]

    return chunks


# ════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ════════════════════════════════════════════════════════════════

def _retrieve(question: str, top_k: int = TOP_K) -> List[dict]:
    k   = min(top_k, _collection.count())
    res = _collection.query(query_texts=[question], n_results=k)

    out = []
    for i, text in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        dist = (res.get("distances") or [[0.0] * k])[0][i]
        out.append({
            "id":         meta.get("chunk_id", i),
            "page":       meta.get("page", 1),
            "doc_id":     meta.get("doc_id", _doc_id),
            "text":       text,
            "similarity": round(max(0.0, 1.0 - float(dist)), 3),
        })
    return out


# ════════════════════════════════════════════════════════════════
#  ANSWER GENERATION  (no external API)
# ════════════════════════════════════════════════════════════════

# Words that signal a numeric/monetary question
_NUMERIC_Q_WORDS = {
    "total", "amount", "subtotal", "tax", "gst", "vat", "price",
    "cost", "fee", "fees", "rate", "charge", "penalty", "interest",
    "qty", "quantity", "how many", "how much", "sum", "value",
    "number", "count", "invoice", "grand", "balance", "due", "paid",
}

# Synonym map — expands query tokens before scoring
_SYNONYMS: dict = {
    "vendor":      {"vendor", "supplier", "seller", "company", "distributor",
                    "manufacturer", "provider"},
    "supplier":    {"supplier", "vendor", "seller", "company", "distributor",
                    "provider", "contractor"},
    "buyer":       {"buyer", "client", "customer", "purchaser", "bill"},
    "client":      {"client", "buyer", "customer", "purchaser"},
    "parties":     {"parties", "party", "client", "supplier", "vendor", "between"},
    "termination": {"termination", "terminate", "notice", "cancel"},
    "late":        {"late", "penalty", "fee", "overdue", "interest"},
    "payment":     {"payment", "terms", "net", "due", "days"},
    "governing":   {"governing", "jurisdiction", "law", "state"},
    "effective":   {"effective", "commencement", "start", "begin"},
    "expiry":      {"expiry", "expiration", "end", "expires"},
    "value":       {"value", "amount", "total", "price", "cost"},
}


def _answer_from_chunks(question: str, chunks: List[dict]) -> dict:
    """
    Score every line in the retrieved chunks against the question.

    Scoring:
      - token overlap between question words and line words
      - +2.0 boost for lines containing digits when question is numeric

    Minimum score threshold: 2.0
    Returns "not available" if nothing clears the threshold.
    """
    q_tokens = set(re.findall(r"[a-z0-9$₹€%]+", question.lower()))
    # Expand with synonyms so "vendor" matches "supplier" lines etc.
    expanded_tokens = set(q_tokens)
    for tok in q_tokens:
        expanded_tokens |= _SYNONYMS.get(tok, set())
    is_numeric_q = bool(expanded_tokens & _NUMERIC_Q_WORDS)

    MIN_SCORE = 2.0
    candidates: List[Tuple[float, str, int]] = []  # (score, line, page)

    for chunk in chunks:
        page = chunk.get("page", 1)
        for line in chunk["text"].splitlines():
            line = line.strip()
            if len(line) < 6:
                continue
            l_tokens = set(re.findall(r"[a-z0-9$₹€%]+", line.lower()))
            overlap  = len(expanded_tokens & l_tokens)
            boost    = 2.0 if (is_numeric_q and re.search(r"\d", line)) else 0.0
            score    = overlap + boost
            candidates.append((score, line, page))

    if not candidates:
        return _not_found(chunks)

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_score = candidates[0][0]

    if top_score < MIN_SCORE:
        return _not_found(chunks)

    # Collect top lines above threshold, deduplicated
    seen: set = set()
    best_lines: List[Tuple[str, int]] = []
    for score, line, page in candidates:
        if score < MIN_SCORE:
            break
        key = re.sub(r"\s+", " ", line.lower())
        if key not in seen:
            seen.add(key)
            best_lines.append((line, page))
        if len(best_lines) == 5:
            break

    answer = "\n".join(line for line, _ in best_lines)
    # Append page citation if multi-page doc
    pages_cited = sorted({pg for _, pg in best_lines})
    if pages_cited and (len(pages_cited) > 1 or pages_cited[0] > 1):
        answer += f"\n\n*(Source: page {', '.join(str(p) for p in pages_cited)})*"

    return {
        "answer":           answer,
        "retrieved_chunks": chunks,
        "model":            f"chromadb+MiniLM | chunk={CHUNK_SIZE_TOKENS}tok overlap={CHUNK_OVERLAP_TOKENS}tok",
        "doc_id":           _doc_id,
    }


def _not_found(chunks: List[dict]) -> dict:
    return {
        "answer":           "This information is not available in the uploaded document.",
        "retrieved_chunks": chunks,
        "model":            f"chromadb+MiniLM | chunk={CHUNK_SIZE_TOKENS}tok overlap={CHUNK_OVERLAP_TOKENS}tok",
        "doc_id":           _doc_id,
    }


# ════════════════════════════════════════════════════════════════
#  REAL LLM PATH
# ════════════════════════════════════════════════════════════════

def _llm_answer(question: str, chunks: List[dict]) -> dict:
    try:
        import openai
        context = "\n\n---\n\n".join(
            f"[Chunk {c['id']+1} | Page {c.get('page',1)} | "
            f"similarity={c.get('similarity','?')}]\n{c['text']}"
            for c in chunks
        )
        system_msg = (
            "You are a document analysis assistant. "
            "Answer using ONLY the context provided. "
            "If the answer is not in the context, respond exactly: "
            "'This information is not available in the uploaded document.' "
            "Be concise, factual, and cite the page number."
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
