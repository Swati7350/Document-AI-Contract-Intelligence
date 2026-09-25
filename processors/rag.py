"""
Contract RAG processor — ChromaDB vector store.
═══════════════════════════════════════════════════════════════════
Pipeline:
  1. text + pages  →  _recursive_chunk()   CHUNK_SIZE=100, OVERLAP=30
  2. chunks        →  ChromaDB.add()       ONNX MiniLM cosine embeddings
  3. query         →  ChromaDB.query()     Top-K retrieval
  4. chunks        →  _answer_from_chunks() grounded extraction

Chunking config (fixed):
  CHUNK_SIZE    = 100   # words (~130 tokens)
  CHUNK_OVERLAP = 30    # words
  Recursive splitting: paragraph → sentence → word boundaries
  Page number stored in every chunk metadata.

Source of truth: ONLY the text from embed_document().
No hardcoded answers. No sample contract data in the answer path.
═══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Tuple

ENABLE_REAL_LLM  = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"
CHUNK_SIZE       = 100   # words
CHUNK_OVERLAP    = 30    # words
SAMPLE_PDF_PATH  = Path(__file__).parent.parent / "assets" / "sample_beverage_invoice.pdf"

# ── Module-level state ────────────────────────────────────────────────────────
_client     = None
_collection = None
_doc_id: str = ""
_chunks: List[dict] = []   # [{text, page, chunk_idx}]


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def embed_document(text: str, doc_id: str, pages: List[str] = None) -> dict:
    """
    Chunk the text recursively, embed each chunk, store in ChromaDB.
    Always wipes the previous collection — the uploaded file is the only source.

    Args:
        text:    Full extracted text of the document.
        doc_id:  Filename or identifier (used for metadata only).
        pages:   Optional list of per-page texts for page-level metadata.
                 If None, the whole text is treated as page 1.
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
    _chunks = _recursive_chunk(text, pages or [text])

    if not _chunks:
        return {"chunks_created": 0, "doc_id": doc_id}

    _collection.add(
        documents=[c["text"]      for c in _chunks],
        ids=[f"c{i}"              for i in range(len(_chunks))],
        metadatas=[{
            "chunk_idx": c["chunk_idx"],
            "page":      c["page"],
            "doc_id":    doc_id,
        } for c in _chunks],
    )
    return {"chunks_created": len(_chunks), "doc_id": doc_id}


def embed_sample_doc() -> dict:
    """
    Embed the bundled sample PDF (Beverage Sales Invoice).
    Reads the real PDF bytes through the OCR extractor — same path as an upload.
    Falls back to inline text if the PDF is missing.
    """
    if SAMPLE_PDF_PATH.exists():
        from processors.ocr import extract_text
        text = extract_text(SAMPLE_PDF_PATH.read_bytes(), SAMPLE_PDF_PATH.name)
    else:
        text = ""

    if not text or len(text.strip()) < 30:
        # Inline fallback — Beverage invoice only, never the Service Agreement
        text = (
            "Beverage Sales Invoice\n"
            "Beverage Distribution Co., 123 Market Street, Bengaluru\n"
            "Invoice No: INV-2026-1048  Date: 24 Sep 2026\n"
            "Item               Qty  Unit Price  Amount\n"
            "Coca-Cola 330ml     12  $1.25       $15.00\n"
            "Pepsi 330ml          8  $1.20        $9.60\n"
            "Sprite 330ml        10  $1.15       $11.50\n"
            "Fanta Orange 330ml   6  $1.30        $7.80\n"
            "Red Bull 250ml       5  $2.80       $14.00\n"
            "Subtotal: $57.90  Tax: $5.79  Total: $63.69\n"
            "Payment Terms: Net 15 Days\n"
        )

    return embed_document(text, "sample_beverage_invoice.pdf")


def query(question: str) -> dict:
    """Embed question, retrieve Top-3 chunks, return a grounded answer."""
    if _collection is None or _collection.count() == 0:
        return {
            "answer": (
                "No document has been processed yet. "
                "Please upload a document and click Start RAG."
            ),
            "retrieved_chunks": [],
        }
    top = _retrieve(question, top_k=3)
    return _llm_answer(question, top) if ENABLE_REAL_LLM \
           else _answer_from_chunks(question, top)


# ══════════════════════════════════════════════════════════════════
#  RECURSIVE CHUNKING  (CHUNK_SIZE=100, OVERLAP=30)
# ══════════════════════════════════════════════════════════════════

def _recursive_chunk(full_text: str, pages: List[str]) -> List[dict]:
    """
    Recursive text splitting strategy:
      1. Split on paragraph boundaries (\\n\\n or \\n followed by blank line).
      2. If a paragraph exceeds CHUNK_SIZE words, split on sentence boundaries.
      3. If a sentence still exceeds CHUNK_SIZE words, split on word boundaries.
    Overlap is applied at the word level across all resulting chunks.
    Each chunk carries a page number derived from the pages list.
    """
    # Build a page-boundary map so we can tag each chunk with a page number
    page_offsets: List[int] = []
    offset = 0
    for pg in pages:
        page_offsets.append(offset)
        offset += len(pg.split())

    # Step 1 — paragraph split
    paragraphs = re.split(r"\n\s*\n|\r\n\s*\r\n", full_text)
    sentences: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= CHUNK_SIZE:
            sentences.append(para)
        else:
            # Step 2 — sentence split within long paragraphs
            sents = re.split(r"(?<=[.!?])\s+", para)
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                s_words = sent.split()
                if len(s_words) <= CHUNK_SIZE:
                    sentences.append(sent)
                else:
                    # Step 3 — hard word-window split
                    for i in range(0, len(s_words), CHUNK_SIZE):
                        sentences.append(" ".join(s_words[i:i + CHUNK_SIZE]))

    # Apply overlap: sliding window over the sentence list by word count
    all_words: List[Tuple[str, int]] = []   # (word, sentence_idx)
    for si, sent in enumerate(sentences):
        for w in sent.split():
            all_words.append((w, si))

    chunks: List[dict] = []
    i = 0
    while i < len(all_words):
        window = all_words[i : i + CHUNK_SIZE]
        chunk_text = " ".join(w for w, _ in window)
        # Page number: find which page the first word of this chunk belongs to
        word_pos = i
        page_num = 1
        for pg_idx, pg_start in enumerate(reversed(page_offsets)):
            if word_pos >= pg_start:
                page_num = len(page_offsets) - pg_idx
                break

        chunks.append({
            "chunk_idx": len(chunks),
            "text":      chunk_text,
            "page":      page_num,
        })
        step = CHUNK_SIZE - CHUNK_OVERLAP
        i += max(step, 1)

    return chunks if chunks else [{"chunk_idx": 0, "text": full_text[:2000], "page": 1}]


# ══════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════════

def _retrieve(question: str, top_k: int = 3) -> List[dict]:
    k   = min(top_k, _collection.count())
    res = _collection.query(query_texts=[question], n_results=k)

    out = []
    for i, text in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        dist = (res.get("distances") or [[0.0] * k])[0][i]
        out.append({
            "id":         meta.get("chunk_idx", i),
            "page":       meta.get("page", 1),
            "doc_id":     meta.get("doc_id", _doc_id),
            "text":       text,
            "similarity": round(max(0.0, 1.0 - dist), 3),
        })
    return out


# ══════════════════════════════════════════════════════════════════
#  ANSWER GENERATION (no external API)
# ══════════════════════════════════════════════════════════════════

def _answer_from_chunks(question: str, chunks: List[dict]) -> dict:
    """
    Extract the best-matching lines from retrieved chunks.
    Answer comes ONLY from chunk text — no hardcoded strings.
    Numeric/quantity questions get a score boost for lines containing digits.
    """
    time.sleep(0.1)

    q_tokens    = set(re.findall(r"[a-z0-9$₹%]+", question.lower()))
    num_q_words = {"total", "amount", "qty", "quantity", "price", "subtotal",
                   "tax", "cost", "how many", "how much", "sum", "value",
                   "number", "count", "invoice", "grand"}

    candidates: List[Tuple[float, str]] = []
    for chunk in chunks:
        for line in re.split(r"[\n.!?]+", chunk["text"]):
            line = line.strip()
            if len(line) < 8:
                continue
            l_tokens = set(re.findall(r"[a-z0-9$₹%]+", line.lower()))
            overlap  = len(q_tokens & l_tokens)
            boost    = 1.5 if (re.search(r"\d", line) and q_tokens & num_q_words) else 0.0
            candidates.append((overlap + boost, line))

    candidates.sort(key=lambda x: x[0], reverse=True)

    seen, best = set(), []
    for score, line in candidates:
        key = re.sub(r"\s+", " ", line.lower())
        if key not in seen and score > 0:
            seen.add(key)
            best.append(line)
        if len(best) == 4:
            break

    if best:
        answer = "  \n".join(best)
    else:
        raw = chunks[0]["text"] if chunks else ""
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if len(s.strip()) > 10]
        answer = sents[0] if sents else \
                 "This information is not available in the uploaded document."

    return {
        "answer":           answer,
        "retrieved_chunks": chunks,
        "model":            f"chromadb+MiniLM (chunk={CHUNK_SIZE}/overlap={CHUNK_OVERLAP})",
        "doc_id":           _doc_id,
    }


# ══════════════════════════════════════════════════════════════════
#  REAL LLM PATH
# ══════════════════════════════════════════════════════════════════

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
            "If the answer is not in the context, say: "
            "'This information is not available in the uploaded document.' "
            "Be concise, factual, and cite the chunk and page number."
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
