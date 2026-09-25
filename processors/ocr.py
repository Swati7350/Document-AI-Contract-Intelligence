"""
OCR & Layout Parsing processor.
────────────────────────────────────────────────────────────────
PDF text extraction : pypdf  (handles all standard PDF encodings)
Image text          : returns demo fallback (real OCR engine needed)
Demo fallback       : Beverage Sales Invoice — never the Service Agreement
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import io, os, re, time, random
from typing import List, Tuple

OCR_ENGINE = os.getenv("OCR_ENGINE", "mock")


# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def process_document(file_bytes: bytes, filename: str) -> dict:
    if OCR_ENGINE != "mock":
        return _real_process(file_bytes, filename)
    return _mock_process(file_bytes, filename)


# ══════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Public helper — returns plain text from any supported file.
    Used by both OCR display pipeline and RAG embed pipeline.
    """
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return _extract_pdf(file_bytes)
    # Images require a real OCR engine; return empty so callers use demo
    return ""


def _extract_pdf(data: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf.
    Returns full text as a single string.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages  = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def extract_pages(data: bytes) -> List[str]:
    """
    Extract per-page text list from a PDF.
    Returns ["page1 text", "page2 text", …].
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages  = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            pages.append(text)
        return pages if any(pages) else []
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
#  MOCK PROCESSOR — uses real extracted text, beverage demo fallback
# ══════════════════════════════════════════════════════════════════

def _mock_process(file_bytes: bytes, filename: str) -> dict:
    time.sleep(0.5)
    is_pdf = filename.lower().endswith(".pdf")

    # ── Extract real text ─────────────────────────────────────────
    real_text  = extract_text(file_bytes, filename) if file_bytes else ""
    page_texts = extract_pages(file_bytes) if (file_bytes and is_pdf) else []

    if real_text and len(real_text.strip()) > 40:
        text_blocks, tables, full_text = _build_from_text(real_text)
        source = "extracted"
    else:
        # No extractable text (scanned image / empty PDF) → beverage demo
        full_text, text_blocks, tables = _beverage_demo()
        page_texts = [full_text]
        source = "demo"

    layout = {
        "pages":            len(page_texts) if page_texts else (2 if is_pdf else 1),
        "columns_detected": 1,
        "stamps_seals":     [],
        "headers_footers":  True,
        "tables_count":     len(tables),
        "signatures_count": 0,
        "bounding_boxes": [
            {"id": 1, "label": "Heading",   "x": 0.10, "y": 0.05, "w": 0.80, "h": 0.06, "color": "#3b82f6"},
            {"id": 2, "label": "Paragraph", "x": 0.08, "y": 0.14, "w": 0.84, "h": 0.22, "color": "#10b981"},
            {"id": 3, "label": "Table",     "x": 0.08, "y": 0.40, "w": 0.84, "h": 0.30, "color": "#f59e0b"},
        ],
    }

    return {
        "engine":            f"mock-{source}",
        "filename":          filename,
        "text_blocks":       text_blocks,
        "full_text":         full_text,
        "page_texts":        page_texts or [full_text],   # always present
        "tables":            tables,
        "layout":            layout,
        "avg_confidence":    round(
            sum(b["confidence"] for b in text_blocks) / max(len(text_blocks), 1), 3
        ),
        "processing_time_ms": random.randint(300, 700),
    }


def _build_from_text(text: str) -> Tuple[List[dict], List[dict], str]:
    """Build text_blocks and tables from real extracted text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    text_blocks: List[dict] = []
    for i, line in enumerate(lines[:30]):
        block_type = "heading" if (i == 0 or line.isupper() or
                                    (len(line) < 60 and line.endswith(":"))) \
                     else "paragraph"
        text_blocks.append({
            "id":         i + 1,
            "type":       block_type,
            "text":       line,
            "confidence": round(random.uniform(0.88, 0.97), 2),
        })

    tables = _detect_tables(lines)
    full_text = "\n".join(lines)
    return text_blocks, tables, full_text


def _detect_tables(lines: List[str]) -> List[dict]:
    """Detect table rows: lines containing currency, digits+whitespace, or pipes."""
    pattern = re.compile(r"(\$|₹|€|£|\d+\.\d{2}|\d+\s{3,}\d)")
    table_lines = [l for l in lines if pattern.search(l) and len(l) > 8]
    if len(table_lines) < 2:
        return []

    rows = []
    for l in table_lines[:10]:
        # Split on 2+ spaces, tabs, or pipes
        cols = [c.strip() for c in re.split(r"\s{2,}|\t|\|", l) if c.strip()]
        if cols:
            rows.append(cols)

    if not rows:
        return []

    max_cols = max(len(r) for r in rows)
    return [{
        "id":      1,
        "title":   "Extracted Table",
        "headers": rows[0] if len(rows[0]) == max_cols else
                   [f"Col {i+1}" for i in range(max_cols)],
        "rows":    rows[1:] if len(rows[0]) == max_cols else rows,
    }]


# ══════════════════════════════════════════════════════════════════
#  BEVERAGE DEMO  — fallback ONLY for non-extractable uploads
# ══════════════════════════════════════════════════════════════════

def _beverage_demo() -> Tuple[str, List[dict], List[dict]]:
    text_blocks = [
        {"id":  1, "type": "heading",   "text": "Beverage Sales Invoice",                                "confidence": 0.99},
        {"id":  2, "type": "paragraph", "text": "Beverage Distribution Co., 123 Market Street, Bengaluru","confidence": 0.97},
        {"id":  3, "type": "paragraph", "text": "Invoice No: INV-2026-1048  |  Date: 24 Sep 2026",       "confidence": 0.97},
        {"id":  4, "type": "paragraph", "text": "Coca-Cola 330ml   Qty: 12   Unit: $1.25   Amount: $15.00","confidence": 0.96},
        {"id":  5, "type": "paragraph", "text": "Pepsi 330ml       Qty:  8   Unit: $1.20   Amount: $9.60", "confidence": 0.96},
        {"id":  6, "type": "paragraph", "text": "Sprite 330ml      Qty: 10   Unit: $1.15   Amount: $11.50","confidence": 0.96},
        {"id":  7, "type": "paragraph", "text": "Fanta Orange 330ml Qty: 6   Unit: $1.30   Amount: $7.80", "confidence": 0.95},
        {"id":  8, "type": "paragraph", "text": "Red Bull 250ml    Qty:  5   Unit: $2.80   Amount: $14.00","confidence": 0.95},
        {"id":  9, "type": "paragraph", "text": "Subtotal: $57.90  |  Tax (10%): $5.79  |  Total: $63.69","confidence": 0.97},
        {"id": 10, "type": "paragraph", "text": "Payment Terms: Net 15 Days",                             "confidence": 0.98},
    ]
    tables = [{
        "id": 1, "title": "Items Purchased",
        "headers": ["Item", "Qty", "Unit Price", "Amount"],
        "rows": [
            ["Coca-Cola 330ml",     "12", "$1.25", "$15.00"],
            ["Pepsi 330ml",          "8", "$1.20",  "$9.60"],
            ["Sprite 330ml",        "10", "$1.15", "$11.50"],
            ["Fanta Orange 330ml",   "6", "$1.30",  "$7.80"],
            ["Red Bull 250ml",       "5", "$2.80", "$14.00"],
        ],
    }]
    full_text = "\n".join(b["text"] for b in text_blocks)
    return full_text, text_blocks, tables


# ══════════════════════════════════════════════════════════════════
#  REAL OCR STUB
# ══════════════════════════════════════════════════════════════════

def _real_process(file_bytes: bytes, filename: str) -> dict:
    raise NotImplementedError(f"Real OCR engine '{OCR_ENGINE}' not wired yet.")
