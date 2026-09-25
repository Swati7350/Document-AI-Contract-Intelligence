"""
OCR & Layout Parsing processor.
Mock mode: extracts real text from uploaded files (PDF/image via basic parsing),
           then wraps results in structured layout metadata.
Real mode: swap OCR_ENGINE=docling or tesseract in .env.
"""
from __future__ import annotations
import os, time, random, re

OCR_ENGINE = os.getenv("OCR_ENGINE", "mock")


def process_document(file_bytes: bytes, filename: str) -> dict:
    """
    Entry point. Returns a standardised result dict.
    In mock mode: extracts real text from the uploaded file bytes.
    In real mode: delegates to the configured OCR engine.
    """
    if OCR_ENGINE != "mock":
        return _real_process(file_bytes, filename)
    return _mock_process(file_bytes, filename)


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from uploaded PDF or image bytes.
    PDF  → parse cross-reference table for embedded text streams.
    Image → return empty string (real OCR needed for images in mock mode).
    Falls back to empty string on any error.
    """
    fname_lower = filename.lower()

    if fname_lower.endswith(".pdf"):
        return _extract_pdf_text(file_bytes)

    if fname_lower.endswith((".png", ".jpg", ".jpeg")):
        # Real text extraction from images requires an OCR engine.
        # Return empty string here; caller will use the demo fallback.
        return ""

    return ""


def _extract_pdf_text(data: bytes) -> str:
    """
    Lightweight pure-Python PDF text extraction.
    Decodes FlateDecode (zlib) compressed content streams and strips
    PDF operators to return raw readable text.
    """
    try:
        import zlib

        text_parts: list[str] = []

        # Find all compressed or uncompressed stream blocks
        stream_re = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.DOTALL)
        for match in stream_re.finditer(data):
            raw = match.group(1)
            # Try zlib decompression (FlateDecode)
            try:
                raw = zlib.decompress(raw)
            except Exception:
                pass  # already plain text or unsupported filter

            decoded = raw.decode("latin-1", errors="ignore")

            # Extract text from BT...ET blocks (PDF text operators)
            bt_re = re.compile(r"BT(.*?)ET", re.DOTALL)
            for bt in bt_re.finditer(decoded):
                block = bt.group(1)
                # Tj / TJ operators carry the actual text
                tj_re = re.compile(r"\(([^)]*)\)\s*Tj|\[(.*?)\]\s*TJ", re.DOTALL)
                for tj in tj_re.finditer(block):
                    part = tj.group(1) or tj.group(2) or ""
                    # TJ arrays: extract parenthesised strings
                    if tj.group(2):
                        part = " ".join(re.findall(r"\(([^)]*)\)", part))
                    part = part.replace("\\n", "\n").replace("\\r", "").strip()
                    if part:
                        text_parts.append(part)

        extracted = "\n".join(text_parts).strip()
        return extracted if len(extracted) > 30 else ""

    except Exception:
        return ""


# ── Mock processor ────────────────────────────────────────────────────────────

def _mock_process(file_bytes: bytes, filename: str) -> dict:
    """
    Process the uploaded file.
    Uses real extracted text when available; falls back to demo data only
    when the file yields no text (e.g. scanned image with no OCR engine).
    """
    time.sleep(0.6)
    is_pdf = filename.lower().endswith(".pdf")

    # ── Try to extract real text from the file ────────────────────────────────
    real_text = _extract_text_from_bytes(file_bytes, filename) if file_bytes else ""

    if real_text and len(real_text.strip()) > 40:
        # Use the real extracted text, split into logical blocks
        raw_lines = [l.strip() for l in real_text.splitlines() if l.strip()]
        text_blocks = [
            {"id": i + 1,
             "type": "paragraph" if i > 0 else "heading",
             "text": line,
             "confidence": round(random.uniform(0.88, 0.97), 2)}
            for i, line in enumerate(raw_lines[:20])     # cap display at 20 blocks
        ]
        full_text = real_text
        tables = _detect_tables_from_text(real_text)
        source = "extracted"
    else:
        # No extractable text → use the sample Beverage Invoice as demo content
        full_text, text_blocks, tables = _beverage_demo()
        source = "demo"

    layout = {
        "pages":              2 if is_pdf else 1,
        "columns_detected":   1,
        "stamps_seals":       [],
        "headers_footers":    True,
        "tables_count":       len(tables),
        "signatures_count":   0,
        "bounding_boxes": [
            {"id": 1, "label": "Heading",   "x": 0.10, "y": 0.05, "w": 0.80, "h": 0.06, "color": "#3b82f6"},
            {"id": 2, "label": "Paragraph", "x": 0.08, "y": 0.14, "w": 0.84, "h": 0.22, "color": "#10b981"},
            {"id": 3, "label": "Table",     "x": 0.08, "y": 0.40, "w": 0.84, "h": 0.30, "color": "#f59e0b"},
        ],
    }

    return {
        "engine":           f"mock-{source}",
        "filename":         filename,
        "text_blocks":      text_blocks,
        "full_text":        full_text,
        "tables":           tables,
        "layout":           layout,
        "avg_confidence":   round(sum(b["confidence"] for b in text_blocks) /
                                  max(len(text_blocks), 1), 3),
        "processing_time_ms": random.randint(400, 900),
    }


def _detect_tables_from_text(text: str) -> list:
    """Heuristically detect table-like rows (lines with consistent separators)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    table_rows = [l for l in lines if re.search(r"\t|\s{3,}|\||\$|₹", l)]
    if len(table_rows) < 3:
        return []
    headers = re.split(r"\s{2,}|\t|\|", table_rows[0])
    rows    = [re.split(r"\s{2,}|\t|\|", r) for r in table_rows[1:7]]
    return [{
        "id":      1,
        "title":   "Detected Table",
        "headers": [h.strip() for h in headers if h.strip()],
        "rows":    [[c.strip() for c in row if c.strip()] for row in rows if any(c.strip() for c in row)],
    }]


# ── Beverage demo fallback ────────────────────────────────────────────────────

def _beverage_demo() -> tuple:
    """
    Returns (full_text, text_blocks, tables) for the Beverage Sales Invoice demo.
    Used ONLY when the uploaded file yields no extractable text.
    """
    text_blocks = [
        {"id": 1, "type": "heading",   "text": "Beverage Sales Invoice",           "confidence": 0.99},
        {"id": 2, "type": "paragraph", "text": "Beverage Distribution Co., 123 Market Street, Bengaluru", "confidence": 0.97},
        {"id": 3, "type": "paragraph", "text": "Invoice No: INV-2026-1048  |  Date: 24 Sep 2026",       "confidence": 0.97},
        {"id": 4, "type": "paragraph", "text": "Coca-Cola 330ml  Qty: 12  Unit: $1.25  Amount: $15.00", "confidence": 0.96},
        {"id": 5, "type": "paragraph", "text": "Pepsi 330ml      Qty:  8  Unit: $1.20  Amount: $9.60",  "confidence": 0.96},
        {"id": 6, "type": "paragraph", "text": "Sprite 330ml     Qty: 10  Unit: $1.15  Amount: $11.50", "confidence": 0.96},
        {"id": 7, "type": "paragraph", "text": "Fanta Orange 330ml Qty: 6 Unit: $1.30  Amount: $7.80",  "confidence": 0.95},
        {"id": 8, "type": "paragraph", "text": "Red Bull 250ml   Qty:  5  Unit: $2.80  Amount: $14.00", "confidence": 0.95},
        {"id": 9, "type": "paragraph", "text": "Subtotal: $57.90  |  Tax: $5.79  |  Total: $63.69",     "confidence": 0.97},
        {"id":10, "type": "paragraph", "text": "Payment Terms: Net 15 Days",                             "confidence": 0.98},
    ]
    tables = [{
        "id": 1,
        "title": "Items Purchased",
        "headers": ["Item", "Qty", "Unit Price", "Amount"],
        "rows": [
            ["Coca-Cola 330ml",    "12", "$1.25", "$15.00"],
            ["Pepsi 330ml",         "8", "$1.20",  "$9.60"],
            ["Sprite 330ml",       "10", "$1.15", "$11.50"],
            ["Fanta Orange 330ml",  "6", "$1.30",  "$7.80"],
            ["Red Bull 250ml",      "5", "$2.80", "$14.00"],
        ],
    }]
    full_text = "\n".join(b["text"] for b in text_blocks)
    return full_text, text_blocks, tables


# ── Real OCR stub ─────────────────────────────────────────────────────────────

def _real_process(file_bytes: bytes, filename: str) -> dict:
    raise NotImplementedError(f"Real OCR engine '{OCR_ENGINE}' not wired yet.")

