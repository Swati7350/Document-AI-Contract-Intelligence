"""
OCR & Layout Parsing processor.
Mock mode returns realistic demo data.
Swap OCR_ENGINE=docling or tesseract in .env for real processing.
"""
from __future__ import annotations
import os, time, random

OCR_ENGINE = os.getenv("OCR_ENGINE", "mock")


def process_document(file_bytes: bytes, filename: str) -> dict:
    """
    Entry point. Returns a standardised result dict.
    Replace the mock block with real engine calls when ready.
    """
    if OCR_ENGINE != "mock":
        return _real_process(file_bytes, filename)
    return _mock_process(filename)


# ── Mock ──────────────────────────────────────────────────────────────────────

def _mock_process(filename: str) -> dict:
    time.sleep(1.2)   # simulate processing time
    is_pdf = filename.lower().endswith(".pdf")

    text_blocks = [
        {"id": 1, "type": "heading",    "text": "SERVICE AGREEMENT",                       "confidence": 0.99},
        {"id": 2, "type": "paragraph",  "text": "This Service Agreement ('Agreement') is entered into as of January 1, 2024, by and between Acme Corporation ('Client') and TechSolutions Ltd ('Supplier').", "confidence": 0.97},
        {"id": 3, "type": "paragraph",  "text": "1. SERVICES. Supplier agrees to provide software development and maintenance services as described in Exhibit A attached hereto.", "confidence": 0.96},
        {"id": 4, "type": "paragraph",  "text": "2. PAYMENT. Client shall pay Supplier a monthly fee of USD 12,500 payable within 30 days of invoice.", "confidence": 0.95},
        {"id": 5, "type": "paragraph",  "text": "3. TERM. This Agreement shall commence on January 1, 2024 and continue through December 31, 2025 unless earlier terminated.", "confidence": 0.94},
        {"id": 6, "type": "paragraph",  "text": "4. TERMINATION. Either party may terminate this Agreement upon 30 days written notice. Client may terminate immediately for cause.", "confidence": 0.96},
        {"id": 7, "type": "paragraph",  "text": "5. CONFIDENTIALITY. Each party agrees to keep confidential all proprietary information disclosed by the other party.", "confidence": 0.97},
        {"id": 8, "type": "signature",  "text": "Signed: John Smith, CEO — Acme Corporation",                         "confidence": 0.91},
        {"id": 9, "type": "signature",  "text": "Signed: Sarah Johnson, Director — TechSolutions Ltd",                "confidence": 0.90},
    ]

    tables = [
        {
            "id": 1,
            "title": "Payment Schedule",
            "headers": ["Milestone", "Due Date", "Amount (USD)"],
            "rows": [
                ["Project Kickoff",       "Jan 15, 2024",  "12,500"],
                ["Phase 1 Delivery",      "Mar 31, 2024",  "12,500"],
                ["Phase 2 Delivery",      "Jun 30, 2024",  "12,500"],
                ["Final Delivery",        "Sep 30, 2024",  "12,500"],
            ],
        }
    ]

    layout = {
        "pages": 2 if is_pdf else 1,
        "columns_detected": 1,
        "stamps_seals": ["CONFIDENTIAL stamp detected (p.1)", "Company seal detected (p.2)"],
        "headers_footers": True,
        "tables_count": 1,
        "signatures_count": 2,
        "bounding_boxes": [
            {"id": 1, "label": "Heading",    "x": 0.10, "y": 0.05, "w": 0.80, "h": 0.06, "color": "#3b82f6"},
            {"id": 2, "label": "Paragraph",  "x": 0.08, "y": 0.14, "w": 0.84, "h": 0.10, "color": "#10b981"},
            {"id": 3, "label": "Table",      "x": 0.08, "y": 0.40, "w": 0.84, "h": 0.22, "color": "#f59e0b"},
            {"id": 4, "label": "Signature",  "x": 0.08, "y": 0.82, "w": 0.38, "h": 0.07, "color": "#8b5cf6"},
            {"id": 5, "label": "Seal/Stamp", "x": 0.70, "y": 0.80, "w": 0.18, "h": 0.10, "color": "#ef4444"},
        ],
    }

    return {
        "engine": "mock",
        "filename": filename,
        "text_blocks": text_blocks,
        "full_text": "\n\n".join(b["text"] for b in text_blocks),
        "tables": tables,
        "layout": layout,
        "avg_confidence": round(sum(b["confidence"] for b in text_blocks) / len(text_blocks), 3),
        "processing_time_ms": random.randint(900, 1400),
    }


# ── Real (stub) ───────────────────────────────────────────────────────────────

def _real_process(file_bytes: bytes, filename: str) -> dict:
    """Placeholder for real OCR engines (Docling, Tesseract, etc.)"""
    raise NotImplementedError(f"Real OCR engine '{OCR_ENGINE}' not wired yet.")
