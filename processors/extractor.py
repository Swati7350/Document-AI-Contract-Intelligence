"""
Structured field extraction processor.
────────────────────────────────────────────────────────────────
Mock mode : regex + keyword parsing over the ACTUAL OCR text.
            Never returns hardcoded Acme/TechSolutions data.
Real mode : OpenAI-compatible LLM (set ENABLE_REAL_LLM=true).
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import os, re, time, json, random
from typing import Any

ENABLE_REAL_LLM = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"

DEFAULT_SYSTEM_PROMPT = (
    "You are a document analysis AI. Extract structured fields from the "
    "document text. Return a valid JSON object with only the fields present. "
    "Be precise and concise."
)

DEFAULT_EXTRACTION_PROMPT = (
    "Extract all key fields from the following document text.\n"
    "Return a JSON object with any fields you can find, such as:\n"
    "invoice_number, invoice_date, vendor, bill_to, ship_to, items (list),\n"
    "subtotal, tax, total, payment_terms, contract_number, parties,\n"
    "effective_date, expiry_date, contract_value, governing_law, signatories.\n"
    "Only include fields that are present in the text.\n\n"
    "Document text:\n{text}\n\nReturn ONLY valid JSON."
)


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def extract(text: str,
            system_prompt: str = None,
            extraction_prompt: str = None) -> dict:
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    ext_p = (extraction_prompt or DEFAULT_EXTRACTION_PROMPT).replace(
        "{text}", text[:4000]
    )
    if ENABLE_REAL_LLM:
        return _real_extract(text, sys_p, ext_p)
    return _parse_extract(text, sys_p, ext_p)


# ══════════════════════════════════════════════════════════════════
#  MOCK PARSER — derives every field from the actual OCR text
# ══════════════════════════════════════════════════════════════════

def _parse_extract(text: str, system_prompt: str, extraction_prompt: str) -> dict:
    """
    Regex + keyword extraction over the real document text.
    Returns only fields actually found — no invented values.
    """
    time.sleep(0.4)
    t = text.strip()
    fields: dict[str, Any] = {}

    # ── Invoice / document number ─────────────────────────────────
    inv = _find(r"(?:invoice\s*(?:no|number|#)[:\s#]*)([\w\-]+)", t)
    if inv:
        fields["invoice_number"] = inv

    # ── Date ─────────────────────────────────────────────────────
    date = _find(
        r"(?:invoice\s*date|date)[:\s]+(\d{1,2}\s+\w+\s+\d{4}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        t,
    )
    if date:
        fields["invoice_date"] = date

    # ── Vendor / company ─────────────────────────────────────────
    vendor = _find(
        r"^([A-Z][A-Za-z\s&]+(?:Ltd|Inc|Corp|Co|Pvt|LLC|LLP|Supplies|Distribution|Services|Solutions)[^\n]*)",
        t, flags=re.MULTILINE,
    )
    if vendor:
        fields["vendor"] = vendor.strip()

    # ── Bill To / parties ────────────────────────────────────────
    bill_to = _find(r"bill\s*to[:\s]+([^\n]+)", t)
    if bill_to:
        fields["bill_to"] = bill_to.strip()

    ship_to = _find(r"ship\s*to[:\s]+([^\n]+)", t)
    if ship_to:
        fields["ship_to"] = ship_to.strip()

    # ── Contract parties ─────────────────────────────────────────
    between = _find(r"between\s+([^\n]+)\s+and\s+([^\n]+)", t)
    if between:
        fields["parties"] = between.strip()

    # ── Financial totals ─────────────────────────────────────────
    subtotal = _find(r"subtotal[:\s]+(\$?₹?€?[\d,]+\.?\d*)", t)
    if subtotal:
        fields["subtotal"] = subtotal

    tax = _find(r"\btax\b[:\s]+(\$?₹?€?[\d,]+\.?\d*)", t)
    if tax:
        fields["tax"] = tax

    total = _find(r"(?:grand\s+)?total[:\s]+(\$?₹?€?[\d,]+\.?\d*)", t)
    if total:
        fields["total"] = total

    contract_value = _find(r"(?:contract\s+value|total\s+value)[:\s]+(\$?₹?€?[\d,]+\.?\d*)", t)
    if contract_value:
        fields["contract_value"] = contract_value

    # ── Payment terms ─────────────────────────────────────────────
    pterm = _find(r"payment\s*terms[:\s]+([^\n]+)", t)
    if pterm:
        fields["payment_terms"] = pterm.strip()

    # ── Items table ───────────────────────────────────────────────
    items = _extract_items(t)
    if items:
        fields["items"] = items

    # ── Contract-specific fields ──────────────────────────────────
    contract_no = _find(r"contract\s*(?:no|number|#)[:\s]+([^\n]+)", t)
    if contract_no:
        fields["contract_number"] = contract_no.strip()

    eff_date = _find(r"effective\s*date[:\s]+([^\n]+)", t)
    if eff_date:
        fields["effective_date"] = eff_date.strip()

    expiry = _find(r"(?:expiry|expiration|end)\s*date[:\s]+([^\n]+)", t)
    if expiry:
        fields["expiry_date"] = expiry.strip()

    gov_law = _find(r"governing\s*law[:\s]+([^\n]+)", t)
    if gov_law:
        fields["governing_law"] = gov_law.strip()

    gstin = _find(r"(?:gstin|gst)[:\s]+([A-Z0-9]{15})", t)
    if gstin:
        fields["gstin"] = gstin

    # ── PO Number ─────────────────────────────────────────────────
    po = _find(r"p\.?o\.?\s*(?:no|number|#)[:\s]+([^\n]+)", t)
    if po:
        fields["po_number"] = po.strip()

    # ── If nothing was found, surface the first meaningful lines ──
    if not fields:
        lines = [l.strip() for l in t.splitlines() if len(l.strip()) > 8]
        fields["extracted_text"] = "\n".join(lines[:6])

    return {
        "engine":            "mock-parser",
        "fields":            fields,
        "system_prompt":     system_prompt,
        "extraction_prompt": extraction_prompt,
        "tokens_used":       0,
        "processing_time_ms": random.randint(200, 500),
    }


# ── Helpers ───────────────────────────────────────────────────────

def _find(pattern: str, text: str, flags: int = re.IGNORECASE) -> str:
    m = re.search(pattern, text, flags)
    if not m:
        return ""
    return m.group(1).strip() if m.lastindex else m.group(0).strip()


def _extract_items(text: str) -> list:
    """Extract line-item rows: lines with a product name + quantity + price."""
    item_re = re.compile(
        r"^(.{3,40}?)\s{2,}(\d+)\s{2,}(\$?₹?[\d,]+\.?\d*)\s{2,}(\$?₹?[\d,]+\.?\d*)",
        re.MULTILINE,
    )
    items = []
    for m in item_re.finditer(text):
        items.append({
            "description": m.group(1).strip(),
            "quantity":    m.group(2).strip(),
            "unit_price":  m.group(3).strip(),
            "amount":      m.group(4).strip(),
        })
    return items


# ══════════════════════════════════════════════════════════════════
#  REAL LLM PATH
# ══════════════════════════════════════════════════════════════════

def _real_extract(text: str, system_prompt: str, extraction_prompt: str) -> dict:
    try:
        import openai
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": extraction_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        fields = json.loads(resp.choices[0].message.content)
        return {
            "engine":            "openai",
            "fields":            fields,
            "system_prompt":     system_prompt,
            "extraction_prompt": extraction_prompt,
            "tokens_used":       resp.usage.total_tokens,
            "processing_time_ms": 0,
        }
    except Exception as e:
        return {"error": str(e), "fields": {}, "engine": "openai-error"}
