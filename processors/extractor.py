"""
Structured field extraction processor.
Mock mode returns realistic contract fields.
Set ENABLE_REAL_LLM=true and provide OPENAI_API_KEY for real extraction.
"""
from __future__ import annotations
import os, time, json, random

ENABLE_REAL_LLM = os.getenv("ENABLE_REAL_LLM", "false").lower() == "true"

DEFAULT_SYSTEM_PROMPT = """You are a contract analysis AI. Extract structured fields from the contract text provided.
Return a valid JSON object with the fields specified. Be precise and concise."""

DEFAULT_EXTRACTION_PROMPT = """Extract the following fields from this contract:
- parties: list of party names and roles
- contract_number: unique contract identifier
- effective_date: when the contract starts
- expiry_date: when the contract ends
- contract_value: total monetary value
- payment_terms: payment schedule details
- termination_clause: notice period and conditions
- governing_law: jurisdiction
- signatories: list of signatories with names and titles

Contract text:
{text}

Return ONLY valid JSON."""


def extract(text: str, system_prompt: str = None, extraction_prompt: str = None) -> dict:
    """
    Main extraction entry point.
    Returns structured fields + the prompts used (for Prompt Studio display).
    """
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    ext_p = (extraction_prompt or DEFAULT_EXTRACTION_PROMPT).replace("{text}", text[:3000])

    if ENABLE_REAL_LLM:
        return _real_extract(text, sys_p, ext_p)
    return _mock_extract(sys_p, ext_p)


# ── Mock ──────────────────────────────────────────────────────────────────────

def _mock_extract(system_prompt: str, extraction_prompt: str) -> dict:
    time.sleep(0.8)
    fields = {
        "parties": [
            {"name": "Acme Corporation",   "role": "Client"},
            {"name": "TechSolutions Ltd",  "role": "Supplier"},
        ],
        "contract_number":    "SVC-2024-0042",
        "effective_date":     "January 1, 2024",
        "expiry_date":        "December 31, 2025",
        "contract_value":     "USD 150,000",
        "payment_terms":      "Monthly installments of USD 12,500, due within 30 days of invoice",
        "termination_clause": "30 days written notice by either party; immediate termination for cause by Client",
        "governing_law":      "State of California, USA",
        "signatories": [
            {"name": "John Smith",    "title": "CEO",      "company": "Acme Corporation"},
            {"name": "Sarah Johnson", "title": "Director", "company": "TechSolutions Ltd"},
        ],
    }
    return {
        "engine": "mock",
        "fields": fields,
        "system_prompt": system_prompt,
        "extraction_prompt": extraction_prompt,
        "tokens_used": random.randint(420, 680),
        "processing_time_ms": random.randint(700, 1100),
    }


# ── Real (stub) ───────────────────────────────────────────────────────────────

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
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": extraction_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        fields = json.loads(resp.choices[0].message.content)
        return {
            "engine": "openai",
            "fields": fields,
            "system_prompt": system_prompt,
            "extraction_prompt": extraction_prompt,
            "tokens_used": resp.usage.total_tokens,
            "processing_time_ms": 0,
        }
    except Exception as e:
        return {"error": str(e), "fields": {}}
