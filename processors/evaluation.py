"""
Evaluation module with a built-in test dataset.
Returns field-level accuracy, precision, recall, F1.
"""
from __future__ import annotations
from typing import List, Dict

# ── Static test dataset ────────────────────────────────────────────────────────
TEST_DATASET = [
    {
        "doc_id": "doc_001",
        "ground_truth": {
            "parties":          "Acme Corporation, TechSolutions Ltd",
            "contract_number":  "SVC-2024-0042",
            "effective_date":   "January 1, 2024",
            "expiry_date":      "December 31, 2025",
            "contract_value":   "USD 150,000",
            "payment_terms":    "Monthly USD 12,500",
            "governing_law":    "California, USA",
        },
        "predicted": {
            "parties":          "Acme Corporation, TechSolutions Ltd",
            "contract_number":  "SVC-2024-0042",
            "effective_date":   "January 1, 2024",
            "expiry_date":      "December 31, 2025",
            "contract_value":   "USD 150,000",
            "payment_terms":    "Monthly USD 12,500",
            "governing_law":    "California",          # partial match
        },
    },
    {
        "doc_id": "doc_002",
        "ground_truth": {
            "parties":          "GlobalTech Inc, DataFlow Systems",
            "contract_number":  "MSA-2023-0118",
            "effective_date":   "March 15, 2023",
            "expiry_date":      "March 14, 2025",
            "contract_value":   "USD 240,000",
            "payment_terms":    "Quarterly USD 30,000",
            "governing_law":    "New York, USA",
        },
        "predicted": {
            "parties":          "GlobalTech Inc, DataFlow Systems",
            "contract_number":  "MSA-2023-0118",
            "effective_date":   "March 15, 2023",
            "expiry_date":      "March 14, 2025",
            "contract_value":   "USD 240,000",
            "payment_terms":    "Quarterly USD 30,000",
            "governing_law":    "New York, USA",
        },
    },
    {
        "doc_id": "doc_003",
        "ground_truth": {
            "parties":          "NexaCorp, CloudBase Ltd",
            "contract_number":  "NDA-2024-0009",
            "effective_date":   "February 1, 2024",
            "expiry_date":      "January 31, 2026",
            "contract_value":   "N/A",
            "payment_terms":    "N/A",
            "governing_law":    "Delaware, USA",
        },
        "predicted": {
            "parties":          "NexaCorp, CloudBase Ltd",
            "contract_number":  "NDA-2024-0009",
            "effective_date":   "February 2024",       # minor error
            "expiry_date":      "January 31, 2026",
            "contract_value":   "Not specified",       # mismatch
            "payment_terms":    "N/A",
            "governing_law":    "Delaware, USA",
        },
    },
    {
        "doc_id": "doc_004",
        "ground_truth": {
            "parties":          "Meridian Healthcare, MedSupply Co",
            "contract_number":  "PO-2024-5521",
            "effective_date":   "April 1, 2024",
            "expiry_date":      "March 31, 2025",
            "contract_value":   "USD 88,000",
            "payment_terms":    "Net 45 days",
            "governing_law":    "Texas, USA",
        },
        "predicted": {
            "parties":          "Meridian Healthcare, MedSupply Co",
            "contract_number":  "PO-2024-5521",
            "effective_date":   "April 1, 2024",
            "expiry_date":      "March 31, 2025",
            "contract_value":   "USD 88,000",
            "payment_terms":    "Net 45 days",
            "governing_law":    "Texas, USA",
        },
    },
    {
        "doc_id": "doc_005",
        "ground_truth": {
            "parties":          "StartupXYZ, Venture Capital Partners",
            "contract_number":  "INV-2024-0301",
            "effective_date":   "June 1, 2024",
            "expiry_date":      "May 31, 2025",
            "contract_value":   "USD 500,000",
            "payment_terms":    "Lump sum on signing",
            "governing_law":    "UK (England & Wales)",
        },
        "predicted": {
            "parties":          "StartupXYZ, Venture Capital Partners",
            "contract_number":  "INV-2024-0301",
            "effective_date":   "June 1, 2024",
            "expiry_date":      "May 2025",             # minor error
            "contract_value":   "USD 500,000",
            "payment_terms":    "Lump sum",             # partial
            "governing_law":    "England and Wales",    # partial
        },
    },
]

FIELDS = ["parties", "contract_number", "effective_date", "expiry_date",
          "contract_value", "payment_terms", "governing_law"]


def _match(gt: str, pred: str) -> bool:
    """Fuzzy match — exact or one contains the other."""
    gt_n   = gt.lower().strip()
    pred_n = pred.lower().strip()
    return gt_n == pred_n or gt_n in pred_n or pred_n in gt_n


def compute_metrics() -> dict:
    field_stats: Dict[str, dict] = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}

    for sample in TEST_DATASET:
        gt   = sample["ground_truth"]
        pred = sample["predicted"]
        for field in FIELDS:
            gt_val   = gt.get(field,   "")
            pred_val = pred.get(field, "")
            if gt_val and pred_val:
                if _match(gt_val, pred_val):
                    field_stats[field]["tp"] += 1
                else:
                    field_stats[field]["fp"] += 1
                    field_stats[field]["fn"] += 1
            elif gt_val:
                field_stats[field]["fn"] += 1
            elif pred_val:
                field_stats[field]["fp"] += 1

    rows = []
    for field in FIELDS:
        s  = field_stats[field]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy  = tp / len(TEST_DATASET)
        rows.append({
            "field":     field.replace("_", " ").title(),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall":    round(recall,    3),
            "f1":        round(f1,        3),
            "accuracy":  round(accuracy,  3),
        })

    all_tp = sum(r["tp"] for r in rows)
    all_fp = sum(r["fp"] for r in rows)
    all_fn = sum(r["fn"] for r in rows)
    macro_p  = sum(r["precision"] for r in rows) / len(rows)
    macro_r  = sum(r["recall"]    for r in rows) / len(rows)
    macro_f1 = sum(r["f1"]        for r in rows) / len(rows)
    overall_acc = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) else 0

    return {
        "total_documents": len(TEST_DATASET),
        "total_fields_evaluated": len(FIELDS),
        "overall_accuracy": round(overall_acc, 3),
        "macro_precision":  round(macro_p,  3),
        "macro_recall":     round(macro_r,  3),
        "macro_f1":         round(macro_f1, 3),
        "field_metrics":    rows,
        "dataset":          TEST_DATASET,
    }
