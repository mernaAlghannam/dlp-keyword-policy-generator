from __future__ import annotations
from typing import List, Dict

def suggest_regexes_from_phrases(phrases: List[str]) -> List[Dict[str, str]]:
    """
    Heuristic regex templates for common enterprise identifiers.
    Add more patterns if you want (still generic).
    """
    patterns = [
        {"name": "Invoice number", "regex": r"\bINV-\d{4}-\d{4,6}\b"},
        {"name": "PO number", "regex": r"\bPO-\d{5,8}\b"},
        {"name": "Work Permit", "regex": r"\bWP-[A-Z0-9]{2,10}-\d{4}-\d{4,6}\b"},
        {"name": "MSA agreement id", "regex": r"\bMSA-[A-Z0-9]{2,10}-\d{4}-\d{2}\b"},
        {"name": "Cost center", "regex": r"\bCC-\d{3,6}\b"},
        {"name": "Project code", "regex": r"\bPRJ-[A-Z0-9]{2,20}-[A-Z0-9]{2,20}-\d{2}\b"},
        {"name": "Employee ID", "regex": r"\bEMP-\d{5,8}\b"},
        {"name": "Vendor ID", "regex": r"\bVND-[A-Z0-9]{2,20}-\d{3,6}\b"},
        {"name": "IBAN (SA)", "regex": r"\bSA\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b"},
    ]

    # Keep only patterns that are "hinted" by phrases to stay relevant-ish
    hint = " ".join(phrases).lower()
    keep = []
    for p in patterns:
        key = p["name"].lower()
        if any(w in hint for w in ["invoice", "inv"]) and "invoice" in key:
            keep.append(p)
        elif any(w in hint for w in ["po", "purchase"]) and "po" in key:
            keep.append(p)
        elif "msa" in hint and "msa" in key:
            keep.append(p)
        elif "cc-" in hint and "cost center" in key:
            keep.append(p)
        elif "prj" in hint and "project" in key:
            keep.append(p)
        elif "emp" in hint and "employee" in key:
            keep.append(p)
        elif "vnd" in hint and "vendor" in key:
            keep.append(p)
        elif "iban" in hint and "iban" in key:
            keep.append(p)

    # If nothing matched, return a small generic set
    return keep or patterns[:4]
