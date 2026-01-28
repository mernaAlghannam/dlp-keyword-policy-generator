from __future__ import annotations
from typing import List, Dict, Any, Set
import re

TOKEN_RX = re.compile(r"[A-Za-z0-9_./-]+|[\u0600-\u06FF0-9_./-]+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RX.finditer(text)]

def auto_stopwords_from_stats(term_stats: List[Dict[str, Any]]) -> Set[str]:
    """
    term_stats items should include: term, df_ratio
    """
    sw = set()
    for t in term_stats:
        term = (t.get("term") or "").strip().lower()
        df = float(t.get("df_ratio") or 0.0)
        if not term:
            continue
        # overly common across docs -> likely generic/template
        if df >= 0.85:
            sw.add(term)
        if len(term) <= 2:
            sw.add(term)

    # Hard safety boilerplate (light, not a manual list – just universal template noise)
    universal = {
        "confidential", "internal", "use", "only", "notice", "regards", "thanks",
        "page", "document", "version", "generated", "template",
        "سري", "سرية", "داخلي", "للاستخدام", "فقط", "تنبيه", "صفحة"
    }
    sw |= universal
    return sw

def auto_stopwords_from_boilerplate(boiler_lines: List[str]) -> Set[str]:
    sw = set()
    for ln in boiler_lines:
        for tok in tokenize(ln):
            if len(tok) <= 2:
                continue
            sw.add(tok)
    return sw
