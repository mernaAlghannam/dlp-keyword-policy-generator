from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import re

def summarize_corpus(docs: List[str], max_chars: int = 2500) -> str:
    # Simple non-LLM summary: most frequent non-trivial tokens + a few snippets
    joined = "\n".join(docs)[:200_000]
    joined = re.sub(r"\s+", " ", joined)
    return joined[:max_chars]

def heuristic_exclude(top_terms: List[Dict[str, Any]]) -> List[str]:
    # Exclude terms that are overly broad or suspiciously generic by df_ratio
    out = []
    for t in top_terms:
        term = t["term"]
        if t.get("df_ratio", 0) > 0.80:  # appears in almost all docs
            out.append(term)
        if len(term) <= 3:
            out.append(term)
    return sorted(set(out))

def build_symantec_like_boolean(term_groups: List[Dict[str, Any]], *, near_k: int = 10, min_groups: int = 2) -> str:
    """
    Create something Symantec-friendly:
      ( (G1) NEAR/10 (G2) ) OR ( (G1) NEAR/10 (G3) ) ...
    Where each Gi is (t1 OR t2 OR ...)
    """
    def group_expr(g):
        terms = [escape_term(x) for x in g["terms"] if x.strip()]
        if not terms:
            return "( )"
        op = " OR " if (g.get("match") == "OR") else " AND "
        return "(" + op.join(f"\"{x}\"" for x in terms) + ")"

    groups = [g for g in term_groups if g.get("terms")]
    if len(groups) < 2:
        # fallback to OR of everything
        all_terms = []
        for g in groups:
            all_terms += g["terms"]
        all_terms = [escape_term(x) for x in all_terms]
        return "(" + " OR ".join(f"\"{x}\"" for x in all_terms[:25]) + ")"

    # pick top N groups
    groups = groups[:max(2, min_groups + 1)]
    clauses = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            clauses.append(f"({group_expr(groups[i])} NEAR/{near_k} {group_expr(groups[j])})")
    return " OR ".join(clauses[:12])

def escape_term(s: str) -> str:
    return s.replace('"', '').strip()

def build_output_payload(
    *,
    top_terms: List[Dict[str, Any]],
    include_terms: List[str],
    exclude_terms: List[str],
    term_groups: List[Dict[str, Any]],
    near_k: int,
    threshold: int
) -> Dict[str, Any]:
    boolean_expr = build_symantec_like_boolean(term_groups, near_k=near_k)

    # “Threshold” for Symantec keyword rules is usually configured in UI;
    # we output it as metadata + guidance text.
    return {
        "keyword_list_include": sorted(set(include_terms)),
        "keyword_list_exclude": sorted(set(exclude_terms)),
        "term_groups": term_groups,
        "policy": {
            "type": "symantec_keyword_policy_like",
            "boolean_expression": boolean_expr,
            "proximity": {"mode": "NEAR", "distance": near_k},
            "threshold": threshold,
            "notes": [
                "In Symantec DLP, create a Keyword Rule and paste the boolean expression in the rule logic (or build equivalent groups in UI).",
                "Add Include terms as a Keyword List. Add Exclude terms as exceptions where possible.",
                "Tune: reduce max_df/min_df and increase threshold to reduce false positives."
            ]
        },
        "debug": {
            "top_terms": top_terms[:80]
        }
    }
