from __future__ import annotations

import os, json
from typing import List, Dict, Any

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm_refine_stopwords(
    *,
    candidate_terms: List[Dict[str, Any]],
    boilerplate_examples: List[str],
    corpus_hint: str = "",
) -> Dict[str, Any]:
    """
    Input: candidates like [{"term": "...", "df_ratio": 0.93, "score": 1.23}, ...]
    Output: {"stopwords": [...], "maybe_stopwords": [...], "keep_terms": [...], "rationale": "..."}
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    schema = {
        "type": "object",
        "properties": {
            "stopwords": {"type": "array", "items": {"type": "string"}},
            "maybe_stopwords": {"type": "array", "items": {"type": "string"}},
            "keep_terms": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": ["stopwords", "maybe_stopwords", "keep_terms", "rationale"],
        "additionalProperties": False,
    }

    payload = {
        "corpus_hint": corpus_hint,
        "boilerplate_examples": boilerplate_examples[:12],
        "candidate_terms": [
            {
                "term": (t.get("term") or "").strip(),
                "df_ratio": round(float(t.get("df_ratio") or 0.0), 3),
                "score": round(float(t.get("score") or 0.0), 6),
                "kind": t.get("kind", "general"),
            }
            for t in candidate_terms[:120]
            if (t.get("term") or "").strip()
        ],
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are assisting a DLP keyword policy generator.\n"
                "Goal: identify generic/boilerplate terms to EXCLUDE (stopwords).\n"
                "Rules:\n"
                "- Use df_ratio: very high df_ratio => likely boilerplate.\n"
                "- Prefer precision: if unsure, put in maybe_stopwords.\n"
                "- Arabic/English both possible.\n"
                "- Do NOT invent new terms not in candidate_terms.\n"
                "- Output must match the JSON schema."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "stopword_refine", "schema": schema},
        },
    )

    return json.loads(resp.choices[0].message.content)
