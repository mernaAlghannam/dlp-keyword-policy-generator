# backend/app/term_mining.py
from __future__ import annotations

import re
from typing import List, Dict, Set

from sklearn.feature_extraction.text import CountVectorizer


def load_stopwords(path: str) -> set[str]:
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.add(s.lower())
    return out


_RX_WS = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    s = s.replace("\u200f", " ").replace("\u200e", " ")
    s = _RX_WS.sub(" ", s)
    return s.strip()


def mine_terms(
    docs: List[str],
    stopwords: Set[str],
    top_k: int = 200,
    ngram_max: int = 3,
    min_term_len: int = 3,
) -> List[Dict]:
    """
    Returns list of terms with:
      - tf: total count across all docs
      - df: number of docs containing term
      - df_ratio: df / n_docs

    NOTE: stopwords are applied internally; output is for analysis UI & scoring.
    """
    if not docs:
        return []

    docs_n = [_normalize_text(d) for d in docs]

    # We do NOT rely on sklearn's stop_words for Arabic, but we can still pass it as a list.
    sw_list = sorted([s.lower() for s in stopwords if s and len(s) >= 2])

    vec = CountVectorizer(
        ngram_range=(1, ngram_max),
        lowercase=True,
        stop_words=sw_list if sw_list else None,
        token_pattern=r"(?u)\b[\w\-\/]{2,}\b",
        min_df=1,
    )
    X = vec.fit_transform(docs_n)  # (n_docs, n_terms)
    terms = vec.get_feature_names_out()
    n_docs = X.shape[0]

    # TF: sum counts across docs
    tf = X.sum(axis=0).A1

    # DF: count docs where term appears (binary)
    df = (X > 0).sum(axis=0).A1

    out = []
    for t, tf_i, df_i in zip(terms, tf, df):
        t2 = t.strip()
        if len(t2) < min_term_len:
            continue
        out.append(
            {
                "term": t2,
                "tf": int(tf_i),
                "df": int(df_i),
                "df_ratio": float(df_i) / float(n_docs),
            }
        )

    out.sort(key=lambda x: (x["df_ratio"], x["tf"]), reverse=True)
    return out[:top_k]
