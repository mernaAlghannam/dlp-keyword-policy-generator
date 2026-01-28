# backend/app/topics.py
from __future__ import annotations

import re
from typing import List, Dict, Set, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


_RX_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RX_MOSTLY_NUM = re.compile(r"^[\d\W_]+$")
_RX_ID_LIKE = re.compile(r".*\d{3,}.*")  # any >=3 consecutive digits somewhere
_RX_REPEAT_TOKEN = re.compile(r"^(\w+)\s+\1(\s+\1)*$")


def is_low_generalization_phrase(p: str) -> bool:
    """
    Filters terms that usually don't generalize:
    - exact dates
    - mostly numeric tokens
    - ID-like strings with long digit runs
    - repeated token sequences ("ppm ppm")
    """
    if not p:
        return True
    s = p.strip()
    if not s:
        return True
    sl = s.lower()

    if _RX_DATE.match(sl):
        return True
    if _RX_MOSTLY_NUM.match(sl.replace(" ", "")):
        return True
    if _RX_REPEAT_TOKEN.match(sl):
        return True

    # avoid long ID-like tokens (invoice numbers, work permit IDs, etc.)
    # (you asked: remove regexes; we also should avoid anchoring on IDs)
    if _RX_ID_LIKE.match(sl) and (" " not in sl):
        return True
    if len(sl) >= 18 and any(ch.isdigit() for ch in sl) and "-" in sl:
        return True

    return False


def infer_topics(
    docs: List[str],
    stopwords: Set[str],
    max_topics: int = 6,
    ngram_max: int = 3,
    df_ratio_min: float = 0.20,
    df_ratio_max: float = 0.85,
    per_topic_terms: int = 24,
) -> Dict[str, Any]:
    """
    Topic inference via TF-IDF + KMeans.

    Returns:
      topic_phrases: {topic_id: [phrases...]}
      topic_doc_counts: {topic_id: count}
      df_ratio_by_term: {term: df_ratio}
    """
    if not docs:
        return {"topic_phrases": {}, "topic_doc_counts": {}, "df_ratio_by_term": {}}

    sw_list = sorted([s.lower() for s in stopwords if s and len(s) >= 2])

    vec = TfidfVectorizer(
        ngram_range=(1, ngram_max),
        lowercase=True,
        stop_words=sw_list if sw_list else None,
        token_pattern=r"(?u)\b[\w\-\/]{2,}\b",
        min_df=1,
    )
    X = vec.fit_transform(docs)  # (n_docs, n_terms)
    terms = vec.get_feature_names_out()
    n_docs = X.shape[0]

    # Document frequency ratio for each term
    df = (X > 0).sum(axis=0).A1
    df_ratio = df / float(n_docs)
    df_ratio_by_term = {t: float(r) for t, r in zip(terms, df_ratio)}

    # Choose k
    if n_docs <= 2:
        k = 1
    else:
        k = min(max_topics, max(2, int(np.sqrt(n_docs)) + 1))

    if k == 1:
        # single topic: just take global TF-IDF sum top terms with df filtering
        scores = X.sum(axis=0).A1
        order = np.argsort(scores)[::-1]
        phrases = []
        for idx in order:
            t = terms[idx]
            r = df_ratio_by_term.get(t, 0.0)
            if r < df_ratio_min or r > df_ratio_max:
                continue
            if is_low_generalization_phrase(t):
                continue
            phrases.append(t)
            if len(phrases) >= per_topic_terms:
                break
        return {
            "topic_phrases": {0: phrases},
            "topic_doc_counts": {0: n_docs},
            "df_ratio_by_term": df_ratio_by_term,
        }

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    topic_phrases: Dict[int, List[str]] = {}
    topic_doc_counts: Dict[int, int] = {}

    # per-topic centroid importance
    centroids = km.cluster_centers_  # shape (k, n_terms)

    for tid in range(k):
        topic_doc_counts[tid] = int((labels == tid).sum())

        weights = centroids[tid]
        order = np.argsort(weights)[::-1]

        phrases = []
        for idx in order:
            t = terms[idx]
            r = df_ratio_by_term.get(t, 0.0)

            # keep mid-frequency terms for generalization
            if r < df_ratio_min or r > df_ratio_max:
                continue
            if is_low_generalization_phrase(t):
                continue
            phrases.append(t)
            if len(phrases) >= per_topic_terms:
                break

        topic_phrases[tid] = phrases

    return {
        "topic_phrases": topic_phrases,
        "topic_doc_counts": topic_doc_counts,
        "df_ratio_by_term": df_ratio_by_term,
    }
