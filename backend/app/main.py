# backend/app/main.py
from __future__ import annotations

import io
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import PlainTextResponse

from app.llm_providers import get_llm_client, llm_text


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="DLP Keyword Policy Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions (fine for local prototype)
SESSIONS: Dict[str, Dict[str, Any]] = {}

import re
from typing import Optional

class TestPolicyResponse(BaseModel):
    n_files: int
    results: List[Dict[str, Any]]

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def parse_pretty_policy_rules(policy_text: str) -> List[Dict[str, str]]:
    """
    Accepts either:
      - **Rule Name**: `( ... )`
      - OR:
        ## Rule X: Name
        `( ... )`  or  ( ... )

    Returns [{name, expr}, ...]
    """
    txt = _strip_code_fences(policy_text)

    rules = []

    # Pattern A: headings "## Rule X: Name" followed by expression line
    # We'll capture blocks between rule headings.
    blocks = re.split(r"\n(?=##\s*Rule\s*\d+\s*:)", txt)
    for b in blocks:
        m = re.search(r"^##\s*Rule\s*\d+\s*:\s*(.+?)\s*$", b.strip(), re.MULTILINE)
        if not m:
            continue
        name = m.group(1).strip()
        # find first expression line: in backticks or parentheses
        expr_m = re.search(r"`([^`]+)`", b)
        if not expr_m:
            expr_m = re.search(r"\((.+)\)", b, re.DOTALL)
        if expr_m:
            expr = expr_m.group(1).strip()
            rules.append({"name": name, "expr": expr})

    # Pattern B: "**Name**: `( ... )`"
    if not rules:
        for m in re.finditer(r"\*\*(.+?)\*\*\s*:\s*`([^`]+)`", txt):
            rules.append({"name": m.group(1).strip(), "expr": m.group(2).strip()})

    return rules

_TOKEN_RE = re.compile(
    r'\"([^"]+)\"|\bAND\b|\bOR\b|\(|\)',
    flags=re.IGNORECASE
)

def tokenize_expr(expr: str):
    """Tokens: ("phrase"), AND, OR, (, )"""
    toks = []
    for m in _TOKEN_RE.finditer(expr):
        if m.group(1) is not None:
            toks.append(("PHRASE", m.group(1)))
        else:
            val = m.group(0).upper()
            if val in ("AND", "OR", "(", ")"):
                toks.append((val, val))
    return toks

def to_rpn(tokens):
    """Shunting-yard: AND > OR"""
    prec = {"AND": 2, "OR": 1}
    out = []
    stack = []
    for ttype, tval in tokens:
        if ttype == "PHRASE":
            out.append((ttype, tval))
        elif ttype in ("AND", "OR"):
            while stack and stack[-1] in ("AND", "OR") and prec[stack[-1]] >= prec[ttype]:
                out.append((stack.pop(), None))
            stack.append(ttype)
        elif ttype == "(":
            stack.append("(")
        elif ttype == ")":
            while stack and stack[-1] != "(":
                out.append((stack.pop(), None))
            if stack and stack[-1] == "(":
                stack.pop()
    while stack:
        op = stack.pop()
        if op != "(":
            out.append((op, None))
    return out

from typing import Dict, List, Set, Tuple

def eval_rpn(rpn, raw_text: str, hit_counts: Dict[str, int]) -> bool:
    """
    Evaluates RPN boolean expression and records ONLY phrases that contributed
    to the final True result (not every phrase that appears in the text).

    hit_counts keys are normalized phrase strings.
    """
    tokens = tokenize_for_match(raw_text)
    token_set = set(tokens)

    # Stack elements: (truth_value, contributing_phrase_keys_set)
    stack: List[Tuple[bool, Set[str]]] = []

    for ttype, tval in rpn:
        if ttype == "PHRASE":
            needle = (tval or "").strip()
            ok = phrase_in_tokens(needle, tokens, token_set)

            contrib: Set[str] = set()
            if ok:
                k = normalize_for_match(needle)
                contrib.add(k)

            stack.append((ok, contrib))

        elif ttype == "AND":
            (b, cb) = stack.pop() if stack else (False, set())
            (a, ca) = stack.pop() if stack else (False, set())
            ok = a and b
            # For AND, both sides must be true; contributing phrases are union
            stack.append((ok, (ca | cb) if ok else set()))

        elif ttype == "OR":
            (b, cb) = stack.pop() if stack else (False, set())
            (a, ca) = stack.pop() if stack else (False, set())
            ok = a or b
            # For OR, contributing phrases come from whichever side(s) are true
            if a and b:
                stack.append((True, ca | cb))
            elif a:
                stack.append((True, ca))
            elif b:
                stack.append((True, cb))
            else:
                stack.append((False, set()))

        else:
            # ignore unknown tokens safely
            continue

    if not stack:
        return False

    ok, contrib = stack[-1]
    if ok:
        for k in contrib:
            hit_counts[k] = hit_counts.get(k, 0) + 1

    return ok



import unicodedata

_ARABIC_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_TATWEEL = "\u0640"

def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    s = s.replace(_ARABIC_TATWEEL, "")
    #s = re.sub(r"[-_/]+", " ", s)
    s = _ARABIC_DIACRITICS.sub("", s)

    # normalize common Arabic alef variants
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    # unicode normalize
    s = unicodedata.normalize("NFKC", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

TOKEN_MATCH_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF]+", re.UNICODE)

def tokenize_for_match(text: str) -> List[str]:
    return TOKEN_MATCH_RE.findall(normalize_for_match(text))

def phrase_in_tokens(phrase: str, tokens: List[str], token_set: set) -> bool:
    p = normalize_for_match(phrase)
    if not p:
        return False

    p_tokens = tokenize_for_match(p)
    if not p_tokens:
        return False

    # single token: exact token presence (NO substring)
    if len(p_tokens) == 1:
        return p_tokens[0] in token_set

    # multi-token phrase: contiguous sequence match
    n = len(p_tokens)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i : i + n] == p_tokens:
            return True
    return False


def phrase_present(phrase: str, text_norm: str) -> bool:
    p = normalize_for_match(phrase)

    # If phrase is empty after normalization
    if not p:
        return False

    # single token => word boundary match
    if " " not in p:
        # \b doesn't work well for Arabic, so use custom boundaries:
        # "not alnum/arabic" on both sides
        return re.search(rf"(?<![A-Za-z0-9\u0600-\u06FF]){re.escape(p)}(?![A-Za-z0-9\u0600-\u06FF])", text_norm) is not None

    # multi-token phrase => require whole phrase with boundaries around ends
    return re.search(rf"(?<![A-Za-z0-9\u0600-\u06FF]){re.escape(p)}(?![A-Za-z0-9\u0600-\u06FF])", text_norm) is not None

def split_top_level(expr: str, op_word: str) -> List[str]:
    """
    Split expr by op_word at top level (depth 0), ignoring parentheses.
    op_word should be 'AND' or 'OR'.
    """
    s = expr.strip()
    out = []
    buf = []
    depth = 0
    i = 0
    op = op_word.upper()

    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue

        # try match " AND " / " OR " at depth 0
        if depth == 0:
            m = re.match(rf"\s*{op}\s*", s[i:], flags=re.IGNORECASE)
            if m:
                part = "".join(buf).strip()
                if part:
                    out.append(part)
                buf = []
                i += m.end()
                continue

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if last:
        out.append(last)
    return out

_PHRASE_QUOTED_RE = re.compile(r'"([^"]+)"')

def extract_phrases_from_group(group_expr: str) -> List[str]:
    # group like: ("a" OR "b" OR "c")
    return [m.group(1) for m in _PHRASE_QUOTED_RE.finditer(group_expr)]


def test_rule(expr: str, text: str):
    tokens = tokenize_expr(expr)
    rpn = to_rpn(tokens)
    hits = {}
    ok = eval_rpn(rpn, text, hits)   # NOTE: pass raw text, eval_rpn normalizes itself
    return ok, hits


@app.post("/api/test_policy", response_model=TestPolicyResponse)
async def test_policy(
    policy_text: str = Form(...),
    files: List[UploadFile] = File(...),
):
    rules = parse_pretty_policy_rules(policy_text)
    if not rules:
        raise HTTPException(
            status_code=400,
            detail='No rules found. Expected either: **Rule Name**: `( ... )` OR "## Rule X: Name" followed by `(... )`',
        )

    results = []
    for f in files:
        data = await f.read()
        text = extract_text_from_upload(f, data)
        text = normalize_ws(text)
        if not text:
            results.append({
                "filename": f.filename or "unknown",
                "readable": False,
                "matched_rules": [],
                "matched_any": False,
            })
            continue

        matched_rules = []
        for r in rules:
            ok, hits = test_rule(r["expr"], text)
            if ok:
                top_hits = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)[:12]
                matched_rules.append({
                    "rule_name": r["name"],
                    "hit_phrases": [h[0] for h in top_hits],
                })


        results.append({
            "filename": f.filename or "unknown",
            "readable": True,
            "matched_any": len(matched_rules) > 0,
            "matched_rules": matched_rules,
        })

    return TestPolicyResponse(
        n_files=len(results),
        results=results,
    )


from pydantic import BaseModel, Field
from typing import List

class PhraseRow(BaseModel):
    text: str
    df_ratio: float = Field(..., ge=0.0, le=1.0)

class TopicPreview(BaseModel):
    name: str
    doc_count: int
    phrases: List[PhraseRow]

class TermRow(BaseModel):
    term: str
    tf: int
    df: int
    df_ratio: float = Field(..., ge=0.0, le=1.0)

class AnalyzeResponse(BaseModel):
    session_id: str
    n_docs: int

    # keep existing markdown fields (optional, nice for debugging)
    topics_preview_md: str
    top_terms_md: str

    # ✅ NEW: structured fields that your UI wants
    topics_preview: List[TopicPreview] = []
    top_terms: List[TermRow] = []
    generic_terms: List[str] = []



class GeneratePrettyRequest(BaseModel):
    session_id: str
    policy_title: str = "DLP Keyword Policy (Proposed)"
    corpus_hint: str = ""
    max_rules: int = 5

    # phrase selection window to encourage generalization
    df_ratio_min: float = 0.10
    df_ratio_max: float = 0.85

    llm_required: bool = True

    # Optional: still allow suggestions separately (not inside rules)
    include_regex_suggestions: bool = True
    max_regex_suggestions: int = 8

    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.2"))


# -----------------------------
# Text extraction
# -----------------------------
def _read_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        # fallback: no PDF support
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for p in reader.pages:
            parts.append(p.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


def _read_docx_bytes(data: bytes) -> str:
    try:
        import docx  # type: ignore
    except Exception:
        return ""
    try:
        f = io.BytesIO(data)
        d = docx.Document(f)
        parts = []
        for para in d.paragraphs:
            if para.text:
                parts.append(para.text)
        return "\n".join(parts)
    except Exception:
        return ""


def _read_txt_bytes(data: bytes) -> str:
    # best-effort decoding
    for enc in ("utf-8", "utf-16", "cp1256", "cp1252"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_upload(file: UploadFile, data: bytes) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        t = _read_pdf_bytes(data)
        return t.strip()
    if name.endswith(".docx"):
        t = _read_docx_bytes(data)
        return t.strip()
    # basic support
    return _read_txt_bytes(data).strip()


# -----------------------------
# Tokenization / stats
# -----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF]+", re.UNICODE)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return TOKEN_RE.findall(text)


def ngrams(tokens: List[str], n: int) -> List[str]:
    if n <= 1:
        return tokens
    out = []
    for i in range(0, len(tokens) - n + 1):
        out.append(" ".join(tokens[i : i + n]))
    return out


def build_doc_terms(text: str, max_tokens: int = 120_000) -> List[str]:
    # Large docs: cap tokens for stats to keep things fast; still representative.
    toks = tokenize(text)
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    # Use 1–3 grams for phrase power
    terms = []
    terms.extend(ngrams(toks, 1))
    terms.extend(ngrams(toks, 2))
    terms.extend(ngrams(toks, 3))
    return terms


@dataclass
class TermStats:
    tf: int
    df: int
    df_ratio: float


def compute_tf_df(docs_terms: List[List[str]]) -> Dict[str, TermStats]:
    tf_map: Dict[str, int] = {}
    df_map: Dict[str, int] = {}
    n_docs = len(docs_terms)

    for terms in docs_terms:
        seen = set()
        for t in terms:
            tf_map[t] = tf_map.get(t, 0) + 1
            if t not in seen:
                df_map[t] = df_map.get(t, 0) + 1
                seen.add(t)

    out: Dict[str, TermStats] = {}
    for term, tfv in tf_map.items():
        dfv = df_map.get(term, 0)
        out[term] = TermStats(tf=tfv, df=dfv, df_ratio=(dfv / max(1, n_docs)))
    return out


def top_terms_table_md(stats: Dict[str, TermStats], limit: int = 40) -> str:
    # Sort by DF desc then TF desc
    items = sorted(stats.items(), key=lambda kv: (kv[1].df, kv[1].tf), reverse=True)[:limit]
    lines = ["Term | TF | DF | DF%", "---|---:|---:|---:"]
    for term, st in items:
        dfp = int(round(st.df_ratio * 100))
        lines.append(f"{term} | {st.tf} | {st.df} | {dfp}%")
    return "\n".join(lines)


# -----------------------------
# LLM helpers
# -----------------------------
def _require_llm(client: Any, llm_required: bool):
    if llm_required and client is None:
        raise HTTPException(
            status_code=400,
            detail="LLM required but OPENAI_API_KEY is not set. Set OPENAI_API_KEY and restart backend.",
        )


def llm_json(client: Any, model: str, system: str, user: str) -> Dict[str, Any]:
    """
    Force JSON. If the model returns extra text, try to extract JSON object.
    """
    raw = llm_text(client, model, system, user).strip()
    if not raw:
        raise ValueError("Empty LLM output.")

    # Try direct JSON parse
    try:
        return json.loads(raw)
    except Exception:
        # Attempt to extract the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"LLM policy writer failed: Expecting value: line 1 column 1 (char 0). Raw: {raw[:200]}")
        return json.loads(m.group(0))


# -----------------------------
# Topic inference (fast, local)
# -----------------------------
def infer_topics_local(
    docs_text: List[str],
    stats: Dict[str, TermStats],
    max_topics: int,
    df_ratio_min: float,
    df_ratio_max: float,
) -> List[Dict[str, Any]]:
    """
    Simple heuristic topic inference:
    - pick discriminative phrases by DF window
    - cluster docs with TF-IDF+KMeans
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        # If sklearn missing, fallback to 1 topic from global phrases
        phrases = [
            t
            for t, st in stats.items()
            if df_ratio_min <= st.df_ratio <= df_ratio_max and 2 <= len(t) <= 60
        ]
        phrases = sorted(phrases, key=lambda t: (stats[t].df_ratio, stats[t].tf), reverse=True)[:25]
        return [
            {
                "topic_name": "General Topic",
                "doc_ids": list(range(len(docs_text))),
                "candidates": [{"phrase": p, "df_ratio": stats[p].df_ratio} for p in phrases],
            }
        ]

    n_docs = len(docs_text)
    if n_docs <= 1:
        phrases = sorted(stats.keys(), key=lambda t: (stats[t].df, stats[t].tf), reverse=True)[:20]
        return [
            {
                "topic_name": "Single Document Topic",
                "doc_ids": [0],
                "candidates": [{"phrase": p, "df_ratio": stats[p].df_ratio} for p in phrases],
            }
        ]

    # Choose topic count
    k = max(2, min(max_topics, int(round(math.sqrt(n_docs))) + 1))

    # TF-IDF vectorizer for bilingual (we already lowercased Arabic unaffected)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=25_000,
        ngram_range=(1, 3),
        token_pattern=r"(?u)[A-Za-z0-9\u0600-\u06FF]+",
        min_df=1,
    )
    X = vectorizer.fit_transform(docs_text)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    # Build clusters
    clusters: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)

    vocab = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_

    topics = []
    for lab, doc_ids in clusters.items():
        # Top terms by centroid weight, then filter by DF window
        weights = centers[lab]
        idxs = weights.argsort()[::-1]

        candidates = []
        for idx in idxs[:250]:
            phrase = vocab[idx]
            st = stats.get(phrase)
            if not st:
                continue
            if not (df_ratio_min <= st.df_ratio <= df_ratio_max):
                continue
            # avoid junky single-char / pure numbers
            if len(phrase) < 3:
                continue
            if phrase.isdigit():
                continue
            candidates.append({"phrase": phrase, "df_ratio": st.df_ratio})
            if len(candidates) >= 40:
                break

        topics.append(
            {
                "topic_name": f"Topic {lab + 1}",
                "doc_ids": doc_ids,
                "candidates": candidates,
            }
        )

    # Sort bigger clusters first
    topics.sort(key=lambda t: len(t["doc_ids"]), reverse=True)
    return topics[:max_topics]


# -----------------------------
# Pretty rule building (readable)
# -----------------------------
def _q(s: str) -> str:
    s = normalize_ws(s).strip().strip('"').strip("'")
    return f"\"{s}\""


def _or_group(items: List[str]) -> str:
    clean = [normalize_ws(x) for x in items if normalize_ws(x)]
    if not clean:
        return ""
    if len(clean) == 1:
        return _q(clean[0])
    return "(" + " OR ".join(_q(x) for x in clean) + ")"


def _and_group(groups: List[List[str]]) -> str:
    parts = []
    for g in groups:
        s = _or_group(g)
        if s:
            parts.append(s)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " AND ".join(parts) + ")"

def _expr_from_clauses(clauses: List[List[List[str]]]) -> str:
    # clauses: [ clause1_groups, clause2_groups, ...]
    clause_exprs = []
    for groups in clauses:
        e = _and_group(groups)
        if e:
            clause_exprs.append(e)
    if not clause_exprs:
        return ""
    if len(clause_exprs) == 1:
        return clause_exprs[0]
    return "(" + " OR ".join(clause_exprs) + ")"


def _format_rule_line(rule_name: str, groups: List[List[str]]) -> str:
    expr = _expr_from_clauses(groups)
    return f"`{expr}`"


def _safe_topic_title(name: str) -> str:
    name = normalize_ws(name)
    name = re.sub(r"[^\w\u0600-\u06FF\s&/-]+", "", name, flags=re.UNICODE).strip()
    return name[:80] if name else "Untitled Topic"


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    files: List[UploadFile] = File(...),
    llm_required: bool = Form(True),
    model: str = Form(os.getenv("OPENAI_MODEL", "gpt-5.2")),
    max_topics: int = Form(8),
    df_ratio_min: float = Form(0.10),
    df_ratio_max: float = Form(0.85),
    corpus_hint: str = Form(""),
):
    client = get_llm_client()
    _require_llm(client, llm_required)

    docs_text: List[str] = []
    docs_meta: List[Dict[str, Any]] = []

    for f in files:
        data = await f.read()
        text = extract_text_from_upload(f, data)
        text = normalize_ws(text)
        if not text:
            continue
        docs_text.append(text)
        docs_meta.append({"filename": f.filename or "unknown", "bytes": len(data)})

    if not docs_text:
        raise HTTPException(status_code=400, detail="No readable text extracted from uploaded files.")

    # build stats
    docs_terms = [build_doc_terms(t) for t in docs_text]
    stats = compute_tf_df(docs_terms)

    # local topic inference
    topics = infer_topics_local(
        docs_text=docs_text,
        stats=stats,
        max_topics=max(2, min(max_topics, 20)),
        df_ratio_min=df_ratio_min,
        df_ratio_max=df_ratio_max,
    )

    # LLM refinement for:
    # - naming topics
    # - proposing stopwords/generic terms to avoid in policy
    # - selecting "high-signal phrases per topic"
    # We do NOT hardcode stopwords; the model proposes them based on this corpus + hint.
    system = (
        "You are a DLP policy SME. You will help label topics and identify generic words/stopwords "
        "that would cause false positives. You must be conservative and practical."
    )

    # Build compact corpus summary for LLM (do not send full 50 pages)
    # Provide top DF terms and topic candidate phrases with DF%.
    top_global = sorted(stats.items(), key=lambda kv: (kv[1].df, kv[1].tf), reverse=True)[:60]
    top_global_lines = [f"- {t} (DF={st.df}, DF%={int(st.df_ratio*100)}%)" for t, st in top_global]

    topic_lines = []
    for t in topics:
        cands = t.get("candidates", [])[:25]
        cand_lines = [f"{c['phrase']} ({int(c['df_ratio']*100)}%)" for c in cands]
        topic_lines.append(
            f"- {t['topic_name']} | docs={len(t['doc_ids'])}\n  candidates: " + ", ".join(cand_lines)
        )

    user = (
        f"Department hint (optional): {corpus_hint}\n\n"
        "We extracted topics from uploaded docs. Please:\n"
        "1) Rename each topic to a short, business-meaningful name.\n"
        "2) Provide a list of 'generic terms/stopwords' that should be avoided in rules because they are too broad.\n"
        "3) For each topic, choose 10–18 high-signal phrases from candidates that are likely to appear in OTHER similar department docs.\n"
        "   Prefer mid DF% (not 100% generic, not 1-doc identifiers). Keep bilingual phrases if helpful.\n\n"
        "Return STRICT JSON with keys:\n"
        "{\n"
        '  "generic_terms": [..],\n'
        '  "topics": [\n'
        '     {"old_name":"Topic 1","new_name":"...","phrases":[{"p":"...", "df_pct": 25}, ...]},\n'
        "  ]\n"
        "}\n\n"
        "Global top terms:\n" + "\n".join(top_global_lines) + "\n\n"
        "Topic candidates:\n" + "\n".join(topic_lines)
    )

    refined = None
    if client is not None:
        refined = llm_json(client, model, system, user)

    generic_terms: List[str] = []
    renamed_topics: Dict[str, str] = {}
    topic_phrases: Dict[str, List[str]] = {}

    if refined and isinstance(refined, dict):
        generic_terms = [normalize_ws(x) for x in refined.get("generic_terms", []) if normalize_ws(x)]
        for t in refined.get("topics", []):
            old = normalize_ws(str(t.get("old_name", "")))
            new = _safe_topic_title(str(t.get("new_name", "")))
            if old and new:
                renamed_topics[old] = new
            phr = []
            for item in t.get("phrases", []):
                p = normalize_ws(str(item.get("p", "")))
                if p:
                    phr.append(p)
            if old and phr:
                topic_phrases[old] = phr

    # Build markdown preview
    preview_lines = [
        "Topics (preview)",
        "High-signal phrases per topic (DF% hints generalization).",
        "",
    ]
    for t in topics:
        old_name = t["topic_name"]
        name = renamed_topics.get(old_name, old_name)
        docs_count = len(t["doc_ids"])
        preview_lines.append(f"### {name}")
        preview_lines.append(f"{docs_count} docs")
        phrases = topic_phrases.get(old_name)
        if not phrases:
            # fallback to local candidates
            phrases = [c["phrase"] for c in t.get("candidates", [])[:12]]
        for p in phrases[:14]:
            st = stats.get(p)
            dfpct = int(round((st.df_ratio if st else 0.0) * 100))
            preview_lines.append(f"- {p} ({dfpct}%)")
        preview_lines.append("")

    top_md = top_terms_table_md(stats, limit=45)

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "docs_text": docs_text,
        "docs_meta": docs_meta,
        "stats": stats,
        "topics_local": topics,
        "generic_terms": generic_terms,
        "topic_names": renamed_topics,
        "topic_phrases": topic_phrases,
        "corpus_hint": corpus_hint,
    }

    # Build structured topics_preview (for frontend)
    topics_preview_struct: List[TopicPreview] = []
    for t in topics:
        old_name = t["topic_name"]
        name = renamed_topics.get(old_name, old_name)
        doc_count = len(t["doc_ids"])

        # phrases: prefer LLM-picked phrases; fallback to local candidates
        phrases = topic_phrases.get(old_name)
        if not phrases:
            phrases = [c["phrase"] for c in t.get("candidates", [])[:12]]

        phrase_rows: List[PhraseRow] = []
        for p in phrases[:18]:
            st = stats.get(p)
            df_ratio = float(st.df_ratio) if st else 0.0
            phrase_rows.append(PhraseRow(text=p, df_ratio=df_ratio))

        topics_preview_struct.append(
            TopicPreview(name=name, doc_count=doc_count, phrases=phrase_rows)
        )

    # Build structured top_terms (for frontend)
    top_terms_struct: List[TermRow] = []
    for term, st in top_global:
        top_terms_struct.append(
            TermRow(term=term, tf=int(st.tf), df=int(st.df), df_ratio=float(st.df_ratio))
        )

    return AnalyzeResponse(
        session_id=session_id,
        n_docs=len(docs_text),
        topics_preview_md="\n".join(preview_lines),
        top_terms_md=top_md,

        # ✅ NEW structured outputs
        topics_preview=topics_preview_struct,
        top_terms=top_terms_struct,
        generic_terms=generic_terms,
    )

# -----------------------------
# Co-occurrence helpers (topic-level AND gating)
# -----------------------------
from collections import defaultdict
from typing import Iterable

def _norm_lc(s: str) -> str:
    return normalize_ws(s).lower()

def _phrase_present(text_lc: str, phrase: str) -> bool:
    """
    Symantec-like-ish: match phrase as a 'token boundary' rather than raw substring.
    For English/Arabic, \w includes letters/digits/underscore and is Unicode-aware.
    This prevents 'po' matching 'support' etc.
    """
    p = _norm_lc(phrase)
    if not p:
        return False
    # escape phrase for regex
    esc = re.escape(p)
    # token-boundary: not preceded or followed by a word char
    pat = re.compile(rf"(?<!\w){esc}(?!\w)", flags=re.UNICODE)
    return bool(pat.search(text_lc))

def build_presence_matrix(
    docs_text: List[str],
    phrases: List[str],
) -> Tuple[List[List[bool]], List[str]]:
    """
    Returns:
      presence[doc_i][phrase_j] = True if phrase_j is present in doc_i
      phrases_norm = normalized phrase list (same order)
    """
    phrases_norm = [normalize_ws(p) for p in phrases if normalize_ws(p)]
    presence: List[List[bool]] = []
    for d in docs_text:
        tlc = _norm_lc(d)
        row = [ _phrase_present(tlc, p) for p in phrases_norm ]
        presence.append(row)
    return presence, phrases_norm

def cooc_stats_for_phrases(
    presence: List[List[bool]],
) -> Tuple[List[int], Dict[Tuple[int,int], int]]:
    """
    Returns:
      df[j] = number of docs where phrase j appears
      pair_df[(i,j)] = number of docs where both i and j appear (i<j)
    """
    n_docs = len(presence)
    if n_docs == 0:
        return [], {}

    m = len(presence[0]) if presence else 0
    df = [0] * m
    pair_df: Dict[Tuple[int,int], int] = defaultdict(int)

    for row in presence:
        idxs = [i for i, v in enumerate(row) if v]
        for i in idxs:
            df[i] += 1
        # pair counts
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair_df[(idxs[a], idxs[b])] += 1

    return df, dict(pair_df)

def lift_score(p_ab: float, p_a: float, p_b: float) -> float:
    if p_a <= 0 or p_b <= 0:
        return 0.0
    return p_ab / (p_a * p_b)

def propose_and_groups_from_cooc(
    phrases: List[str],
    df: List[int],
    pair_df: Dict[Tuple[int,int], int],
    n_docs: int,
    *,
    generic_terms: Iterable[str],
    max_groups: int = 4,
    group_size: int = 6,
    min_df_docs: int = 2,
    min_pair_docs: int = 2,
) -> List[List[str]]:
    """
    Build 2–4 AND-groups (each is an OR-list) based on co-occurrence.
    Strategy:
      - pick 2–3 anchor phrases with mid DF (not too rare, not too common)
      - collect high-lift phrases that co-occur with anchors
      - output groups: [anchors], [context], [optional metadata/controls]
    """
    gset = { _norm_lc(x) for x in (generic_terms or []) if _norm_lc(x) }
    m = len(phrases)

    # candidate phrase indices
    cand = []
    for i, p in enumerate(phrases):
        pl = _norm_lc(p)
        if not pl or pl in gset:
            continue
        if df[i] < min_df_docs:
            continue
        # avoid pure numbers
        if pl.isdigit():
            continue
        # mid-DF preference: df_ratio*(1-df_ratio)
        r = df[i] / max(1, n_docs)
        midness = r * (1.0 - r)
        cand.append((midness, df[i], i))

    if not cand:
        # fallback: single group with whatever we have
        return [phrases[: min(group_size, len(phrases))]]

    cand.sort(reverse=True)
    anchor_idxs = [c[2] for c in cand[:3]]  # up to 3 anchors

    # compute best context phrases by max lift to ANY anchor
    context_scores = []
    for j in range(m):
        if j in anchor_idxs:
            continue
        if df[j] < min_df_docs:
            continue
        pj = _norm_lc(phrases[j])
        if not pj or pj in gset:
            continue

        best = 0.0
        best_pair_docs = 0
        for a in anchor_idxs:
            i, k = (a, j) if a < j else (j, a)
            pab_docs = pair_df.get((i, k), 0)
            if pab_docs < min_pair_docs:
                continue
            p_ab = pab_docs / max(1, n_docs)
            p_a = df[a] / max(1, n_docs)
            p_b = df[j] / max(1, n_docs)
            l = lift_score(p_ab, p_a, p_b)
            if l > best:
                best = l
                best_pair_docs = pab_docs

        if best > 0:
            # prefer higher lift, then more pair support
            context_scores.append((best, best_pair_docs, df[j], j))

    context_scores.sort(reverse=True)

    anchors = [phrases[i] for i in anchor_idxs][: min(group_size, len(anchor_idxs))]
    context = [phrases[t[3]] for t in context_scores][:group_size]

    groups: List[List[str]] = []
    if anchors:
        groups.append(anchors)
    if context:
        groups.append(context)

    # Optional: add a 3rd group of "control" terms (slightly higher DF)
    # This helps Symantec-style precision: require doc-type / process words too.
    controls = []
    for _, _, _, j in context_scores[group_size : group_size + 20]:
        r = df[j] / max(1, n_docs)
        if 0.35 <= r <= 0.85:
            controls.append(phrases[j])
        if len(controls) >= group_size:
            break
    if controls and len(groups) < max_groups:
        groups.append(controls)

    return groups[:max_groups]

from typing import Dict, List, Any, Optional, Tuple, Set

from typing import Any, Dict, List, Optional, Set, Tuple
import math

# -----------------------------
# Whitelist helpers
# -----------------------------
def _flatten_groups(groups: List[List[str]]) -> List[str]:
    out: List[str] = []
    for g in (groups or []):
        if isinstance(g, list):
            for x in g:
                x2 = normalize_ws(str(x))
                if x2:
                    out.append(x2)
    return out


def _is_letter_garbage(s: str) -> bool:
    """
    Reject strings like 'p r o c u r e m e n t' or 'a c m e'.
    Heuristic: if >50% of tokens are 1-char, treat as garbage.
    """
    toks = tokenize_for_match(s)
    if len(toks) >= 4:
        one = sum(1 for t in toks if len(t) == 1)
        if one / max(1, len(toks)) >= 0.5:
            return True
    return False


def _allowed_phrase_map(candidates: List[str], proposed_groups: List[List[str]]) -> Dict[str, str]:
    """
    Map normalized->canonical phrase. Only phrases we provided are allowed.
    Keys are normalize_for_match(phrase).
    """
    m: Dict[str, str] = {}
    for p in (candidates or []) + _flatten_groups(proposed_groups or []):
        p2 = normalize_ws(str(p))
        if not p2:
            continue

        # reject garbage early
        if _is_letter_garbage(p2):
            continue

        key = normalize_for_match(p2)

        # block tiny keys
        if len(key) < 3:
            continue

        # keep first seen canonical form
        m.setdefault(key, p2)
    return m


def _clean_phrase_with_map(
    phrase: Any,
    allowed_map: Dict[str, str],
    generic_terms: List[str],
) -> Optional[str]:
    """
    Returns canonical phrase ONLY if it is whitelisted in allowed_map,
    and not generic, and not tiny/letter garbage.
    """
    p = normalize_ws(str(phrase))
    if not p:
        return None

    # reject letter-split garbage
    if _is_letter_garbage(p):
        return None

    key = normalize_for_match(p)

    # tiny/letter garbage
    if len(key) < 3:
        return None
    ptoks = tokenize_for_match(p)
    if len(ptoks) == 1 and len(ptoks[0]) < 3:
        return None

    # must exist in allowed_map (hard whitelist)
    canon = allowed_map.get(key)
    if not canon:
        return None

    # remove generic terms
    gset = {normalize_for_match(x) for x in (generic_terms or []) if normalize_ws(x)}
    if normalize_for_match(canon) in gset:
        return None

    return canon


# -----------------------------
# Expr & coverage helpers
# -----------------------------
def _groups_to_expr(groups: List[List[str]]) -> str:
    # AND of OR-lists
    return _and_group(groups)


def _docs_covered_by_expr(expr: str, docs: List[str]) -> Set[int]:
    covered: Set[int] = set()
    for i, d in enumerate(docs):
        ok, _hits = test_rule(expr, d)
        if ok:
            covered.add(i)
    return covered


def _rule_penalty(groups: List[List[str]], stats: Dict[str, TermStats]) -> float:
    """
    Penalize broad/weak rules so coverage-aware selection doesn't pick junk.
    - too few AND groups
    - very generic phrases (high df_ratio)
    - too-small OR groups
    """
    if len(groups) < 2:
        return 999.0

    pen = 0.0

    # penalize small OR groups (encourages robust OR-lists)
    for g in groups:
        if len(g) < 2:
            pen += 2.0
        elif len(g) < 3:
            pen += 1.0

    # penalize high-DF terms
    for g in groups:
        for p in g:
            st = stats.get(p)
            if not st:
                continue
            if st.df_ratio >= 0.85:
                pen += 2.0
            elif st.df_ratio >= 0.70:
                pen += 1.0

    return pen


def _enforce_symantec_shape(groups: List[List[str]]) -> List[List[str]]:
    """
    Enforce: 2–4 AND-groups, each 3–8 phrases when possible.
    We'll trim oversize groups and drop tiny ones.
    """
    # drop empty
    groups = [g for g in groups if isinstance(g, list) and len(g) > 0]

    # trim phrases within each group
    fixed: List[List[str]] = []
    for g in groups:
        # dedupe preserving order
        seen = set()
        g2: List[str] = []
        for x in g:
            k = normalize_for_match(x)
            if k in seen:
                continue
            seen.add(k)
            g2.append(x)

        # cap to 8
        g2 = g2[:8]

        # drop tiny groups (we want OR lists, not single anchors)
        if len(g2) >= 2:
            fixed.append(g2)

    # cap AND groups
    fixed = fixed[:4]

    # must have at least 2 AND groups to be meaningful
    if len(fixed) < 2:
        return []

    return fixed


@app.post("/api/generate_pretty", response_class=PlainTextResponse)
def generate_pretty(req: GeneratePrettyRequest):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session_id. Run /api/analyze first.")

    client = get_llm_client()
    _require_llm(client, req.llm_required)

    stats: Dict[str, TermStats] = sess["stats"]
    topics_local: List[Dict[str, Any]] = sess["topics_local"]
    generic_terms: List[str] = sess.get("generic_terms", []) or []
    topic_names: Dict[str, str] = sess.get("topic_names", {}) or {}
    topic_phrases: Dict[str, List[str]] = sess.get("topic_phrases", {}) or {}
    corpus_hint = req.corpus_hint or sess.get("corpus_hint", "")
    docs_all: List[str] = sess["docs_text"]

    base_max = max(1, min(int(req.max_rules), 20))
    hard_cap = 20
    extra_cap = min(8, max(0, hard_cap - base_max))
    desired_coverage = 0.85

    topics = topics_local[: min(len(topics_local), base_max + extra_cap)]

    topics_for_llm: List[Dict[str, Any]] = []
    allowed_maps_by_topic: List[Dict[str, str]] = []
    fallback_groups_by_topic: List[List[List[str]]] = []  # ✅ store AND-groups directly

    for t in topics:
        old_name = t["topic_name"]
        name = topic_names.get(old_name, old_name)

        phrases = topic_phrases.get(old_name)
        if not phrases:
            phrases = [c["phrase"] for c in t.get("candidates", [])[:30]]

        candidates: List[str] = []
        for p in phrases:
            p = normalize_ws(p)
            if not p:
                continue
            if p not in stats:
                continue

            if _is_letter_garbage(p):
                continue

            if len(normalize_for_match(p)) < 3:
                continue

            ptoks = tokenize_for_match(p)
            if len(ptoks) == 1 and len(ptoks[0]) < 3:
                continue

            st = stats.get(p)
            if not st:
                continue
            if not (req.df_ratio_min <= st.df_ratio <= req.df_ratio_max):
                continue
            if p.isdigit():
                continue

            candidates.append(p)

        doc_ids = t.get("doc_ids", [])
        topic_docs = [docs_all[i] for i in doc_ids if 0 <= i < len(docs_all)]
        n_topic_docs = len(topic_docs)

        proposed_groups: List[List[str]] = []
        if n_topic_docs >= 2 and len(candidates) >= 8:
            presence, cand_norm = build_presence_matrix(topic_docs, candidates[:40])
            df_local, pair_df = cooc_stats_for_phrases(presence)
            proposed_groups = propose_and_groups_from_cooc(
                phrases=cand_norm,
                df=df_local,
                pair_df=pair_df,
                n_docs=n_topic_docs,
                generic_terms=generic_terms,
                max_groups=4,
                group_size=6,
                min_df_docs=2,
                min_pair_docs=2,
            )

        if not proposed_groups:
            proposed_groups = [
                candidates[:6],
                candidates[6:12] if len(candidates) > 6 else candidates[:6],
            ]

        allowed_map = _allowed_phrase_map(candidates, proposed_groups)
        allowed_maps_by_topic.append(allowed_map)
        fallback_groups_by_topic.append(proposed_groups)  # ✅ correct shape

        topics_for_llm.append(
            {
                "topic_name": _safe_topic_title(name),
                "doc_count": len(doc_ids),
                "candidate_phrases": [
                    {"p": p, "df_pct": int(round(stats[p].df_ratio * 100)) if p in stats else 0}
                    for p in candidates[:25]
                ],
                "proposed_groups_from_cooccurrence": proposed_groups,
            }
        )

    system = (
        "You write Symantec-style DLP keyword rules using a boolean engine.\n"
        "You MUST ONLY use phrases that appear in candidate_phrases.p OR proposed_groups_from_cooccurrence.\n"
        "Do NOT invent phrases. Do NOT split words into letters. Do NOT output single characters.\n"
        "Rule shape:\n"
        "- Each rule is AND of 2–4 groups.\n"
        "- Each group is an OR-list of 3–8 phrases (no singletons).\n"
        "No proximity logic. No regex inside rules.\n"
        "Output strict JSON only."
    )

    user = {
        "policy_title": req.policy_title,
        "department_hint": corpus_hint,
        "generic_terms_to_avoid": generic_terms[:60],
        "topics": topics_for_llm,
        "required_schema": {
            "rules": [
                {
                    "rule_title": "Short readable name",
                    "groups": [
                        ["phrase1", "phrase2", "phrase3"],
                        ["phrase4", "phrase5", "phrase6"],
                    ],
                    "why": ["bullet 1", "bullet 2", "bullet 3"],
                }
            ],
            "optional_regex_suggestions": [
                {"name": "IBAN", "pattern": "\\b...\\b", "why": "short reason"}
            ],
        },
    }

    structured: Dict[str, Any] = {}
    if client is not None:
        structured = llm_json(client, req.model, system, json.dumps(user, ensure_ascii=False))

    rules = structured.get("rules", []) if isinstance(structured, dict) else []
    if not rules or not isinstance(rules, list):
        raise HTTPException(
            status_code=500,
            detail="LM policy writer failed: returned no rules.",
        )

    # -----------------------------
    # Clean rules with HARD whitelist + robust fallback
    # -----------------------------
    cleaned_rules: List[Dict[str, Any]] = []
    for idx, r in enumerate(rules[: len(topics_for_llm)]):
        title = _safe_topic_title(str(r.get("rule_title", f"Rule {idx+1}")))
        groups = r.get("groups", [])

        allowed_map = allowed_maps_by_topic[idx] if idx < len(allowed_maps_by_topic) else {}
        fallback_groups = fallback_groups_by_topic[idx] if idx < len(fallback_groups_by_topic) else []

        clean_groups: List[List[str]] = []
        if isinstance(groups, list):
            for g in groups:
                if not isinstance(g, list):
                    continue
                gg: List[str] = []
                for x in g:
                    cp = _clean_phrase_with_map(x, allowed_map, generic_terms)
                    if cp:
                        gg.append(cp)
                if gg:
                    clean_groups.append(gg)

        # enforce Symantec-like shape
        clean_groups = _enforce_symantec_shape(clean_groups)

        # fallback if wiped / invalid
        if not clean_groups:
            fb = []
            for g in fallback_groups:
                g2 = []
                for x in g:
                    cp = _clean_phrase_with_map(x, allowed_map, generic_terms) or normalize_ws(x)
                    if cp and not _is_letter_garbage(cp):
                        g2.append(cp)
                if g2:
                    fb.append(g2)
            clean_groups = _enforce_symantec_shape(fb)

        if not clean_groups:
            continue

        expr = _groups_to_expr(clean_groups)
        cleaned_rules.append(
            {
                "title": title,
                "groups": clean_groups,
                "expr": expr,
                "why": r.get("why", []),
            }
        )

    if not cleaned_rules:
        raise HTTPException(
            status_code=500,
            detail="All generated rules were removed by whitelist cleaning.",
        )

    # -----------------------------
    # Coverage-aware selection with penalty (precision-aware)
    # -----------------------------
    n_docs = len(docs_all)
    target_docs = int(math.ceil(desired_coverage * max(1, n_docs)))

    for rr in cleaned_rules:
        rr["covered"] = _docs_covered_by_expr(rr["expr"], docs_all)

    selected: List[Dict[str, Any]] = []
    covered: Set[int] = set()
    remaining = cleaned_rules[:]

    while remaining and len(covered) < target_docs and len(selected) < hard_cap:
        best_i = -1
        best_score = -1e9
        for i, rr in enumerate(remaining):
            gain = len(rr["covered"] - covered)
            pen = _rule_penalty(rr["groups"], stats)
            score = gain - pen
            if score > best_score:
                best_score = score
                best_i = i

        if best_i < 0:
            break

        pick = remaining.pop(best_i)
        gain = len(pick["covered"] - covered)
        if gain <= 0:
            break

        selected.append(pick)
        covered |= pick["covered"]

        if len(selected) >= base_max and len(covered) >= target_docs:
            break

    # ensure at least base_max
    if len(selected) < base_max:
        for rr in cleaned_rules:
            if rr in selected:
                continue
            selected.append(rr)
            if len(selected) >= base_max or len(selected) >= hard_cap:
                break

    # -----------------------------
    # Output markdown
    # -----------------------------
    out_lines: List[str] = []
    out_lines.append(f"# {req.policy_title}")
    out_lines.append("")
    out_lines.append("Format: Symantec-style boolean rules (AND groups of OR phrases). No proximity logic.")
    out_lines.append("")
    out_lines.append("## Rules")
    out_lines.append("")

    for i, rr in enumerate(selected, start=1):
        out_lines.append(f"## Rule {i}: {rr['title']}")
        out_lines.append(f"`{rr['expr']}`")
        out_lines.append("")
        out_lines.append("**Why chosen:**")
        why = rr.get("why", [])
        if isinstance(why, list) and why:
            for b in why[:5]:
                b2 = normalize_ws(str(b))
                if b2:
                    out_lines.append(f"- {b2}")
        else:
            out_lines.append("- Built from co-occurrence and whitelisted topic phrases to balance recall and precision.")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")

    if req.include_regex_suggestions:
        regex_list = structured.get("optional_regex_suggestions", [])
        if isinstance(regex_list, list) and regex_list:
            out_lines.append("## Optional Regex Suggestions (separate from rules)")
            out_lines.append("")
            for item in regex_list[: max(0, int(req.max_regex_suggestions))]:
                name = normalize_ws(str(item.get("name", "")))
                pat = str(item.get("pattern", "")).strip()
                why = normalize_ws(str(item.get("why", "")))
                if not name or not pat:
                    continue
                out_lines.append(f"- **{name}:** `{pat}`" + (f" — {why}" if why else ""))
            out_lines.append("")

    return "\n".join(out_lines)
