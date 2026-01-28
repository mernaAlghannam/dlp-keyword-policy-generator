from __future__ import annotations
import io
import re
from typing import Optional

import pdfplumber
from docx import Document

MAX_CHARS = 2_000_000  # safety cap

def _clean_text(s: str) -> str:
    s = s.replace("\u0000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
        return _clean_text(text)[:MAX_CHARS]

    if lower.endswith(".docx"):
        f = io.BytesIO(content)
        doc = Document(f)
        parts = []
        for p in doc.paragraphs:
            if p.text:
                parts.append(p.text)
        text = "\n".join(parts)
        return _clean_text(text)[:MAX_CHARS]

    if lower.endswith(".pdf"):
        f = io.BytesIO(content)
        parts = []
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages[:50]:  # cap pages
                t = page.extract_text() or ""
                if t:
                    parts.append(t)
        text = "\n".join(parts)
        return _clean_text(text)[:MAX_CHARS]

    raise ValueError("Unsupported file type. Use .txt, .docx, or .pdf")
