# DLP Keyword Policy Generator (Symantec-style)

A local prototype that helps you **generate high-signal Symantec DLP keyword rules** from a small set of uploaded department documents (Arabic + English), then **test coverage** of those rules against files and see **which phrases actually caused a match**.

This is designed for DLP teams who want better keyword policies (fewer false positives, fewer missed hits) without relying on brittle, hand-written rules.

---
## Preview

<p align="center">
  <img src="docs/images/img1.png" alt="CSV Dashboard Generator UI" width="900" />
</p>

<p align="center">
  <img src="docs/images/img2.png" alt="CSV Dashboard Generator UI" width="900" />
</p>
## What it does

### 1) Analyze uploaded documents (`/api/analyze`)
- Extracts text from:
  - `.txt`
  - `.docx`
  - `.pdf` (best-effort; depends on PDF text extractability)
- Builds TF/DF statistics using **1–3 gram phrases** (English + Arabic tokens).
- Infers “topics” (clusters) locally (TF-IDF + KMeans when available).
- Optionally uses an LLM to:
  - Rename topics into business-meaningful labels
  - Suggest **generic terms** to avoid in rules (stopword-like)
  - Select **high-signal phrases** per topic for generalization

### 2) Generate Symantec-style boolean rules (`/api/generate_pretty`)
Outputs a human-readable “pretty” policy like:

- One rule per topic
- Rule format is **OR of clauses**
- Each clause is **AND of groups**
- Each group is an **OR list of phrases**

Example shape:
(
("vendor acme" OR "acme industrial supplies")
AND
("procurement" OR "pricing" OR "supplies ltd")
)
OR
(
("اكمي للتوريدات" OR "اكمي للتوريدات الصناعية")
AND
("اختبار" OR "عينة" OR "رمز المشروع")
)


Key protections built in:
- **Hard whitelist enforcement**: the model can ONLY use phrases you provided (no inventing).
- **Co-occurrence grouping**: proposes AND-groups from phrases that co-occur in the same topic docs.
- **Coverage-aware logic**: can add extra rules (within a cap) to raise overall doc coverage.

---

## Typical workflow

1. Upload sample department documents (Arabic/English)
2. Run `/api/analyze` to get:
   - Topics preview
   - DF hints (generalization)
   - Proposed phrases and generic terms
3. Run `/api/generate_pretty` to get:
   - Symantec-style boolean rules (pretty output)
   - Optional regex suggestions (kept separate)
4. Upload more documents (including “should NOT match” examples) and run `/api/test_policy`
5. Iterate: widen DF window / add docs / adjust caps until coverage & precision look good

---




What it does:
- Upload a set of confidential documents for a department
- Mines high-signal candidate terms using TF-IDF n-grams (English+Arabic)
- Lets you manually INCLUDE/EXCLUDE terms
- Generates a Symantec-like keyword policy:
    - boolean logic
    - proximity NEAR/k
    - threshold (min matches)
- Optional: LLM refinement (OpenAI or local HTTP endpoint)

Run backend:
  cd backend
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8000

Run frontend:
  cd frontend
  npm install
  npm run dev
  open http://localhost:5173

Optional LLM:
  Use no LLM:
    (do nothing)

  OpenAI:
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=...
    export OPENAI_MODEL=gpt-4.1-mini

  Local endpoint that returns JSON:
    export LLM_PROVIDER=local_http_json
    export LOCAL_LLM_URL=http://localhost:8080/refine
