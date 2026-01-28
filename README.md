DLP Keyword Policy Generator (Frontend + Backend)

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
