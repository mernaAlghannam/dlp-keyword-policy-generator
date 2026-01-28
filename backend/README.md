Backend - DLP Keyword Policy Generator (FastAPI)

Run:
  cd backend
  python -m venv .venv
  source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
  pip install -r requirements.txt
  uvicorn app.main:app --reload --port 8000

Optional LLM:
  (A) OpenAI:
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=...
    export OPENAI_MODEL=gpt-4.1-mini

  (B) Local JSON HTTP:
    export LLM_PROVIDER=local_http_json
    export LOCAL_LLM_URL=http://localhost:8080/refine
