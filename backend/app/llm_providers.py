# backend/app/llm_providers.py
from __future__ import annotations

import os
from typing import Any, Optional

# Uses the modern OpenAI Python SDK:
# pip install openai
#
# Env vars supported:
# - OPENAI_API_KEY (required if llm_required=True)
# - OPENAI_MODEL (optional, default "gpt-5.2")
# - OPENAI_BASE_URL (optional, for proxies / gateways)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


def get_llm_client() -> Optional[Any]:
    """
    Returns an OpenAI client instance if OPENAI_API_KEY is set,
    otherwise returns None (so the app can run in no-LLM mode).
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from e

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def llm_text(client: Any, model: str, system: str, user: str) -> str:
    """
    Calls OpenAI Chat Completions and returns plain text.
    This avoids the 'client.responses' mismatch error.

    NOTE:
    - We intentionally use chat.completions.create for broad compatibility.
    """
    if client is None:
        raise RuntimeError("LLM client is None (missing OPENAI_API_KEY).")

    use_model = model or DEFAULT_MODEL

    resp = client.chat.completions.create(
        model=use_model,
        messages=[
            {"role": "system", "content": system or ""},
            {"role": "user", "content": user or ""},
        ],
        temperature=0.2,
    )

    # OpenAI SDK returns choices[0].message.content
    content = (resp.choices[0].message.content or "").strip()
    return content
