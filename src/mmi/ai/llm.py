"""Provider-agnostic LLM client — the project's 'future-proofing' seam.

Swap models with one env var (``LLM_PROVIDER`` = gemini | groq | claude). Defaults to a
**free** provider so the platform stays at £0; Claude is a drop-in when you want it.
If no key is configured, callers fall back to a deterministic template (see narrative.py).
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from mmi.settings import settings
from mmi.utils.logging import get_logger

log = get_logger("ai.llm")

# Default models per provider (override here as model names evolve).
MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "claude": "claude-sonnet-4-6",
}


def _key() -> str:
    return {
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
        "claude": settings.anthropic_api_key,
    }[settings.llm_provider]


def available() -> bool:
    """True if the selected provider has an API key configured."""
    return bool(_key())


def provider_model() -> str:
    return f"{settings.llm_provider}:{MODELS[settings.llm_provider]}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def complete(prompt: str, *, system: str | None = None, max_tokens: int = 800) -> str:
    """Return a completion from the configured provider."""
    provider = settings.llm_provider
    if provider == "gemini":
        return _gemini(prompt, system, max_tokens)
    if provider == "groq":
        return _groq(prompt, system, max_tokens)
    if provider == "claude":
        return _claude(prompt, system, max_tokens)
    raise ValueError(f"unknown provider {provider}")


def _gemini(prompt: str, system: str | None, max_tokens: int) -> str:
    model = MODELS["gemini"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    with httpx.Client(timeout=60) as client:
        r = client.post(url, params={"key": _key()}, json=body)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _groq(prompt: str, system: str | None, max_tokens: int) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {_key()}"},
            json={"model": MODELS["groq"], "messages": messages, "max_tokens": max_tokens},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def _claude(prompt: str, system: str | None, max_tokens: int) -> str:
    body: dict = {
        "model": MODELS["claude"],
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": _key(), "anthropic-version": "2023-06-01"},
            json=body,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
