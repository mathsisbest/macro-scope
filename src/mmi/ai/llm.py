"""Provider-agnostic LLM client — the project's 'future-proofing' seam.

Swap models with one env var (``LLM_PROVIDER`` = gemini | groq | claude). Defaults to a
**free** provider so the platform stays at £0; Claude is a drop-in when you want it.
If no key is configured, callers fall back to a deterministic template (see narrative.py).
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    reraise=True,
)
def complete(
    prompt: str, *, system: str | None = None, max_tokens: int = 800
) -> tuple[str, str]:
    """Return ``(text, engine)`` with multi-provider failover fallback.

    ``engine`` is the provider:model that actually served the response
    (e.g. ``"groq:llama-3.3-70b-versatile"``), which may differ from
    ``settings.llm_provider`` when a fallback was used.

    Tries configured primary provider (settings.llm_provider), and falls over
    to secondary providers (gemini -> groq -> claude) if primary API is unavailable
    or rate-limited.
    """
    primary = settings.llm_provider
    providers_order = [primary] + [p for p in ("gemini", "groq", "claude") if p != primary]

    last_error = None
    for p in providers_order:
        try:
            if p == "gemini" and settings.gemini_api_key:
                return _gemini(prompt, system, max_tokens), f"{p}:{MODELS[p]}"
            if p == "groq" and settings.groq_api_key:
                return _groq(prompt, system, max_tokens), f"{p}:{MODELS[p]}"
            if p == "claude" and settings.anthropic_api_key:
                return _claude(prompt, system, max_tokens), f"{p}:{MODELS[p]}"
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM provider '%s' failed: %s; trying next provider...", p, exc)
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError("No configured LLM provider has a valid API key.")


def _gemini(prompt: str, system: str | None, max_tokens: int) -> str:
    model = MODELS["gemini"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            # Gemini thinking effort (low|medium|high).
            "thinkingConfig": {"thinkingLevel": settings.gemini_thinking_level},
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    with httpx.Client(timeout=60) as client:
        r = client.post(url, params={"key": _key()}, json=body)
        r.raise_for_status()
        candidate = r.json()["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            # Thinking can exhaust maxOutputTokens before any answer text is emitted
            # (finishReason=MAX_TOKENS). Fail loudly so the caller falls back to the template.
            raise RuntimeError(
                f"Gemini returned no text (finishReason={candidate.get('finishReason')}); "
                "raise max_tokens or lower GEMINI_THINKING_LEVEL"
            )
        return parts[0]["text"].strip()


def _groq(prompt: str, system: str | None, max_tokens: int) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
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
            headers={"x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01"},
            json=body,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
