"""Scrub secrets from text before it is logged or persisted.

API keys that travel as URL query params (FRED ``api_key``, Gemini ``key``) end up inside
httpx error strings; without scrubbing they would land in ``raw.pipeline_runs.message`` and CI
logs. Apply :func:`redact` wherever an exception becomes a stored/logged string.
"""

from __future__ import annotations

import re

# Query-param secrets, e.g. ...?api_key=SECRET&...  /  ?key=SECRET  /  &token=SECRET
_QUERY = re.compile(r"(?i)([?&](?:api_?key|access_token|motherduck_token|token|key)=)[^&\s'\"\\]+")
# Authorization: Bearer <token>
_BEARER = re.compile(r"(?i)(bearer\s+)[^\s'\"]+")


def redact(text: str) -> str:
    """Return ``text`` with known secret values replaced by ``***``."""
    text = _QUERY.sub(r"\1***", text)
    return _BEARER.sub(r"\1***", text)
