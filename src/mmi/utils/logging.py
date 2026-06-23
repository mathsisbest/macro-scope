"""Minimal structured-ish logging helper."""

from __future__ import annotations

import logging

from mmi.settings import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, initialising root handlers once."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=settings.log_level.upper(),
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # httpx logs the full request URL at INFO on every (even successful) call. Our FRED and
        # Gemini calls carry the API key as a query param (?api_key=… / ?key=…), so that INFO
        # line would leak the secret verbatim — redact() only covers exception strings, not this
        # happy path. Clamp httpx to WARNING so the per-request URL is never emitted. This holds
        # regardless of MMI_LOG_LEVEL, keeping the "secrets never leak" constraint intact.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
