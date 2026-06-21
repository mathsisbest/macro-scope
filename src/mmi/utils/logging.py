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
        _CONFIGURED = True
    return logging.getLogger(name)
