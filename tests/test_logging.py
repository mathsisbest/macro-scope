"""get_logger configures root once; httpx is clamped to WARNING regardless of log level."""

import logging

from mmi.utils.logging import get_logger


def _reset():
    import mmi.utils.logging as m
    object.__setattr__(m, "_CONFIGURED", False)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)


def test_get_logger_returns_logger_with_given_name():
    _reset()
    log = get_logger("test_mod")
    assert log.name == "test_mod"
    assert isinstance(log, logging.Logger)


def test_get_logger_clamps_httpx_to_warning():
    _reset()
    get_logger("x")
    assert logging.getLogger("httpx").level == logging.WARNING


def test_get_logger_respects_log_level(monkeypatch):
    _reset()
    monkeypatch.setattr("mmi.utils.logging.settings.log_level", "ERROR")
    get_logger("x")
    assert logging.getLogger().level == logging.ERROR


def test_get_logger_only_configures_once():
    _reset()
    get_logger("first")
    root = logging.getLogger()
    assert root.hasHandlers()
    n_handlers = len(root.handlers)
    get_logger("second")
    assert len(root.handlers) == n_handlers
