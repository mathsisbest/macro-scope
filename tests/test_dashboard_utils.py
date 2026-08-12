"""Unit tests for the pure helpers in dashboard/components/utils.py.

The st.warning / st.caption rendering choice is not tested here (it needs a Streamlit
runtime) — the verdict_status_message() label + is_warning pair it feeds is.
"""

from __future__ import annotations

from dashboard.components.utils import verdict_status_message

_CLEARED = {
    "cleared": True,
    "reasons": [],
    "oos_r2": 0.2,
    "qlike_skill_ratio": 0.5,
    "folds_passed": 4,
    "n_folds": 5,
    "n_obs": 500,
}

_NOT_CLEARED = {
    "cleared": False,
    "reasons": [
        "oos_r2=0.0200 < R2_MIN=0.10 — model does not beat the persistence baseline out-of-sample",
        "qlike_skill_ratio=1.5000 >= 0.99 — model QLIKE does not meaningfully improve "
        "on baseline QLIKE",
    ],
    "oos_r2": 0.02,
    "qlike_skill_ratio": 1.5,
    "folds_passed": 4,
    "n_folds": 5,
    "n_obs": 500,
}


def test_not_cleared_verdict_renders_as_warning_with_baseline_only_label() -> None:
    label, is_warning = verdict_status_message(_NOT_CLEARED)
    assert is_warning is True
    assert "NOT CLEARED" in label
    assert "baseline-only" in label
    assert "no demonstrated out-of-sample edge" in label


def test_cleared_verdict_renders_as_caption_with_beaten_baseline_label() -> None:
    label, is_warning = verdict_status_message(_CLEARED)
    assert is_warning is False
    assert "CLEARED" in label
    assert "beats the persistence baseline" in label
    assert "baseline-only" not in label


def test_not_cleared_without_reasons_says_metrics_unavailable() -> None:
    verdict = dict(_NOT_CLEARED, reasons=[])
    label, is_warning = verdict_status_message(verdict)
    assert is_warning is True
    assert "metrics not yet available" in label


def test_cleared_label_carries_gate_numbers() -> None:
    label, is_warning = verdict_status_message(_CLEARED)
    assert is_warning is False
    assert "OOS R²=0.200" in label
    assert "QLIKE skill ratio=0.500" in label
    assert "4/5 folds passed" in label
