"""Tests for the pure skill_verdict() helper (Contract E, task C5).

All fixtures are synthetic long-format DataFrames — no DB, no network,
no model import.  The gate constants are tested against their documented
thresholds; they must never be weakened to make a test pass.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from mmi.ml.skill_gate import (
    N_OBS_MIN,
    QLIKE_MARGIN,
    R2_MIN,
    SUSTAIN_FRAC,
    skill_verdict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TRAINED_AT = "2025-01-01T00:00:00"


def _make_df(
    *,
    model: str = "rv_har",
    symbol: str = "SPY",
    oos_r2: float = 0.35,
    qlike_skill_ratio: float = 0.95,
    folds_passed: int = 4,
    n_folds: int = 5,
    n_obs: int = 500,
    trained_at: str = _BASE_TRAINED_AT,
) -> pd.DataFrame:
    """Return a minimal long-format model_metrics DataFrame that passes the gate."""
    rows = [
        {
            "model": model,
            "symbol": symbol,
            "metric": "oos_r2",
            "value": oos_r2,
            "trained_at": trained_at,
        },
        {
            "model": model,
            "symbol": symbol,
            "metric": "qlike_skill_ratio",
            "value": qlike_skill_ratio,
            "trained_at": trained_at,
        },
        {
            "model": model,
            "symbol": symbol,
            "metric": "folds_passed",
            "value": float(folds_passed),
            "trained_at": trained_at,
        },
        {
            "model": model,
            "symbol": symbol,
            "metric": "n_folds",
            "value": float(n_folds),
            "trained_at": trained_at,
        },
        {
            "model": model,
            "symbol": symbol,
            "metric": "n_obs",
            "value": float(n_obs),
            "trained_at": trained_at,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Clears-bar: all conditions satisfied
# ---------------------------------------------------------------------------


def test_clears_bar_all_conditions_met():
    df = _make_df(
        oos_r2=0.35,
        qlike_skill_ratio=0.95,
        folds_passed=4,
        n_folds=5,
        n_obs=600,
    )
    v = skill_verdict(df)
    assert v["cleared"] is True, f"Expected cleared=True, reasons={v['reasons']}"
    assert v["reasons"] == []
    assert v["oos_r2"] == pytest.approx(0.35)
    assert v["qlike_skill_ratio"] == pytest.approx(0.95)
    assert v["folds_passed"] == 4
    assert v["n_folds"] == 5
    assert v["n_obs"] == 600


def test_clears_bar_at_exact_thresholds():
    """Exactly at thresholds: oos_r2=R2_MIN, ratio=1-QLIKE_MARGIN-epsilon,
    folds_passed=ceil(SUSTAIN_FRAC * n_folds), n_obs=N_OBS_MIN."""
    n_folds = 5
    folds_needed = math.ceil(SUSTAIN_FRAC * n_folds)
    df = _make_df(
        oos_r2=R2_MIN,  # exactly at floor
        qlike_skill_ratio=1.0 - QLIKE_MARGIN - 1e-9,  # just below threshold
        folds_passed=folds_needed,
        n_folds=n_folds,
        n_obs=N_OBS_MIN,
    )
    v = skill_verdict(df)
    assert v["cleared"] is True, f"Expected cleared=True, reasons={v['reasons']}"


# ---------------------------------------------------------------------------
# 2. Fails on oos_r2 below R2_MIN
# ---------------------------------------------------------------------------


def test_fails_on_r2_below_min():
    df = _make_df(oos_r2=R2_MIN - 0.01)  # just below floor
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("oos_r2" in r for r in v["reasons"]), v["reasons"]
    assert v["oos_r2"] == pytest.approx(R2_MIN - 0.01)


def test_fails_on_r2_negative():
    """Negative R² (worse than mean baseline) must also fail."""
    df = _make_df(oos_r2=-0.05)
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("oos_r2" in r for r in v["reasons"])


# ---------------------------------------------------------------------------
# 3. Fails on qlike_skill_ratio >= 1 - QLIKE_MARGIN
# ---------------------------------------------------------------------------


def test_fails_on_qlike_ratio_at_threshold():
    """ratio == 1 - QLIKE_MARGIN (i.e. 0.99) must fail: condition is strictly <."""
    df = _make_df(qlike_skill_ratio=1.0 - QLIKE_MARGIN)  # 0.99 — not strictly less
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("qlike_skill_ratio" in r for r in v["reasons"]), v["reasons"]


def test_fails_on_qlike_ratio_above_threshold():
    """ratio > 0.99 means model is not improving QLIKE over baseline."""
    df = _make_df(qlike_skill_ratio=1.05)
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("qlike_skill_ratio" in r for r in v["reasons"])


# ---------------------------------------------------------------------------
# 4. Fails when not enough folds pass (overall edge but < ceil(0.6 * n_folds))
# ---------------------------------------------------------------------------


def test_fails_on_not_sustained_folds():
    """oos_r2 looks decent but skill is only in 2/5 folds — below SUSTAIN_FRAC."""
    n_folds = 5
    min_needed = math.ceil(SUSTAIN_FRAC * n_folds)  # 3
    df = _make_df(
        oos_r2=0.30,  # passes R2 gate individually
        qlike_skill_ratio=0.95,  # passes QLIKE gate individually
        folds_passed=min_needed - 1,  # 2 — one short of the 3 required
        n_folds=n_folds,
        n_obs=400,
    )
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("folds_passed" in r for r in v["reasons"]), v["reasons"]


def test_fails_on_not_sustained_exact_boundary():
    """folds_passed = ceil(0.6 * 5) - 1 = 2 must fail; =3 must pass."""
    n_folds = 5
    min_needed = math.ceil(SUSTAIN_FRAC * n_folds)  # 3

    # one below boundary: must fail
    df_fail = _make_df(folds_passed=min_needed - 1, n_folds=n_folds)
    assert skill_verdict(df_fail)["cleared"] is False

    # exactly at boundary: must pass (all other conditions good)
    df_pass = _make_df(folds_passed=min_needed, n_folds=n_folds)
    assert skill_verdict(df_pass)["cleared"] is True


# ---------------------------------------------------------------------------
# 5. Missing metric → not cleared, NO exception
# ---------------------------------------------------------------------------


def test_missing_metric_not_cleared_no_exception():
    """If one metric is absent the verdict must be False, never raise."""
    full_df = _make_df()
    # drop the oos_r2 row to simulate a partially-trained model
    partial_df = full_df[full_df["metric"] != "oos_r2"].copy()
    v = skill_verdict(partial_df)  # must not raise
    assert v["cleared"] is False
    assert v["reasons"], "expected at least one reason when metric is missing"
    assert "oos_r2" in v["reasons"][0]


def test_missing_multiple_metrics_not_cleared_no_exception():
    """Completely absent rv_har rows → cleared=False, no exception."""
    empty_df = pd.DataFrame(columns=["model", "symbol", "metric", "value", "trained_at"])
    v = skill_verdict(empty_df)
    assert v["cleared"] is False
    assert v["reasons"]


def test_wrong_model_name_no_exception():
    """Rows with a different model tag leave rv_har absent → not cleared."""
    df = _make_df(model="random_forest")  # not rv_har
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert "rv_har" in v["reasons"][0]


def test_wrong_symbol_no_exception():
    """Rows for a different symbol → not cleared for the requested symbol."""
    df = _make_df(symbol="QQQ")
    v = skill_verdict(df, symbol="SPY")
    assert v["cleared"] is False


# ---------------------------------------------------------------------------
# 6. Pure-noise row: oos_r2 ~ 0, ratio ~ 1 → MUST yield cleared=False
# ---------------------------------------------------------------------------


def test_pure_noise_must_fail():
    """A model indistinguishable from noise cannot clear the gate.

    This proves the bar cannot be cleared by luck / is not tuned to pass.
    oos_r2 ~ 0 (no predictive signal beyond mean) and qlike_skill_ratio ~ 1.0
    (model QLIKE ≈ baseline QLIKE) are the canonical 'pure noise' signatures.
    """
    noise_df = _make_df(
        oos_r2=0.001,  # near zero — barely above noise floor
        qlike_skill_ratio=0.999,  # model is marginally worse than baseline
        folds_passed=1,  # skill only in 1 of 5 folds
        n_folds=5,
        n_obs=300,
    )
    v = skill_verdict(noise_df)
    assert v["cleared"] is False, (
        "A noise-level model must NEVER clear the gate — "
        "check that threshold constants have not been weakened."
    )
    # All three primary conditions should fire
    assert any("oos_r2" in r for r in v["reasons"])
    assert any("qlike_skill_ratio" in r for r in v["reasons"])
    assert any("folds_passed" in r for r in v["reasons"])


def test_exactly_at_noise_r2_boundary():
    """oos_r2 exactly 0 must fail (< R2_MIN=0.10)."""
    df = _make_df(oos_r2=0.0)
    v = skill_verdict(df)
    assert v["cleared"] is False


# ---------------------------------------------------------------------------
# 7. n_obs below minimum
# ---------------------------------------------------------------------------


def test_fails_on_insufficient_n_obs():
    df = _make_df(n_obs=N_OBS_MIN - 1)
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("n_obs" in r for r in v["reasons"]), v["reasons"]


def test_passes_at_exact_n_obs_minimum():
    df = _make_df(n_obs=N_OBS_MIN)
    v = skill_verdict(df)
    assert v["cleared"] is True


# ---------------------------------------------------------------------------
# 8. Multiple rows per metric (de-dup uses latest trained_at)
# ---------------------------------------------------------------------------


def test_dedup_uses_latest_trained_at():
    """If the same metric appears twice, the most-recent trained_at wins."""
    old_row = {
        "model": "rv_har",
        "symbol": "SPY",
        "metric": "oos_r2",
        "value": -0.5,
        "trained_at": "2024-01-01T00:00:00",
    }
    new_row = {
        "model": "rv_har",
        "symbol": "SPY",
        "metric": "oos_r2",
        "value": 0.35,
        "trained_at": "2025-06-01T00:00:00",
    }

    base = _make_df()
    base_no_r2 = base[base["metric"] != "oos_r2"]
    df = pd.concat([base_no_r2, pd.DataFrame([old_row, new_row])], ignore_index=True)
    v = skill_verdict(df)
    # newer row (0.35) should win → gate should clear (all other conditions good)
    assert v["cleared"] is True, f"Expected new oos_r2=0.35 to win, reasons={v['reasons']}"
    assert v["oos_r2"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# 9. DataFrame with missing required columns → not cleared, no exception
# ---------------------------------------------------------------------------


def test_missing_required_column_no_exception():
    """A DataFrame lacking the 'metric' column must return cleared=False."""
    bad_df = pd.DataFrame([{"model": "rv_har", "symbol": "SPY", "value": 0.5}])
    v = skill_verdict(bad_df)  # must not raise
    assert v["cleared"] is False
    assert "metric" in v["reasons"][0]


# ---------------------------------------------------------------------------
# 10. Module constants are sane (regression guard)
# ---------------------------------------------------------------------------


def test_module_constants_are_sane():
    """Sanity-check the fixed threshold values from Contract E."""
    assert pytest.approx(0.10) == R2_MIN
    assert pytest.approx(0.01) == QLIKE_MARGIN
    assert pytest.approx(0.6) == SUSTAIN_FRAC
    assert N_OBS_MIN == 250


# ---------------------------------------------------------------------------
# 11. Non-finite metric values (NaN / ±inf) must FAIL CLOSED — never silent pass
# ---------------------------------------------------------------------------

_NAN = float("nan")
_INF = float("inf")


def test_nan_oos_r2_not_cleared():
    """NaN oos_r2 must NOT clear: ``nan < R2_MIN`` is False, so the naive check
    would silently pass it. The honesty gate must treat NaN like a missing metric."""
    df = _make_df(oos_r2=_NAN)
    v = skill_verdict(df)
    assert v["cleared"] is False, "NaN oos_r2 must never clear the honesty gate"
    assert any("oos_r2" in r for r in v["reasons"]), v["reasons"]


def test_nan_oos_r2_with_other_metrics_passing_not_cleared():
    """The exact false-pass scenario from the bug report: a degenerate run writes
    NaN oos_r2 while folds_passed / n_obs / qlike happen to satisfy their gates.
    It must STILL fail closed rather than return cleared=True."""
    df = _make_df(
        oos_r2=_NAN,
        qlike_skill_ratio=0.90,  # would pass on its own
        folds_passed=5,  # would pass on its own
        n_folds=5,
        n_obs=600,  # would pass on its own
    )
    v = skill_verdict(df)
    assert v["cleared"] is False


def test_nan_qlike_skill_ratio_not_cleared():
    """NaN qlike_skill_ratio must NOT clear: ``nan >= 0.99`` is False."""
    df = _make_df(qlike_skill_ratio=_NAN)
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("qlike_skill_ratio" in r for r in v["reasons"]), v["reasons"]


def test_inf_oos_r2_not_cleared():
    """+inf oos_r2 is non-finite garbage and must not clear (``inf >= R2_MIN`` is True,
    so without a finiteness guard it would silently pass)."""
    df = _make_df(oos_r2=_INF)
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("oos_r2" in r for r in v["reasons"]), v["reasons"]


def test_nan_folds_passed_not_cleared_no_exception():
    """NaN folds_passed must not raise (``int(float('nan'))`` raises ValueError)
    and must not clear."""
    df = _make_df(folds_passed=_NAN)
    v = skill_verdict(df)  # must not raise
    assert v["cleared"] is False
    assert any("folds_passed" in r for r in v["reasons"]), v["reasons"]


def test_inf_n_obs_not_cleared_no_exception():
    """±inf n_obs must not raise and must not clear."""
    df = _make_df(n_obs=_INF)
    v = skill_verdict(df)  # must not raise
    assert v["cleared"] is False
    assert any("n_obs" in r for r in v["reasons"]), v["reasons"]


def test_nan_n_folds_not_cleared_no_exception():
    """NaN n_folds must not raise (it feeds ``math.ceil`` / ``int``) and must not clear."""
    df = _make_df(n_folds=_NAN)
    v = skill_verdict(df)  # must not raise
    assert v["cleared"] is False
    assert any("n_folds" in r for r in v["reasons"]), v["reasons"]


def test_nan_int_metric_in_partial_row_branch_no_exception():
    """Gap #2 path: a NaN int metric while another metric is entirely absent must
    not raise. The pre-fix code computed ``int(metric_map['folds_passed'])`` while
    building the missing-metric return value, raising ValueError on NaN. It must
    now fail closed without an exception."""
    full = _make_df(folds_passed=_NAN)
    partial = full[full["metric"] != "oos_r2"].copy()  # drop oos_r2 entirely
    v = skill_verdict(partial)  # must not raise
    assert v["cleared"] is False
    assert v["reasons"]


# ---------------------------------------------------------------------------
# Out-of-range (finite-but-impossible) metrics must ALSO fail closed.
# These are finite, so they pass the NaN/inf guard — but each would otherwise
# slip through a threshold comparison and FALSELY clear the gate.
# ---------------------------------------------------------------------------


def test_impossible_oos_r2_above_one_not_cleared():
    """A finite oos_r2 > 1.0 is impossible (R² ≤ 1); it must not satisfy `>= R2_MIN`."""
    df = _make_df(oos_r2=1e308)  # all other metrics pass
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("oos_r2" in r and "1.0" in r for r in v["reasons"]), v["reasons"]


def test_zero_n_folds_not_cleared():
    """n_folds=0 makes ceil(0.6*0)=0, so `folds_passed < 0` is vacuously False —
    the folds check would pass. A zero-fold run must fail closed instead."""
    df = _make_df(n_folds=0, folds_passed=0)  # other metrics pass
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("n_folds" in r for r in v["reasons"]), v["reasons"]


def test_negative_qlike_ratio_not_cleared():
    """A negative qlike_skill_ratio is impossible (ratio of non-negative losses)
    and would satisfy `< 0.99`; it must be rejected, not treated as great skill."""
    df = _make_df(qlike_skill_ratio=-0.5)  # other metrics pass
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("qlike_skill_ratio" in r for r in v["reasons"]), v["reasons"]


def test_folds_passed_exceeds_n_folds_not_cleared():
    """folds_passed > n_folds is impossible and must fail closed."""
    df = _make_df(folds_passed=9, n_folds=5)  # other metrics pass
    v = skill_verdict(df)
    assert v["cleared"] is False
    assert any("folds_passed" in r for r in v["reasons"]), v["reasons"]


def test_negative_n_obs_not_cleared():
    """A negative n_obs is garbage; it must fail closed (and never clear)."""
    df = _make_df(n_obs=-100)
    v = skill_verdict(df)
    assert v["cleared"] is False


def test_in_range_boundary_values_still_clear():
    """Sanity: legitimate in-range values at the edges (oos_r2=1.0, qlike=0.0,
    folds_passed=n_folds) must STILL clear — the domain guard must not over-reject."""
    df = _make_df(oos_r2=1.0, qlike_skill_ratio=0.0, folds_passed=5, n_folds=5, n_obs=500)
    v = skill_verdict(df)
    assert v["cleared"] is True, v["reasons"]
