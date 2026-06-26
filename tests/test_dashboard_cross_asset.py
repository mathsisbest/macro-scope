"""Cross-asset Markets-view compute helpers + the ML-tab holdout readout.

The smoke test only proves the app renders without raising; the *correctness* of the derived
cross-asset frames (leaderboard return/vol, rebased-to-0% performance, the correlation matrix +
its <30-obs guard) and the holdout readout is covered here on fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from dashboard.components import charts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _long(rows: list[tuple]) -> pd.DataFrame:
    """Build the [symbol, asset_class, date, close, daily_return] long frame all helpers take."""
    return pd.DataFrame(rows, columns=["symbol", "asset_class", "date", "close", "daily_return"])


def _series(symbol: str, klass: str, closes: list[float], start="2024-01-01") -> list[tuple]:
    """A symbol's daily rows with close-to-close simple returns (first return NaN, as the mart)."""
    dates = pd.bdate_range(start=start, periods=len(closes))
    rows: list[tuple] = []
    prev: float | None = None
    for d, c in zip(dates, closes, strict=True):
        ret = (c / prev - 1.0) if prev not in (None, 0) else float("nan")
        rows.append((symbol, klass, d.date(), float(c), ret))
        prev = c
    return rows


# ---------------------------------------------------------------------------
# (a) Leaderboard: return = close-ratio − 1, vol = std · √252
# ---------------------------------------------------------------------------
def test_leaderboard_return_is_close_ratio_minus_one_and_vol_is_std_sqrt252():
    # SPY: 100 → 110 → 121  (return = 121/100 − 1 = +21%)
    long_df = _long(_series("SPY", "equities", [100.0, 110.0, 121.0]))
    board = charts.cross_asset_leaderboard(long_df)

    assert list(board["symbol"]) == ["SPY"]
    row = board.iloc[0]
    assert math.isclose(row["period_return"], 121.0 / 100.0 - 1.0, rel_tol=1e-12)

    rets = long_df["daily_return"]  # the in-frame returns, incl. the leading NaN
    expected_vol = float(rets.std(ddof=1) * math.sqrt(252))
    assert math.isclose(row["ann_vol"], expected_vol, rel_tol=1e-12)
    assert row["asset_class"] == "equities"


def test_leaderboard_sorted_by_return_desc():
    long_df = _long(
        _series("DOWN", "bonds", [100.0, 90.0])  # −10%
        + _series("UP", "equities", [100.0, 130.0])  # +30%
        + _series("FLAT", "fx", [100.0, 100.0])  # 0%
    )
    board = charts.cross_asset_leaderboard(long_df)
    assert list(board["symbol"]) == ["UP", "FLAT", "DOWN"]  # descending by period_return


def test_leaderboard_empty_frame_returns_empty():
    board = charts.cross_asset_leaderboard(
        pd.DataFrame(columns=["symbol", "asset_class", "date", "close", "daily_return"])
    )
    assert board.empty
    assert list(board.columns) == ["symbol", "asset_class", "period_return", "ann_vol"]


def test_leaderboard_return_color_is_green_up_red_down():
    assert charts.leaderboard_return_color(0.05) == charts.PALETTE["up"]
    assert charts.leaderboard_return_color(-0.05) == charts.PALETTE["down"]
    assert charts.leaderboard_return_color(0.0) == charts.PALETTE["up"]  # 0 reads as non-negative


# ---------------------------------------------------------------------------
# (b) Rebased performance starts at exactly 0% for each symbol
# ---------------------------------------------------------------------------
def test_rebased_performance_starts_at_zero_for_every_symbol():
    long_df = _long(
        _series("SPY", "equities", [100.0, 110.0, 121.0])
        + _series("TLT", "bonds", [50.0, 48.0, 49.0])
    )
    perf = charts.rebased_performance(long_df)
    for symbol, grp in perf.groupby("symbol"):
        g = grp.sort_values("date")
        assert g["perf"].iloc[0] == 0.0, f"{symbol} must start at exactly 0%"

    # Final compounded value matches the simple close ratio for SPY: 121/100 − 1 = +21%.
    spy_final = perf[perf["symbol"] == "SPY"].sort_values("date")["perf"].iloc[-1]
    assert math.isclose(spy_final, 0.21, rel_tol=1e-9)


def test_rebased_performance_empty_frame_returns_empty():
    perf = charts.rebased_performance(
        pd.DataFrame(columns=["symbol", "asset_class", "date", "close", "daily_return"])
    )
    assert perf.empty
    assert list(perf.columns) == ["symbol", "asset_class", "date", "perf"]


# ---------------------------------------------------------------------------
# (c) Correlation matrix on a fixture + the <30-obs guard
# ---------------------------------------------------------------------------
def test_correlation_matrix_recovers_perfect_and_anti_correlation():
    # 40 obs (> the 30-obs guard). B mirrors A exactly (+1), C is A negated (−1).
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 40)
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows: list[tuple] = []
    for d, a in zip(dates, base, strict=True):
        rows.append(("A", "equities", d.date(), 100.0, float(a)))
        rows.append(("B", "equities", d.date(), 100.0, float(a)))  # identical → ρ=+1
        rows.append(("C", "bonds", d.date(), 100.0, float(-a)))  # negated → ρ=−1
    corr = charts.correlation_matrix(_long(rows))
    assert corr is not None
    assert set(corr.columns) == {"A", "B", "C"}
    assert math.isclose(corr.loc["A", "B"], 1.0, abs_tol=1e-9)
    assert math.isclose(corr.loc["A", "C"], -1.0, abs_tol=1e-9)


def test_correlation_matrix_guard_returns_none_when_too_few_obs():
    # Only 10 overlapping observations (< 30) → the guard signals "too short" (None), not a matrix.
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=10)
    rows: list[tuple] = []
    for d in dates:
        rows.append(("A", "equities", d.date(), 100.0, float(rng.normal())))
        rows.append(("B", "bonds", d.date(), 100.0, float(rng.normal())))
    assert charts.correlation_matrix(_long(rows)) is None
    # The app shows this exact caption in the None branch.
    assert "widen the date range" in charts.CORR_TOO_SHORT


def test_correlation_matrix_none_for_single_asset():
    long_df = _long(_series("SPY", "equities", [100.0] * 40))
    assert charts.correlation_matrix(long_df) is None  # nothing to correlate against


def test_correlation_takeaway_names_the_extreme_pairs():
    corr = pd.DataFrame(
        [[1.0, 0.9, -0.4], [0.9, 1.0, -0.5], [-0.4, -0.5, 1.0]],
        index=["SPY", "QQQ", "TLT"],
        columns=["SPY", "QQQ", "TLT"],
    )
    takeaway = charts.correlation_takeaway(corr)
    assert "SPY–QQQ" in takeaway  # most-correlated off-diagonal pair
    assert "QQQ–TLT" in takeaway  # best diversifier (most negative)


# ---------------------------------------------------------------------------
# (d) ML-tab holdout readout: reads holdout_* from a fixture; None when absent
# ---------------------------------------------------------------------------
def _metrics(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "symbol", "metric", "value", "trained_at"])


def test_holdout_readout_reads_vol_holdout_oos_r2():
    metrics = _metrics(
        [
            ("rv_har", "SPY", "oos_r2", 0.12, "2026-06-20"),  # CV metric — not holdout
            ("rv_har", "SPY", "holdout_oos_r2", 0.085, "2026-06-20"),
            ("rv_har", "SPY", "holdout_qlike_skill_ratio", 0.97, "2026-06-20"),
            ("rv_har", "SPY", "holdout_n_obs", 252.0, "2026-06-20"),
        ]
    )
    out = charts.holdout_readout(metrics, model="rv_har", symbol="SPY")
    assert out is not None
    assert math.isclose(out["holdout_oos_r2"], 0.085)
    assert math.isclose(out["holdout_qlike_skill_ratio"], 0.97)
    assert out["holdout_n_obs"] == 252.0
    assert "oos_r2" not in out  # the non-holdout CV metric is excluded


def test_holdout_readout_reads_direction_holdout_via_exclude_model():
    metrics = _metrics(
        [
            ("rv_har", "SPY", "holdout_oos_r2", 0.05, "2026-06-20"),
            ("random_forest", "SPY", "holdout_dir_acc", 0.54, "2026-06-20"),
            ("random_forest", "SPY", "holdout_baseline_dir_acc", 0.52, "2026-06-20"),
            ("random_forest", "SPY", "holdout_n_obs", 252.0, "2026-06-20"),
        ]
    )
    out = charts.holdout_readout(metrics, symbol="SPY", exclude_model="rv_har")
    assert out is not None
    assert math.isclose(out["holdout_dir_acc"], 0.54)
    assert math.isclose(out["holdout_baseline_dir_acc"], 0.52)
    assert "holdout_oos_r2" not in out  # the vol model's holdout is excluded


def test_holdout_readout_none_when_no_holdout_rows_present():
    # Only CV metrics (small-data skip / pre-re-run snapshot) → render nothing.
    metrics = _metrics(
        [
            ("rv_har", "SPY", "oos_r2", 0.12, "2026-06-20"),
            ("rv_har", "SPY", "qlike_skill_ratio", 0.98, "2026-06-20"),
        ]
    )
    assert charts.holdout_readout(metrics, model="rv_har", symbol="SPY") is None


def test_holdout_readout_none_on_empty_or_missing_columns():
    assert charts.holdout_readout(pd.DataFrame(), model="rv_har") is None
    assert charts.holdout_readout(pd.DataFrame({"value": [0.1]}), model="rv_har") is None


def test_holdout_readout_skips_nan_and_inf_values():
    metrics = _metrics(
        [
            ("rv_har", "SPY", "holdout_oos_r2", float("nan"), "2026-06-20"),
            ("rv_har", "SPY", "holdout_qlike_skill_ratio", float("inf"), "2026-06-20"),
            ("rv_har", "SPY", "holdout_n_obs", 200.0, "2026-06-20"),
        ]
    )
    out = charts.holdout_readout(metrics, model="rv_har", symbol="SPY")
    assert out == {"holdout_n_obs": 200.0}  # the NaN / inf rows are dropped, a finite one survives
