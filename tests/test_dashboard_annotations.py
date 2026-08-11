"""P3-3.4 chart annotations: regime boundaries, OOS-evaluation milestones, VIX spikes.

Pure helpers + annotation presence in figures + empty/no-trigger degradation.
"""

import math

import pandas as pd
import pytest
from dashboard.components import charts

# ---------------------------------------------------------------------------
# regime_boundary_dates — pure helper
# ---------------------------------------------------------------------------


def _regime_df(regimes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=len(regimes)),
            "regime": regimes,
        }
    )


def test_regime_boundaries_only_at_observed_changes():
    b = charts.regime_boundary_dates(
        _regime_df(["Low", "Low", "Medium", "Medium", "High", "High", "Medium", "Medium"])
    )
    assert b["regime"].tolist() == ["Medium", "High", "Medium"]
    assert b["date"].tolist() == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-09"),
    ]


def test_regime_boundaries_first_row_is_not_a_boundary():
    # The prior regime is unknown before the frame's first row — a boundary must never be
    # invented there (a windowed frame that starts mid-regime would otherwise draw a fake line).
    assert charts.regime_boundary_dates(_regime_df(["Low", "Low", "Low"])).empty


def test_regime_boundaries_handle_unsorted_input_and_duplicate_dates():
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
            ],
            "regime": ["High", "Low", "Medium", "Medium"],
        }
    )
    b = charts.regime_boundary_dates(df)
    assert b["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert b["regime"].tolist() == ["Medium", "High"]


def test_regime_boundaries_clip_after_detection():
    df = _regime_df(["Low", "Low", "Medium", "Medium", "High"])
    # start lands exactly ON a boundary — it must survive clipping (detected on the full
    # series first), while boundaries before start and the first row are dropped.
    b = charts.regime_boundary_dates(df, start="2024-01-03", end="2024-01-05")
    assert b["regime"].tolist() == ["Medium", "High"]


def test_regime_boundaries_min_gap_thins_clusters_greedily():
    # Five boundaries in six days (a realistic flip-flop cluster): min_gap keeps only the
    # earliest, then greedily the next ≥ min_gap days after each kept one.
    b = charts.regime_boundary_dates(
        _regime_df(["Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium"]),
        min_gap_days=10,
    )
    assert b["regime"].tolist() == ["Medium"]  # 2024-01-02; next boundary is only +1d away
    assert b["date"].tolist() == [pd.Timestamp("2024-01-02")]
    # A wider span lets later boundaries through again (weekend dates skipped by bdate_range).
    wide = charts.regime_boundary_dates(
        _regime_df(["Low", "Medium", "High", "Low", "Medium", "High", "Low", "Medium"]),
        min_gap_days=4,
    )
    assert wide["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-08"),
    ]
    # min_gap_days=None / 0 keeps every boundary (raw detector).
    assert len(charts.regime_boundary_dates(_regime_df(["Low", "Medium", "High"]))) == 2
    all_zero = charts.regime_boundary_dates(_regime_df(["Low", "Medium", "High"]), min_gap_days=0)
    assert len(all_zero) == 2


def test_regime_boundaries_degrade_on_bad_input():
    assert charts.regime_boundary_dates(None).empty
    assert charts.regime_boundary_dates(pd.DataFrame()).empty
    assert charts.regime_boundary_dates(pd.DataFrame({"date": []})).empty
    assert charts.regime_boundary_dates(_regime_df(["Low", "High"]).assign(regime=None)).empty
    b = charts.regime_boundary_dates(_regime_df(["Low", "High"]).drop(columns=["regime"]))
    assert list(b.columns) == ["date", "regime"] and b.empty


# ---------------------------------------------------------------------------
# vix_spike_dates — pure helper
# ---------------------------------------------------------------------------


def _vix_df(periods: int = 60, base: float = 15.0) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=periods), "value": base})


def test_vix_spikes_are_full_history_zscore_days_above_threshold():
    vix = _vix_df()
    vix.loc[10, "value"] = 60.0
    vix.loc[11, "value"] = 55.0
    vix.loc[40, "value"] = 35.0
    spikes = charts.vix_spike_dates(vix)
    assert spikes["date"].tolist() == [
        pd.Timestamp("2020-01-15"),
        pd.Timestamp("2020-01-16"),
        pd.Timestamp("2020-02-26"),
    ]
    # The most extreme day (60.0) carries the highest z-score.
    assert float(spikes["value"].iloc[0]) == 60.0
    assert float(spikes["zscore"].iloc[0]) > float(spikes["zscore"].iloc[2]) > charts.VIX_SPIKE_Z
    assert list(spikes.columns) == ["date", "value", "zscore"]


def test_vix_spikes_clip_then_top_n():
    vix = _vix_df()
    vix.loc[10, "value"] = 60.0
    vix.loc[11, "value"] = 55.0
    vix.loc[40, "value"] = 35.0
    # Within the clip window only the 35.0 day qualifies → top_n=1 keeps exactly it.
    spikes = charts.vix_spike_dates(vix, start="2020-02-01", end="2020-02-29", top_n=1)
    assert spikes["date"].tolist() == [pd.Timestamp("2020-02-26")]
    assert float(spikes["zscore"].iloc[0]) == pytest.approx(2.2618, abs=1e-3)
    # top_n=None keeps every spike day.
    assert len(charts.vix_spike_dates(vix, top_n=None)) == 3


def test_vix_spikes_degrade_to_empty():
    # No day above threshold.
    assert charts.vix_spike_dates(_vix_df()).empty
    # Fewer than the minimum reliable observations.
    assert charts.vix_spike_dates(_vix_df(periods=10)).empty
    # Zero-variance series (std == 0) → no z-score, no spikes.
    assert charts.vix_spike_dates(_vix_df(periods=40)).empty
    # Malformed / empty input → empty with the documented columns, never a crash.
    assert charts.vix_spike_dates(None).empty
    assert charts.vix_spike_dates(pd.DataFrame()).empty
    assert charts.vix_spike_dates(pd.DataFrame({"date": []})).empty
    small = charts.vix_spike_dates(
        pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5), "value": [10.0] * 5})
    )
    assert list(small.columns) == ["date", "value", "zscore"] and small.empty
    missing_col = charts.vix_spike_dates(
        pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5)})
    )
    assert list(missing_col.columns) == ["date", "value", "zscore"] and missing_col.empty


def test_vix_spikes_tolerate_non_numeric_and_nan_values():
    vix = pd.DataFrame(
        {"date": pd.bdate_range("2020-01-01", periods=60), "value": [15.0] * 60}
    ).astype({"value": object})  # object dtype so non-numeric junk can be injected
    vix.loc[10, "value"] = "not-a-number"  # coerced away, not a crash
    vix.loc[11, "value"] = float("nan")
    vix.loc[40, "value"] = 35.0
    spikes = charts.vix_spike_dates(vix)
    assert spikes["date"].tolist() == [pd.Timestamp("2020-02-26")]


# ---------------------------------------------------------------------------
# oos_count_label — pure helper
# ---------------------------------------------------------------------------


def test_oos_count_label_formats_finite_counts():
    assert charts.oos_count_label(3298) == "OOS n=3,298"
    assert charts.oos_count_label(3298.0) == "OOS n=3,298"
    assert charts.oos_count_label(252) == "OOS n=252"


def test_oos_count_label_none_on_missing_or_non_finite():
    # 0 means "not evaluated out-of-sample" — never a count label.
    for bad in (None, float("nan"), float("inf"), float("-inf"), -3, 0, 0.0, "n/a"):
        assert charts.oos_count_label(bad) is None


# ---------------------------------------------------------------------------
# portfolio_cumulative_chart — regime-boundary annotations in the figure
# ---------------------------------------------------------------------------


def _pf_df() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6).tolist()
    return pd.DataFrame(
        {
            "strategy": ["equal_weight"] * 6 + ["sixty_forty"] * 6,
            "date": dates * 2,
            "cumulative_return": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] * 2,
            "daily_return": [0.01] * 12,
            "drawdown": [0.0] * 12,
            "rolling_sharpe_252": [1.0] * 12,
        }
    )


def test_portfolio_cumulative_chart_draws_boundary_lines_with_regime_labels():
    regime = _regime_df(["Low", "Low", "Medium", "Medium", "High", "High", "Medium"])
    fig = charts.portfolio_cumulative_chart(_pf_df(), regime_df=regime, min_gap_days=None)
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(vlines) == 2  # Medium (2024-01-03) and High (2024-01-05); the first row is not one
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["Medium", "High"]
    # Named-token styling only: muted dashed lines, regime token text colours.
    assert all(s.line.color == charts.PALETTE["muted"] and s.line.dash == "dash" for s in vlines)
    text_colors = {a.font.color for a in fig.layout.annotations}
    assert text_colors <= {charts.PALETTE["up"], charts.SERIES_VOL, charts.PALETTE["down"]}
    # Pinned per-regime colour mapping (Low→up, Medium→vol, High→down) so a swapped map fails.
    by_text = {a.text: a.font.color for a in fig.layout.annotations}
    assert by_text == {"Medium": charts.SERIES_VOL, "High": charts.PALETTE["down"]}


def test_portfolio_cumulative_chart_boundary_labels_pin_y_position():
    # Labels sit at the top of the plot area (paper y=1.0, bottom-anchored) — never on the
    # line itself and never at different heights.
    regime = _regime_df(["Low", "Low", "Medium", "Medium", "High", "High", "Low"])
    fig = charts.portfolio_cumulative_chart(_pf_df(), regime_df=regime, min_gap_days=None)
    assert len(fig.layout.annotations) == 2
    for a in fig.layout.annotations:
        assert a.y == 1.0 and a.yref == "paper" and a.yanchor == "bottom" and not a.showarrow
        assert a.x == pd.Timestamp("2024-01-03") or a.x == pd.Timestamp("2024-01-05")


def test_portfolio_cumulative_chart_default_min_gap_thins_dense_boundaries():
    # The default chart must NOT draw one line per flip-flop: a realistic dense series
    # (boundaries every few days) collapses to ~1 label per year, earliest-first.
    regimes = ["Low", "Medium", "High"] * 130  # 390 rows, boundaries every 2 days
    regime = pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=len(regimes)), "regime": regimes}
    )
    fig = charts.portfolio_cumulative_chart(_pf_df(), regime_df=regime)
    labels = [a.text for a in fig.layout.annotations]
    assert labels  # still some context…
    # …but the earliest boundary only — the rest are < 365d after it.
    assert len(labels) == 1
    assert len([s for s in fig.layout.shapes if s.type == "line"]) == 1


def test_portfolio_cumulative_chart_clips_boundaries_to_chart_range():
    # Regime series runs continuously into and past the portfolio frame: the boundary AT the
    # window start (detected from the full series) is kept, the one AFTER the window end is not.
    regime = pd.DataFrame(
        {
            "date": list(pd.bdate_range("2023-12-25", periods=5))
            + list(pd.bdate_range("2024-01-01", periods=7)),  # Jan 1..9
            "regime": ["High"] * 5 + ["Medium"] * 6 + ["Low"],
        }
    )
    fig = charts.portfolio_cumulative_chart(_pf_df(), regime_df=regime)
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    # Only the High→Medium boundary (2024-01-01, the window's first date) is in range.
    assert len(vlines) == 1
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["Medium"]


def test_portfolio_cumulative_chart_degrades_without_regime_data():
    fig_plain = charts.portfolio_cumulative_chart(_pf_df())
    assert not [s for s in fig_plain.layout.shapes if s.type == "line"]
    assert not fig_plain.layout.annotations
    # A single-regime frame has no observable boundary either.
    fig_single = charts.portfolio_cumulative_chart(_pf_df(), regime_df=_regime_df(["Low"] * 7))
    assert not [s for s in fig_single.layout.shapes if s.type == "line"]
    assert not fig_single.layout.annotations
    # A regime frame with no overlap with the chart range → no markers.
    off_range = _regime_df(["Low", "High", "Low"])  # Jan 1..3, 2024 — pf starts Jan 1 too…
    off_range["date"] = pd.bdate_range("2023-01-01", periods=3)
    fig_off = charts.portfolio_cumulative_chart(_pf_df(), regime_df=off_range)
    assert not [s for s in fig_off.layout.shapes if s.type == "line"]


# ---------------------------------------------------------------------------
# return_performance_chart — skill-gate milestone annotations
# ---------------------------------------------------------------------------


def _perf_df(with_n_obs: bool = True) -> pd.DataFrame:
    d = {
        "symbol": ["SPY", "BTC"],
        "ic": [0.02, -0.01],
        "direction_accuracy": [0.55, 0.52],
        "baseline_direction_accuracy": [0.50, 0.50],
        "direction_edge": [0.05, 0.02],
        "positive_prediction_rate": [0.60, 0.58],
        "sharpe": [1.0, 0.8],
        "r2": [0.01, -0.005],
    }
    if with_n_obs:
        d["n_obs"] = [3298, 4588]
    return pd.DataFrame(d)


def test_return_performance_chart_annotates_oos_counts_and_gate_lines():
    fig = charts.return_performance_chart(_perf_df())
    r2_bar = [t for t in fig.data if t.name == "R²"][0]
    assert list(r2_bar.text) == ["OOS n=3,298", "OOS n=4,588"]
    assert r2_bar.textposition == "outside"
    # Both skill-gate reference lines (return_forecast_skill_verdict criteria) are drawn:
    # R² > 0 on the y-axis, dir-acc > 50% on the y2 axis.
    gate_lines = [(float(s.y0), s.yref, s.line.color, s.line.dash) for s in fig.layout.shapes]
    assert (0.0, "y", charts.PALETTE["up"], "dot") in gate_lines
    assert (0.5, "y2", charts.PALETTE["up"], "dot") in gate_lines
    texts = [a.text for a in fig.layout.annotations]
    assert "skill gate: R² > 0" in texts
    assert "dir-acc gate: > 50%" in texts


def test_return_performance_chart_skips_oos_labels_without_n_obs():
    fig = charts.return_performance_chart(_perf_df(with_n_obs=False))
    r2_bar = [t for t in fig.data if t.name == "R²"][0]
    assert r2_bar.text is None
    assert any(a.text == "skill gate: R² > 0" for a in fig.layout.annotations)


def test_return_performance_chart_empty_frame_no_crash():
    fig = charts.return_performance_chart(pd.DataFrame())
    assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# macro_chart — VIX spike annotations in the figure
# ---------------------------------------------------------------------------


def _macro_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.bdate_range("2024-01-01", periods=5), "value": [15.0, 15.5, 15.2, 15.8, 15.1]}
    )


def _spike_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-05"]),
            "value": [40.0, 35.0],
            "zscore": [4.2, 3.1],
        }
    )


def test_macro_chart_draws_vix_spike_lines_with_factual_z_labels():
    fig = charts.macro_chart(_macro_df(), "VIX", "index", spikes=_spike_frame())
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(vlines) == 2
    assert all(s.line.color == charts.PALETTE["muted"] and s.line.dash == "dash" for s in vlines)
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["VIX z=4.2", "VIX z=3.1"]
    assert all(a.font.color == charts.PALETTE["down"] for a in fig.layout.annotations)


def test_macro_chart_degrades_without_spikes():
    fig = charts.macro_chart(_macro_df(), "VIX", "index")
    assert not fig.layout.shapes and not fig.layout.annotations
    # Explicit empty frame → no crash, no annotations.
    fig_empty = charts.macro_chart(_macro_df(), "VIX", "index", spikes=pd.DataFrame())
    assert not fig_empty.layout.shapes and not fig_empty.layout.annotations
    # A spike frame missing its columns is ignored, not trusted.
    fig_bad = charts.macro_chart(
        _macro_df(), "VIX", "index", spikes=pd.DataFrame({"date": ["2024-01-03"]})
    )
    assert not fig_bad.layout.shapes


# ---------------------------------------------------------------------------
# Annotation labels never editorialise — they quote the data values
# ---------------------------------------------------------------------------


def test_annotation_text_is_factual_not_editorial():
    # VIX labels report the computed z-score (no commentary words like "crash" or "fear");
    # regime labels report the regime name. Documented defaults must stay stable.
    fig = charts.macro_chart(_macro_df(), "VIX", "index", spikes=_spike_frame())
    for a in fig.layout.annotations:
        assert a.text.startswith("VIX z=")
    assert charts.VIX_SPIKE_Z == 2.0 and charts.VIX_SPIKE_TOP_N == 8
    assert math.isfinite(charts.VIX_SPIKE_Z)
