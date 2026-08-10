"""The bootstrap scorecard/verdict builders summarise the pairwise result honestly."""

import math

import pandas as pd
from dashboard.components import charts

_COLS = ["strategy_a", "strategy_b", "sharpe_diff", "diff_lo", "diff_hi", "distinguishable"]


def _pairs(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLS)


def test_verdict_when_nothing_is_distinguishable():
    pairs = _pairs(
        [
            ["equal_weight", "risk_parity", 0.0, -0.1, 0.1, False],
            ["equal_weight", "inverse_vol", -0.05, -0.2, 0.1, False],
        ]
    )
    verdict = charts.distinguishability_verdict(pairs)
    assert "None" in verdict and "within noise" in verdict


def test_verdict_lists_only_distinguishable_pairs_with_labels():
    pairs = _pairs(
        [
            ["equal_weight", "sixty_forty", -3.2, -5.2, -1.3, True],
            ["inverse_vol", "risk_parity", 0.06, -0.08, 0.18, False],
        ]
    )
    verdict = charts.distinguishability_verdict(pairs)
    assert "1 of 2" in verdict
    assert "Equal weight vs 60/40 benchmark" in verdict  # labelled, not raw keys
    assert "Risk parity" not in verdict  # the indistinguishable pair is omitted


def test_verdict_handles_empty():
    assert "Not enough" in charts.distinguishability_verdict(_pairs([]))


# --- annualised-return significance: the "is the return gap real?" surface -------------------

_RETURN_COLS = [
    "strategy_a",
    "strategy_b",
    "ann_return_a",
    "ann_return_b",
    "ann_return_diff",
    "diff_lo",
    "diff_hi",
    "p_value",
    "distinguishable",
]


def _return_pairs(rows: list) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_RETURN_COLS)


def test_return_pairs_table_labels_and_formats_columns():
    table = charts.portfolio_return_pairs_table(
        _return_pairs(
            [
                ["equal_weight", "sixty_forty", 0.08, 0.05, 0.03, 0.01, 0.05, 0.02, True],
                ["risk_parity", "inverse_vol", 0.06, 0.07, -0.01, -0.03, 0.01, 0.6, False],
            ]
        )
    )
    assert list(table.columns) == [
        "Δ Ann. return",
        "CI low",
        "CI high",
        "p-value",
        "Distinguishable",
    ]
    assert "Equal weight − 60/40 benchmark" in table.index  # labelled, not raw keys
    assert table["p-value"].iloc[0] == 0.02


def test_return_verdict_reports_nothing_distinguishable_honestly():
    verdict = charts.return_significance_verdict(
        _return_pairs(
            [
                ["equal_weight", "sixty_forty", 0.08, 0.05, 0.03, -0.01, 0.07, 0.2, False],
                ["equal_weight", "inverse_vol", 0.08, 0.06, 0.02, -0.02, 0.06, 0.4, False],
            ]
        )
    )
    assert "None of the 2" in verdict and "within noise" in verdict


def test_return_verdict_lists_only_distinguishable_pairs():
    verdict = charts.return_significance_verdict(
        _return_pairs(
            [
                ["equal_weight", "sixty_forty", 0.08, 0.04, 0.04, 0.01, 0.07, 0.005, True],
                ["inverse_vol", "risk_parity", 0.06, 0.05, 0.01, -0.02, 0.04, 0.6, False],
            ]
        )
    )
    assert "1 of 2" in verdict and "p ≤ 0.005" in verdict
    assert "Equal weight vs 60/40 benchmark" in verdict  # labelled
    assert "Risk parity" not in verdict  # the indistinguishable pair is omitted


def test_return_verdict_handles_empty():
    assert "Not enough" in charts.return_significance_verdict(_return_pairs([]))


# --- yield_curve_chart: canonical 10Y-3M when available, 10Y-2Y proxy fallback -----------------


def _yc_df(with_3m: bool) -> pd.DataFrame:
    d = {
        "date": pd.bdate_range("2024-01-01", periods=10),
        "yield_curve_10y_2y": [0.3] * 10,
    }
    if with_3m:
        d["yield_curve_10y_3m"] = [0.55] * 10
    return pd.DataFrame(d)


def test_yield_curve_chart_uses_10y_3m_when_present():
    fig = charts.yield_curve_chart(_yc_df(with_3m=True))
    assert "10Y − 3M" in fig.layout.title.text
    assert fig.data[0].name == "10Y − 3M spread"
    assert list(fig.data[0].y) == [0.55] * 10  # plotted the 3M-based spread, not the 2Y
    assert "pp" in fig.data[0].hovertemplate  # hover shows the spread in pp, 2dp


def test_yield_curve_chart_falls_back_to_10y_2y_without_3m():
    fig = charts.yield_curve_chart(_yc_df(with_3m=False))
    assert "10Y − 2Y" in fig.layout.title.text
    assert fig.data[0].name == "10Y − 2Y spread"
    assert list(fig.data[0].y) == [0.3] * 10


def test_yield_curve_chart_falls_back_when_3m_all_null():
    df = _yc_df(with_3m=True)
    df["yield_curve_10y_3m"] = pd.NA  # column present but unpopulated → use the 2Y proxy
    fig = charts.yield_curve_chart(df)
    assert "10Y − 2Y" in fig.layout.title.text


# --- hover formatting: ticks AND hover read the same units ---------------------------------------


def test_price_and_vol_charts_format_hover():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=5),
            "close": [100, 110, 120, 130, 140],
            "ma_50": [100] * 5,
            "vol_20d": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
    )
    assert charts.price_chart(df, "SPY").layout.yaxis.hoverformat == "$,.2f"
    assert charts.vol_chart(df, "SPY").layout.yaxis.hoverformat == ".1%"


def test_macro_chart_hover_appends_unit_without_percent_scaling():
    # Percent series are stored already-in-percent (4.3, not 0.043), so hover must NOT use a
    # Plotly "%" format (that would render 4.3 as 430%); it appends the unit as plain text instead.
    df = pd.DataFrame(
        {"date": pd.bdate_range("2020-01-01", periods=5), "value": [4.3, 4.4, 4.5, 4.6, 4.7]}
    )
    assert "%{y:,.2f} %" in charts.macro_chart(df, "10Y yield", "%").data[0].hovertemplate
    assert "%{y:,.2f} index" in charts.macro_chart(df, "VIX", "index").data[0].hovertemplate


def test_scorecard_shape_and_labels():
    stats = pd.DataFrame(
        {
            "strategy": ["sixty_forty", "equal_weight"],
            "sharpe": [2.5, -0.6],
            "sharpe_lo": [0.4, -2.7],
            "sharpe_hi": [4.6, 1.6],
            "n_obs": [147, 147],
            "n_boot": [2000, 2000],
            "ci_pct": [0.9, 0.9],
        }
    )
    sc = charts.portfolio_scorecard(stats)
    assert list(sc.columns) == ["Sharpe", "CI low", "CI high"]
    assert "60/40 benchmark" in sc.index  # raw key mapped to a display label


def test_attribution_chart_uses_only_the_selected_strategy():
    attr = pd.DataFrame(
        {
            "strategy": ["equal_weight", "equal_weight", "sixty_forty"],
            "symbol": ["SPY", "TLT", "SPY"],
            "contribution_to_return": [0.02, -0.01, 0.05],
            "contribution_to_risk": [0.6, 0.4, 1.0],
        }
    )
    fig = charts.attribution_chart(attr, "equal_weight")
    bar = fig.data[0]
    assert set(bar.y) == {"SPY", "TLT"}  # only the selected strategy's assets, not sixty_forty's
    assert len(bar.x) == 2


def test_regime_sharpe_chart_one_trace_per_strategy_in_low_med_high_order():
    regime = pd.DataFrame(
        {
            "strategy": ["equal_weight"] * 3 + ["sixty_forty"] * 3,
            "regime": ["High", "Low", "Medium"] * 2,  # unordered on purpose
            "ann_return": [0.0] * 6,
            "ann_vol": [0.1] * 6,
            "ann_sharpe": [-0.3, -2.0, 1.1, 3.99, 1.66, 2.84],
        }
    )
    fig = charts.regime_sharpe_chart(regime)
    assert len(fig.data) == 2  # one grouped trace per strategy
    assert list(fig.data[0].x) == ["Low", "Medium", "High"]  # reindexed to a sensible order


def _gate(weight: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-28"]),
            "forecast_skill": [0.0, weight],
            "forecast_weight": [0.0, weight],
        }
    )


def _ml_pair(distinguishable: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_a": ["mvo_histmean"],
            "strategy_b": ["mvo_ml"],
            "sharpe_diff": [0.5 if distinguishable else 0.0],
            "diff_lo": [0.1 if distinguishable else -0.2],
            "diff_hi": [0.9 if distinguishable else 0.2],
            "distinguishable": [distinguishable],
        }
    )


def test_ml_verdict_reports_no_edge_when_not_distinguishable():
    verdict = charts.ml_verdict(_gate(0.0), _ml_pair(distinguishable=False))
    assert "0%" in verdict and "did not beat" in verdict


def test_ml_verdict_reports_edge_when_distinguishable():
    verdict = charts.ml_verdict(_gate(0.20), _ml_pair(distinguishable=True))
    assert "distinguishable" in verdict
    assert "did not beat" not in verdict


def _btc_effect(rows: list) -> pd.DataFrame:
    cols = [
        "strategy",
        "sharpe_ex",
        "sharpe_inc",
        "sharpe_diff",
        "diff_lo",
        "diff_hi",
        "distinguishable",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_btc_effect_verdict_reports_distinguishable_hurt():
    eff = _btc_effect(
        [
            ["equal_weight", 1.26, -0.32, -1.58, -2.76, -0.32, True],
            ["sixty_forty", 3.73, 3.73, 0.0, 0.0, 0.0, False],
        ]
    )
    verdict = charts.btc_effect_verdict(eff)
    assert "1 of 2" in verdict and "hurt" in verdict and "Equal weight" in verdict


def test_btc_effect_verdict_reports_none_when_all_within_noise():
    eff = _btc_effect([["mvo_ml", -0.6, -0.51, 0.09, -0.24, 0.37, False]])
    assert "no statistically distinguishable difference" in charts.btc_effect_verdict(eff)


def test_btc_effect_verdict_handles_empty():
    assert "not computed" in charts.btc_effect_verdict(_btc_effect([]))


# ---------------------------------------------------------------------------
# vol_forecast_value — the ML-tab headline must pick the SPY rv_har forecast
# specifically, never another asset's row via a positional .iloc[0].
# ---------------------------------------------------------------------------

_FC_COLS = ["symbol", "as_of", "predicted_next_return", "model"]


def _forecast(rows: list) -> pd.DataFrame:
    # Mirrors the marts.ml_forecast schema written by run_ml (symbol, as_of, pred, model).
    return pd.DataFrame(rows, columns=_FC_COLS)


def test_vol_forecast_picks_spy_rv_har_even_with_other_symbols_present():
    # BTC's rv_har row sorts first positionally: a model-only filter + .iloc[0] would surface
    # BTC's vol in the SPY headline. The symbol constraint must pin it to SPY's 0.18 daily-scale
    # forecast, then annualise it for display.
    fc = _forecast(
        [
            ["BTC", "2026-06-20", 0.99, "rv_har"],
            ["SPY", "2026-06-20", 0.18, "rv_har"],
            ["SPY", "2026-06-20", 0.01, "random_forest"],
        ]
    )
    assert charts.vol_forecast_value(fc, symbol="SPY") == 0.18 * math.sqrt(252)


def test_vol_forecast_none_when_selected_symbol_absent():
    # Only BTC has an rv_har forecast — the SPY headline must show nothing, not BTC's row.
    fc = _forecast([["BTC", "2026-06-20", 0.99, "rv_har"]])
    assert charts.vol_forecast_value(fc, symbol="SPY") is None


def test_vol_forecast_none_on_empty_or_missing_columns():
    # Empty frame and a frame lacking model/symbol must degrade to None — no IndexError.
    assert charts.vol_forecast_value(pd.DataFrame()) is None
    assert charts.vol_forecast_value(pd.DataFrame({"predicted_next_return": [0.2]})) is None


def test_vol_forecast_none_when_value_is_nan():
    # A NaN forecast must NOT render as "nan %": float(nan) is a float and would slip past the
    # caller's `is not None` check. It must return None so the dashboard shows the honest
    # "No SPY volatility forecast available yet." caption, not a looks-valid-but-isn't headline.
    fc = _forecast([["SPY", "2026-06-20", float("nan"), "rv_har"]])
    assert charts.vol_forecast_value(fc, symbol="SPY") is None


def test_vol_forecast_none_when_value_is_null_object_dtype():
    # An actual None/NULL (object-dtype column) must return None, not raise TypeError in float().
    fc = _forecast([["SPY", "2026-06-20", None, "rv_har"]])
    assert fc["predicted_next_return"].dtype == object  # guard: None stays NULL, not coerced
    assert charts.vol_forecast_value(fc, symbol="SPY") is None


def test_vol_forecast_none_when_value_is_infinite():
    # ±inf is non-finite — pd.isna treats it as non-null, so the explicit math.isfinite guard
    # keeps it out of the headline.
    fc = _forecast([["SPY", "2026-06-20", float("inf"), "rv_har"]])
    assert charts.vol_forecast_value(fc, symbol="SPY") is None


def test_return_forecast_table_sorts_return_rows_by_predicted_return():
    fc = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "as_of": "2026-07-01",
                "horizon": 252,
                "predicted_return": 0.10,
                "daily_mu": 0.0004,
                "model": "return_gb",
            },
            {
                "symbol": "BTC",
                "as_of": "2026-07-01",
                "horizon": 252,
                "predicted_return": 0.30,
                "daily_mu": 0.0012,
                "model": "return_gb",
            },
            {
                "symbol": "SPY",
                "as_of": "2026-07-01",
                "horizon": 5,
                "predicted_return": 0.02,
                "daily_mu": None,
                "model": "rv_har",
            },
        ]
    )

    out = charts.return_forecast_table(fc)

    assert out["symbol"].tolist() == ["BTC", "SPY"]
    assert out["predicted_return"].tolist() == [0.30, 0.10]
    assert "model" not in out.columns


def test_return_performance_table_pivots_asset_metrics():
    metrics = pd.DataFrame(
        [
            ["return_gb", "SPY", "ic", 0.2],
            ["return_gb", "SPY", "direction_accuracy", 0.61],
            ["return_gb", "SPY", "r2", -0.4],
            ["return_gb", "BTC", "ic", 0.5],
            ["return_gb", "BTC", "direction_accuracy", 0.55],
            ["rv_har", "SPY", "ic", 0.99],
        ],
        columns=["model", "symbol", "metric", "value"],
    )

    out = charts.return_performance_table(metrics)

    assert out["symbol"].tolist() == ["BTC", "SPY"]
    assert out.loc[out["symbol"] == "SPY", "r2"].iloc[0] == -0.4
    assert "ic" in out.columns
    assert "direction_accuracy" in out.columns


def test_return_regime_breakdown_table_reads_persisted_regime_metrics():
    metrics = pd.DataFrame(
        [
            ["return_gb", "SPY", "direction_accuracy_low", 0.52],
            ["return_gb", "SPY", "direction_accuracy_high", 0.58],
            ["return_gb", "SPY", "direction_accuracy", 0.55],
        ],
        columns=["model", "symbol", "metric", "value"],
    )

    out = charts.return_regime_breakdown_table(metrics)

    assert set(out["regime"]) == {"low", "high"}
    assert out.loc[out["regime"] == "high", "direction_accuracy"].iloc[0] == 0.58


# --- regime background shading --------------------------------------------------


def test_regime_shading_handles_none_or_empty():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
            "vol_20d": [0.1, 0.1, 0.1, 0.1, 0.1],
        }
    )
    # None or empty regime_df should add zero vrect layout shapes
    fig_none = charts.price_chart(df, "SPY", regime_df=None)
    fig_empty = charts.price_chart(df, "SPY", regime_df=pd.DataFrame())
    assert fig_none.layout.shapes == ()
    assert fig_empty.layout.shapes == ()


def test_price_chart_adds_regime_vrects():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=6),
            "close": [100, 101, 102, 103, 104, 105],
        }
    )
    regime_df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=6),
            "regime": ["Low", "Low", "Medium", "Medium", "High", "High"],
        }
    )

    fig = charts.price_chart(df, "SPY", regime_df=regime_df)
    shapes = fig.layout.shapes
    assert len(shapes) == 3  # 3 regime blocks: Low, Medium, High

    # Verify colors: Low -> faint green, Medium -> faint amber, High -> faint red
    colors = [s.fillcolor for s in shapes]
    assert "rgba(39, 192, 138, 0.12)" in colors[0]  # Low
    assert "rgba(255, 180, 84, 0.12)" in colors[1]  # Medium
    assert "rgba(255, 93, 108, 0.12)" in colors[2]  # High


def test_vol_chart_adds_regime_vrects():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "vol_20d": [0.1, 0.12, 0.15, 0.2],
        }
    )
    regime_df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=4),
            "regime": ["Low", "Low", "High", "High"],
        }
    )

    fig = charts.vol_chart(df, "SPY", regime_df=regime_df)
    shapes = fig.layout.shapes
    assert len(shapes) == 2


def test_rebased_performance_chart_adds_regime_vrects():
    perf_long = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "TLT", "TLT"],
            "asset_class": ["equities", "equities", "bonds", "bonds"],
            "date": pd.bdate_range("2024-01-01", periods=2).tolist() * 2,
            "perf": [0.0, 0.01, 0.0, -0.005],
        }
    )
    regime_df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2),
            "regime": ["Low", "Medium"],
        }
    )

    fig = charts.rebased_performance_chart(perf_long, regime_df=regime_df)
    # The chart's own zero line (add_hline) renders as a non-rect shape, so count only the
    # regime vrects added by the shading helper.
    vrects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(vrects) == 2


def test_regime_shading_handles_multi_symbol_regime_df():
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "close": [100, 101, 102],
        }
    )
    regime_df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "TLT", "TLT"],
            "date": pd.bdate_range("2024-01-01", periods=2).tolist() * 2,
            "regime": ["Low", "High", "Medium", "Medium"],
        }
    )

    # Calling for SPY should select only SPY rows from regime_df
    fig = charts.price_chart(df, "SPY", regime_df=regime_df)
    shapes = fig.layout.shapes
    assert len(shapes) == 2


def test_forecast_fan_points_anchors_and_geometry():
    fan = charts.forecast_fan_points(
        daily_mu=0.001, horizon=20.0, sigma_daily=0.01, oos_r2=0.5, z=1.0, n_points=21
    )
    assert list(fan.columns) == ["t", "center", "lower", "upper"]
    assert len(fan) == 21
    assert fan["t"].iloc[0] == 0.0
    assert fan["t"].iloc[-1] == 20.0
    # Zero uncertainty today: band collapses to the centre.
    assert fan["lower"].iloc[0] == fan["upper"].iloc[0] == fan["center"].iloc[0]
    # Centre is the linear drift; at the horizon it equals daily_mu * horizon.
    assert math.isclose(fan["center"].iloc[-1], 0.001 * 20.0, rel_tol=1e-12)
    # Width at horizon = z * sigma_daily * sqrt(horizon) * sqrt(1 - r2).
    expected_half = 1.0 * 0.01 * math.sqrt(20.0) * math.sqrt(0.5)
    assert math.isclose(fan["upper"].iloc[-1] - fan["center"].iloc[-1], expected_half, rel_tol=1e-9)
    # Band widens monotonically (no-shrinkage).
    widths = fan["upper"] - fan["lower"]
    assert (widths.diff().dropna() >= -1e-12).all()


def test_forecast_fan_points_honest_r2_defaults():
    kwargs = dict(daily_mu=0.001, horizon=20.0, sigma_daily=0.01, n_points=11)
    base = charts.forecast_fan_points(oos_r2=0.0, **kwargs)
    missing = charts.forecast_fan_points(oos_r2=None, **kwargs)
    negative = charts.forecast_fan_points(oos_r2=-0.8, **kwargs)
    perfect = charts.forecast_fan_points(oos_r2=1.0, **kwargs)
    # Missing/negative R2 fall back to the honest no-skill (widest) band.
    for fan in (missing, negative):
        assert fan["upper"].equals(base["upper"])
        assert fan["lower"].equals(base["lower"])
    # R2 = 1 => zero residual error at the horizon.
    assert math.isclose(perfect["upper"].iloc[-1] - perfect["center"].iloc[-1], 0.0, abs_tol=1e-12)
    # z = 0 collapses the band everywhere.
    narrow = charts.forecast_fan_points(oos_r2=0.0, z=0.0, **kwargs)
    assert (narrow["lower"] == narrow["upper"]).all()


def test_forecast_fan_points_guards_bad_inputs():
    empty = charts.forecast_fan_points(daily_mu=0.001, horizon=0.0, sigma_daily=0.01)
    assert empty.empty
    empty2 = charts.forecast_fan_points(daily_mu=0.001, horizon=20.0, sigma_daily=float("nan"))
    assert empty2.empty


def test_forecast_fan_chart_renders_band_and_center():
    fan = charts.forecast_fan_points(
        daily_mu=0.001, horizon=20.0, sigma_daily=0.01, oos_r2=0.5, n_points=21
    )
    fig = charts.forecast_fan_chart(fan, "SPY", as_of="2026-08-07", z=1.0, height=300)
    names = [tr.name for tr in fig.data if tr.name]
    assert "Expected path" in names
    assert "±1σ band" in names
    # The band trace carries the fill.
    band = next(tr for tr in fig.data if tr.name == "±1σ band")
    assert band.fill == "tonexty"
    assert len(fig.data) == 3


def test_forecast_fan_chart_tolerates_empty_fan():
    empty = pd.DataFrame(columns=["t", "center", "lower", "upper"])
    fig = charts.forecast_fan_chart(empty, "SPY", as_of=None)
    assert len(fig.data) == 0


# --- relative value charts: ratio line + rolling z-score (theme tokens only) --------------------


def _rv_long(sym_closes: list[float], bench_closes: list[float]) -> pd.DataFrame:
    """A 2-symbol long frame (SYM vs BEN) over shared business dates."""
    dates = pd.bdate_range("2024-01-01", periods=len(sym_closes))
    rows = []
    for d, (sc, bc) in zip(dates, zip(sym_closes, bench_closes, strict=True), strict=True):
        rows.append(("SYM", "equities", d.date(), float(sc), 0.01))
        rows.append(("BEN", "equities", d.date(), float(bc), 0.01))
    return pd.DataFrame(rows, columns=["symbol", "asset_class", "date", "close", "daily_return"])


def test_relative_strength_chart_uses_accent_token_and_reference_line():
    long_df = _rv_long([100.0, 110.0, 121.0], [100.0] * 3)
    ratio = charts.relative_strength_ratio(long_df, "SYM", "BEN")
    fig = charts.relative_strength_chart(ratio, "SYM", "BEN")
    assert "SYM vs BEN" in fig.layout.title.text
    assert "rebased to 1.0" in fig.layout.title.text
    assert fig.data[0].line.color == charts.PALETTE["accent"]
    # The 1.0 reference line is a muted dashed hline (named token, not inline hex).
    refs = [s for s in fig.layout.shapes if float(s.y0) == 1.0 and float(s.y1) == 1.0]
    assert refs and refs[0].line.color == charts.PALETTE["muted"]
    assert refs[0].line.dash == "dot"


def test_relative_strength_chart_degrades_on_empty():
    fig = charts.relative_strength_chart(pd.DataFrame(), "SYM", "BEN")
    assert len(fig.data) == 0  # reference line only, no crash


def test_ratio_zscore_chart_has_extreme_bands_and_named_tokens():
    long_df = _rv_long([100.0, 110.0, 121.0], [100.0] * 3)
    z = charts.ratio_rolling_zscore(long_df, "SYM", "BEN", window=2)
    fig = charts.ratio_zscore_chart(z, "SYM", "BEN", window=2)
    assert "(2d window)" in fig.layout.title.text
    assert fig.data[0].line.color == charts.SERIES_VOL
    hlines = {(round(float(s.y0), 3), round(float(s.y1), 3)) for s in fig.layout.shapes}
    assert {(0.0, 0.0), (2.0, 2.0), (-2.0, -2.0)} <= hlines  # mean + ±2σ reference lines
    colors = {s.line.color for s in fig.layout.shapes}
    assert colors <= {charts.PALETTE["up"], charts.PALETTE["down"], charts.PALETTE["muted"]}


def test_ratio_zscore_chart_degrades_on_empty():
    fig = charts.ratio_zscore_chart(pd.DataFrame(), "SYM", "BEN")
    assert len(fig.data) == 0  # reference lines only, no crash
