"""Dashboard read-path smoke: exercise the marts-backed accessors so schema drift fails CI.

Run after `mmi seed` + `dbt build` + `mmi ml` + `mmi ai` against a local DuckDB. Because
dashboard.data.query() only swallows missing-table errors, a renamed/removed column on any marts
table raises here. The ML/AI marts (model_metrics/ml_forecast/fct_regime/market_brief) are now
exercised too, since `make ci` runs `mmi ml`/`mmi ai` before this smoke.
"""

import pandas as pd
from dashboard import data
from dashboard.components import charts

assert data.db_exists(), "db_exists() is False — the marts DB is missing"

# Provenance accessors (scope 5): must not raise and must return sane types/values.
assert isinstance(data.data_as_of(), str), "data_as_of() must return a str"
assert data.is_sample_data() in (True, False, None), "is_sample_data() must be tri-state"
print(f"provenance read-path OK (as_of={data.data_as_of()!r}, sample={data.is_sample_data()})")

assets = data.assets()
assert not assets.empty, "marts.dim_asset is empty — dashboard cannot read the core marts"
assert {"symbol", "asset_class"} <= set(assets.columns)

# Exercise every core marts accessor; drift (missing column/table) now surfaces as an error.
data.market_macro()
for sym in assets[assets["asset_class"] != "crypto"]["symbol"].tolist()[:3]:
    data.asset_daily(sym)
for mid in data.macro_ids():
    data.macro(mid)

# Portfolio read-path + chart/summary builders (the backtest mart may be absent on a partial run).
# Phase D: drive every portfolio accessor through one selected window (the dashboard's choke point).
windows_present = data.portfolio_windows()
window_id = windows_present[0] if windows_present else "ex_btc_2002"
pf = data.portfolio_returns(window_id)
if not pf.empty:
    assert windows_present, "fct_portfolio_returns has rows but portfolio_windows() is empty"
    expected = {
        "strategy",
        "date",
        "daily_return",
        "cumulative_return",
        "drawdown",
        "rolling_sharpe_252",
    }
    assert expected <= set(pf.columns), f"fct_portfolio_returns columns drifted: {set(pf.columns)}"
    charts.portfolio_cumulative_chart(pf)
    charts.portfolio_drawdown_chart(pf)
    charts.portfolio_sharpe_chart(pf)
    charts.portfolio_summary(pf)
    print(f"portfolio read-path OK ({pf['strategy'].nunique()} strategies)")

# Bootstrap scorecard read-path + builders (the uncertainty-quantification marts).
stats = data.portfolio_strategy_stats(window_id)
if not stats.empty:
    assert {"strategy", "sharpe", "sharpe_lo", "sharpe_hi", "n_obs", "n_boot", "ci_pct"} <= set(
        stats.columns
    ), f"fct_portfolio_strategy_stats columns drifted: {set(stats.columns)}"
    charts.portfolio_scorecard(stats)
pairs = data.portfolio_strategy_pairs(window_id)
if not pairs.empty:
    assert {
        "strategy_a",
        "strategy_b",
        "sharpe_diff",
        "diff_lo",
        "diff_hi",
        "distinguishable",
    } <= set(pairs.columns), f"fct_portfolio_strategy_pairs columns drifted: {set(pairs.columns)}"
    charts.portfolio_pairs_table(pairs)
    assert isinstance(charts.distinguishability_verdict(pairs), str)
    print(f"bootstrap scorecard read-path OK ({len(pairs)} pairs)")

# Attribution + regime-conditional read-path + chart builders.
attr = data.portfolio_attribution(window_id)
if not attr.empty:
    assert {"strategy", "symbol", "contribution_to_return", "contribution_to_risk"} <= set(
        attr.columns
    ), f"fct_performance_attribution columns drifted: {set(attr.columns)}"
    for strat in attr["strategy"].unique():
        charts.attribution_chart(attr, strat)
regime = data.portfolio_regime_performance(window_id)
if not regime.empty:
    assert {"strategy", "regime", "ann_return", "ann_vol", "ann_sharpe"} <= set(regime.columns), (
        f"fct_portfolio_regime_performance columns drifted: {set(regime.columns)}"
    )
    charts.regime_sharpe_chart(regime)
    print(f"attribution + regime read-path OK ({len(attr)} attr rows, {len(regime)} regime rows)")

# ML gate read-path + chart/verdict builders (the "did the forecast add value?" surface).
gate = data.portfolio_ml_gate(window_id)
if not gate.empty:
    assert {"date", "forecast_skill", "forecast_weight"} <= set(gate.columns), (
        f"fct_portfolio_ml_gate columns drifted: {set(gate.columns)}"
    )
    charts.ml_gate_chart(gate)
    assert isinstance(charts.ml_verdict(gate, pairs), str)
    print(f"ml gate read-path OK ({len(gate)} rebalances)")

# BTC-effect read-path + chart/verdict builders (the cross-window paired comparison).
btc_effect = data.portfolio_btc_effect()
if not btc_effect.empty:
    assert {"strategy", "sharpe_diff", "diff_lo", "diff_hi", "distinguishable"} <= set(
        btc_effect.columns
    ), f"fct_portfolio_btc_effect columns drifted: {set(btc_effect.columns)}"
    charts.btc_effect_chart(btc_effect)
    assert isinstance(charts.btc_effect_verdict(btc_effect), str)
    print(f"btc effect read-path OK ({len(btc_effect)} strategies)")

# ML/AI read-path: `make ci` now runs `mmi ml` + `mmi ai` before this smoke, so these marts exist
# and drift on the ML/AI dashboard tabs is caught here too. model_metrics/ml_forecast are only
# populated when a symbol has enough history to backtest (guard on non-empty); the regime labels
# and the brief are always written, so the brief is asserted present.
metrics = data.model_metrics()
if not metrics.empty:
    assert {"model", "symbol", "metric", "value", "trained_at"} <= set(metrics.columns), (
        f"marts.model_metrics columns drifted: {set(metrics.columns)}"
    )
forecast = data.ml_forecast()
if not forecast.empty:
    assert {"symbol", "as_of", "predicted_next_return", "model"} <= set(forecast.columns), (
        f"marts.ml_forecast columns drifted: {set(forecast.columns)}"
    )
# fct_regime is always written by `mmi ml`; require at least one asset to carry labelled regimes
# (the accessor's explicit SELECT also raises on a renamed column, so drift fails regardless).
regime_found = False
for sym in assets["symbol"].tolist():
    reg = data.regimes(sym)
    if not reg.empty:
        assert {"date", "vol_20d", "regime"} <= set(reg.columns), (
            f"marts.fct_regime columns drifted: {set(reg.columns)}"
        )
        regime_found = True
        break
assert regime_found, "marts.fct_regime has no rows for any asset — `mmi ml` did not label regimes"
brief = data.latest_brief()
assert not brief.empty, "marts.market_brief is empty — `mmi ai` did not persist a brief"
assert {"created_at", "engine", "brief"} <= set(brief.columns), (
    f"marts.market_brief columns drifted: {set(brief.columns)}"
)
print(f"ml/ai read-path OK (brief engine: {brief.iloc[0]['engine']}, {len(metrics)} metric rows)")

# ---------------------------------------------------------------------------
# B7: honest vol-skill chart builders — exercise with both populated and empty
# model_metrics DataFrames so the builders are proven crash-free in both states.
# ---------------------------------------------------------------------------

# -- empty-metrics path: builders must return figures / strings without raising
_empty_metrics = pd.DataFrame(columns=["model", "symbol", "metric", "value", "trained_at"])
_r2_fig_empty = charts.vol_skill_r2_chart(_empty_metrics, symbol="SPY")
assert _r2_fig_empty is not None, "vol_skill_r2_chart returned None on empty metrics"
_qlike_fig_empty = charts.vol_skill_qlike_chart(_empty_metrics, symbol="SPY")
assert _qlike_fig_empty is not None, "vol_skill_qlike_chart returned None on empty metrics"
_verdict_empty = charts.vol_skill_verdict_text(_empty_metrics, symbol="SPY")
assert isinstance(_verdict_empty, str), "vol_skill_verdict_text must return str on empty metrics"
assert "no demonstrated out-of-sample edge" in _verdict_empty, (
    "vol_skill_verdict_text must use honest 'no edge' language when metrics are absent"
)
_dir_fig_empty = charts.direction_skill_chart(_empty_metrics, symbol="SPY")
assert _dir_fig_empty is not None, "direction_skill_chart returned None on empty metrics"
print("vol-skill builders OK (empty-metrics path — honest 'no edge' language verified)")

# -- populated metrics path: exercise with a synthetic not-cleared row set
_not_cleared_rows = pd.DataFrame(
    [
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "oos_r2",
            "value": 0.05,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "qlike",
            "value": 1.2,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "baseline_qlike",
            "value": 1.3,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "qlike_skill_ratio",
            "value": 0.92,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "folds_passed",
            "value": 2,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "n_folds",
            "value": 5,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "n_obs",
            "value": 300,
            "trained_at": "2026-01-01",
        },
        # direction model rows (honest secondary)
        {
            "model": "random_forest",
            "symbol": "SPY",
            "metric": "dir_acc",
            "value": 0.52,
            "trained_at": "2026-01-01",
        },
        {
            "model": "random_forest",
            "symbol": "SPY",
            "metric": "baseline_dir_acc",
            "value": 0.50,
            "trained_at": "2026-01-01",
        },
        {
            "model": "random_forest",
            "symbol": "SPY",
            "metric": "mae",
            "value": 0.008,
            "trained_at": "2026-01-01",
        },
        {
            "model": "random_forest",
            "symbol": "SPY",
            "metric": "mae_baseline",
            "value": 0.009,
            "trained_at": "2026-01-01",
        },
    ]
)
charts.vol_skill_r2_chart(_not_cleared_rows)
charts.vol_skill_qlike_chart(_not_cleared_rows)
_verdict_not_cleared = charts.vol_skill_verdict_text(_not_cleared_rows)
assert isinstance(_verdict_not_cleared, str)
# oos_r2=0.05 < 0.10 and folds_passed=2 < ceil(0.6*5)=3 → not cleared
assert "no demonstrated out-of-sample edge" in _verdict_not_cleared, (
    f"expected honest 'no edge' language when gate not cleared; got: {_verdict_not_cleared!r}"
)
assert "beats baseline" not in _verdict_not_cleared, (
    "must NOT claim 'beats baseline' when gate not cleared"
)
charts.direction_skill_chart(_not_cleared_rows)
print("vol-skill builders OK (not-cleared metrics path — honest language verified)")

# -- cleared metrics path: OOS R²≥0.10, ratio<0.99, folds_passed≥3
_cleared_rows = pd.DataFrame(
    [
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "oos_r2",
            "value": 0.15,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "qlike",
            "value": 1.1,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "baseline_qlike",
            "value": 1.3,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "qlike_skill_ratio",
            "value": 0.846,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "folds_passed",
            "value": 4,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "n_folds",
            "value": 5,
            "trained_at": "2026-01-01",
        },
        {
            "model": "rv_har",
            "symbol": "SPY",
            "metric": "n_obs",
            "value": 300,
            "trained_at": "2026-01-01",
        },
    ]
)
charts.vol_skill_r2_chart(_cleared_rows)
charts.vol_skill_qlike_chart(_cleared_rows)
_verdict_cleared = charts.vol_skill_verdict_text(_cleared_rows)
assert isinstance(_verdict_cleared, str)
assert "beats baseline OOS" in _verdict_cleared, (
    f"expected 'beats baseline OOS' when gate cleared; got: {_verdict_cleared!r}"
)
# Verify scope caption is present in cleared verdict
assert charts.ML_SCOPE_CAPTION in _verdict_cleared or "SPY" in _verdict_cleared, (
    "cleared verdict must reference the forecast scope / SPY"
)
print("vol-skill builders OK (cleared metrics path — 'beats baseline OOS' language verified)")

print(f"dashboard read-path OK ({len(assets)} assets, core marts accessors exercised)")
