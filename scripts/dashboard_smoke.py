"""Dashboard read-path smoke: exercise the marts-backed accessors so schema drift fails CI.

Run after `mmi seed` + `dbt build` against a local DuckDB. Because dashboard.data.query() only
swallows missing-table errors, a renamed/removed column on any marts table raises here.
(The ML/AI marts — model_metrics/ml_forecast/fct_regime/market_brief — are intentionally not
exercised: `make ci` does not run `mmi ml`/`mmi ai`, so those tables don't exist in ci.duckdb.)
"""

from dashboard import data
from dashboard.components import charts

assert data.db_exists(), "db_exists() is False — the marts DB is missing"

assets = data.assets()
assert not assets.empty, "marts.dim_asset is empty — dashboard cannot read the core marts"
assert {"symbol", "asset_class"} <= set(assets.columns)

# Exercise every core marts accessor; drift (missing column/table) now surfaces as an error.
data.market_macro()
for sym in assets[assets["asset_class"] != "crypto"]["symbol"].tolist()[:3]:
    data.asset_daily(sym)
for mid in data.macro_ids():
    data.macro(mid)
for sym in data.crypto_symbols():
    data.crypto_intraday(sym)

# Portfolio read-path + chart/summary builders (the backtest mart may be absent on a partial run).
pf = data.portfolio_returns()
if not pf.empty:
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
stats = data.portfolio_strategy_stats()
if not stats.empty:
    assert {"strategy", "sharpe", "sharpe_lo", "sharpe_hi", "n_obs", "n_boot", "ci_pct"} <= set(
        stats.columns
    ), f"fct_portfolio_strategy_stats columns drifted: {set(stats.columns)}"
    charts.portfolio_scorecard(stats)
pairs = data.portfolio_strategy_pairs()
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
attr = data.portfolio_attribution()
if not attr.empty:
    assert {"strategy", "symbol", "contribution_to_return", "contribution_to_risk"} <= set(
        attr.columns
    ), f"fct_performance_attribution columns drifted: {set(attr.columns)}"
    for strat in attr["strategy"].unique():
        charts.attribution_chart(attr, strat)
regime = data.portfolio_regime_performance()
if not regime.empty:
    assert {"strategy", "regime", "ann_return", "ann_vol", "ann_sharpe"} <= set(regime.columns), (
        f"fct_portfolio_regime_performance columns drifted: {set(regime.columns)}"
    )
    charts.regime_sharpe_chart(regime)
    print(f"attribution + regime read-path OK ({len(attr)} attr rows, {len(regime)} regime rows)")

# ML gate read-path + chart/verdict builders (the "did the forecast add value?" surface).
gate = data.portfolio_ml_gate()
if not gate.empty:
    assert {"date", "forecast_skill", "forecast_weight"} <= set(gate.columns), (
        f"fct_portfolio_ml_gate columns drifted: {set(gate.columns)}"
    )
    charts.ml_gate_chart(gate)
    assert isinstance(charts.ml_verdict(gate, pairs), str)
    print(f"ml gate read-path OK ({len(gate)} rebalances)")

print(f"dashboard read-path OK ({len(assets)} assets, core marts accessors exercised)")
