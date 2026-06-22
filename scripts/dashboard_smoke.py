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

print(f"dashboard read-path OK ({len(assets)} assets, core marts accessors exercised)")
