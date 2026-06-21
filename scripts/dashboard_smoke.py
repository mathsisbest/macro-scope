"""Dashboard read-path smoke: exercise the marts-backed accessors so schema drift fails CI.

Run after `mmi seed` + `dbt build` against a local DuckDB. Because dashboard.data.query() only
swallows missing-table errors, a renamed/removed column on any marts table raises here.
(The ML/AI marts — model_metrics/ml_forecast/fct_regime/market_brief — are intentionally not
exercised: `make ci` does not run `mmi ml`/`mmi ai`, so those tables don't exist in ci.duckdb.)
"""

from dashboard import data

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

print(f"dashboard read-path OK ({len(assets)} assets, core marts accessors exercised)")
