"""H5 — Bootstrap completeness, honesty, and size test.

Validates the committed data/public/ WITHOUT running any pipeline — fast, safe for make ci.

Checks:
  (a) Every required mart present by exact name (17 marts incl. all 7 portfolio marts).
  (b) Each parquet readable via in-memory DuckDB snapshot connection; column NAMES match
      what dashboard/data.py SELECTs.
  (c) fct_asset_daily has the dbt-model column set (not the transform_fallback 5-mart shape)
      AND source is uniformly 'sample'.
  (d) model_metrics contains BOTH direction rows (model='random_forest') and vol rows
      (model='rv_har').
  (e) Each committed *.parquet < 2_000_000 bytes.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

# ---------------------------------------------------------------------------
# Locate data/public — resolve relative to this file so it works from any cwd.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "data" / "public"

# ---------------------------------------------------------------------------
# Required mart table names (exact; enumerated — no globs).
# ---------------------------------------------------------------------------
REQUIRED_MARTS = [
    # Core markets
    "dim_asset",
    "fct_asset_daily",
    "fct_crypto_intraday",
    "fct_macro_indicator",
    "fct_market_macro",
    "fct_regime",
    "fct_recession_risk",
    # ML / AI
    "model_metrics",
    "ml_forecast",
    "market_brief",
    # Portfolio (7 marts)
    "fct_portfolio_returns",
    "fct_portfolio_strategy_stats",
    "fct_portfolio_strategy_pairs",
    "fct_performance_attribution",  # NOT fct_portfolio_attribution
    "fct_portfolio_regime_performance",
    "fct_portfolio_ml_gate",
    "fct_portfolio_btc_effect",
]

# Size cap from Contract A (D6 default).
MAX_BYTES = 2_000_000

# ---------------------------------------------------------------------------
# Snapshot connection helper — mirrors dashboard/data.py _snapshot_connection()
# ---------------------------------------------------------------------------


def _snapshot_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("set python_enable_replacements=false")
    con.execute("create schema if not exists raw")
    con.execute("create schema if not exists marts")
    for path in sorted(SNAPSHOT_DIR.glob("*.parquet")):
        safe_path = str(path).replace("'", "''")
        safe_name = path.stem.replace('"', '""')
        con.execute(
            f"create view marts.\"{safe_name}\" as select * from read_parquet('{safe_path}')"
        )
    return con


# ---------------------------------------------------------------------------
# (a) Completeness — every required mart present as a .parquet file
# ---------------------------------------------------------------------------


def test_all_required_marts_present() -> None:
    """Every mart in REQUIRED_MARTS must have a .parquet in data/public/."""
    assert SNAPSHOT_DIR.is_dir(), (
        f"data/public/ does not exist — bootstrap snapshot not committed? ({SNAPSHOT_DIR})"
    )
    present = {p.stem for p in SNAPSHOT_DIR.glob("*.parquet")}
    missing = [t for t in REQUIRED_MARTS if t not in present]
    assert not missing, f"Missing marts in data/public/: {missing}\nPresent: {sorted(present)}"


# ---------------------------------------------------------------------------
# (b) Column contracts — accessors' selected columns must exist in each mart
# ---------------------------------------------------------------------------

MART_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "dim_asset": {"symbol", "asset_class"},
    "fct_asset_daily": {
        "date",
        "close",
        "daily_return",
        "vol_20d",
        "ma_50",
        "source",
        "open",
        "high",
        "low",
    },
    "fct_crypto_intraday": {"ts", "price_usd", "pct_change", "symbol"},
    "fct_macro_indicator": {"date", "value", "change", "series_id"},
    "fct_market_macro": {"date"},
    "fct_regime": {"date", "vol_20d", "regime"},
    "fct_recession_risk": {"date", "spread_10y_3m", "recession_prob", "model"},
    "model_metrics": {"model", "symbol", "metric", "value", "trained_at"},
    "ml_forecast": {"symbol", "as_of", "predicted_next_return", "model"},
    "market_brief": {"created_at", "engine", "brief"},
    "fct_portfolio_returns": {
        "strategy",
        "date",
        "daily_return",
        "cumulative_return",
        "drawdown",
        "rolling_sharpe_252",
        "window_id",
    },
    "fct_portfolio_strategy_stats": {
        "strategy",
        "sharpe",
        "sharpe_lo",
        "sharpe_hi",
        "n_obs",
        "n_boot",
        "ci_pct",
        "window_id",
    },
    "fct_portfolio_strategy_pairs": {
        "strategy_a",
        "strategy_b",
        "sharpe_diff",
        "diff_lo",
        "diff_hi",
        "distinguishable",
        "window_id",
    },
    "fct_performance_attribution": {
        "strategy",
        "symbol",
        "contribution_to_return",
        "contribution_to_risk",
        "window_id",
    },
    "fct_portfolio_regime_performance": {
        "strategy",
        "regime",
        "n_days",
        "day_share",
        "ann_return",
        "ann_vol",
        "ann_sharpe",
        "window_id",
    },
    "fct_portfolio_ml_gate": {"date", "forecast_skill", "forecast_weight", "window_id"},
    "fct_portfolio_btc_effect": {
        "strategy",
        "sharpe_ex",
        "sharpe_inc",
        "sharpe_diff",
        "diff_lo",
        "diff_hi",
        "distinguishable",
    },
}


@pytest.mark.parametrize("table", REQUIRED_MARTS)
def test_mart_columns_present(table: str) -> None:
    """Each mart's columns must include everything dashboard/data.py SELECTs."""
    if table not in MART_REQUIRED_COLUMNS:
        pytest.skip(f"No column contract specified for {table!r}")

    con = _snapshot_con()
    try:
        df = con.execute(f'describe select * from marts."{table}"').df()
        actual_cols = set(df["column_name"].tolist())
    finally:
        con.close()

    required = MART_REQUIRED_COLUMNS[table]
    missing = required - actual_cols
    assert not missing, (
        f"marts.{table} is missing columns that dashboard/data.py SELECTs: {missing}\n"
        f"Actual columns: {sorted(actual_cols)}"
    )


# ---------------------------------------------------------------------------
# (c) fct_asset_daily — dbt-model shape + source='sample'
# ---------------------------------------------------------------------------


def test_fct_asset_daily_dbt_model_shape() -> None:
    """fct_asset_daily must have the dbt-model column set (not the transform_fallback shape)."""
    # dbt model: symbol, asset_class, date, open, high, low, close, volume, source,
    #            daily_return, vol_20d, ma_50.
    # transform_fallback (5-mart shape) lacks open/high/low/volume and has 'source' as NaN.
    dbt_model_cols = {"open", "high", "low", "volume", "source", "ma_50", "vol_20d", "daily_return"}
    con = _snapshot_con()
    try:
        df = con.execute('describe select * from marts."fct_asset_daily"').df()
        actual = set(df["column_name"].tolist())
    finally:
        con.close()
    missing = dbt_model_cols - actual
    assert not missing, (
        f"fct_asset_daily is missing dbt-model columns: {missing} — "
        "this looks like the transform_fallback shape, not the dbt build output."
    )


def test_fct_asset_daily_source_is_sample() -> None:
    """All rows in fct_asset_daily must have source='sample' so is_sample_data()==True."""
    con = _snapshot_con()
    try:
        df = con.execute('select distinct source from marts."fct_asset_daily"').df()
    finally:
        con.close()
    sources = set(df["source"].dropna().tolist())
    assert sources == {"sample"}, (
        f"fct_asset_daily.source must be uniformly 'sample' for the bootstrap snapshot; "
        f"found: {sources}"
    )


# ---------------------------------------------------------------------------
# (d) model_metrics — both direction rows and rv_har rows
# ---------------------------------------------------------------------------


def test_model_metrics_has_direction_and_vol_rows() -> None:
    """model_metrics must contain BOTH model='random_forest' and model='rv_har' rows."""
    con = _snapshot_con()
    try:
        df = con.execute('select distinct model from marts."model_metrics"').df()
    finally:
        con.close()
    models = set(df["model"].tolist())
    assert "random_forest" in models, (
        f"model_metrics is missing direction model rows (model='random_forest'). Found: {models}"
    )
    assert "rv_har" in models, (
        f"model_metrics is missing HAR vol model rows (model='rv_har'). Found: {models}"
    )


# ---------------------------------------------------------------------------
# (e) Size cap — each .parquet < 2 MB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parquet_path",
    sorted(SNAPSHOT_DIR.glob("*.parquet")) if SNAPSHOT_DIR.is_dir() else [],
    ids=lambda p: p.name,
)
def test_parquet_under_size_cap(parquet_path: Path) -> None:
    """Each committed Parquet file must be < 2 000 000 bytes (Contract A D6 cap)."""
    size = parquet_path.stat().st_size
    assert size < MAX_BYTES, (
        f"{parquet_path.name} is {size:,} bytes — exceeds the 2 MB cap ({MAX_BYTES:,}). "
        "Use a downsampled mart (Contract A D9) if the real snapshot exceeds this limit."
    )


# ---------------------------------------------------------------------------
# Bonus: _manifest.json present
# ---------------------------------------------------------------------------


def test_manifest_present() -> None:
    """data/public/_manifest.json must exist alongside the parquet files."""
    manifest = SNAPSHOT_DIR / "_manifest.json"
    assert manifest.exists(), (
        f"_manifest.json not found in {SNAPSHOT_DIR} — snapshot may be incomplete."
    )
