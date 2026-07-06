"""Cached, read-only access to the marts for the dashboard."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from mmi.portfolio import windows
from mmi.settings import load_assets, settings
from mmi.utils.db import connect


def db_exists() -> bool:
    # Snapshot mode reads committed Parquet (no DB): "present" iff the snapshot dir holds a mart.
    # On MotherDuck the store is remote; locally we check the file exists.
    if settings.snapshot_mode:
        return settings.snapshot_dir.is_dir() and any(settings.snapshot_dir.glob("*.parquet"))
    return True if settings.use_motherduck else Path(settings.duckdb_path).exists()


def _snapshot_connection() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB whose schemas are the committed Parquet snapshot.

    Each ``<table>.parquet`` in ``snapshot_dir`` is registered as a view ``marts.<table>``, so the
    accessors' existing ``select ... from marts.<table>`` queries run unchanged — no live DB, no
    secrets. ``raw``/``marts`` are pre-created so a query for a mart with no Parquet file (e.g.
    ``raw.pipeline_runs``, never snapshotted) raises a clean missing-table CatalogException that
    ``query()`` swallows to an empty frame — exactly as a missing table does on a live DB.
    """
    con = duckdb.connect(":memory:")
    # Disable Python replacement scans: a missing table must raise a clean CatalogException (which
    # query() swallows to empty), NOT get silently "replaced" by a same-named Python object in the
    # call stack — e.g. data.ml_forecast (the accessor function) before that mart is snapshotted.
    con.execute("set python_enable_replacements=false")
    con.execute("create schema if not exists raw")
    con.execute("create schema if not exists marts")
    for path in sorted(settings.snapshot_dir.glob("*.parquet")):
        # The dir is ours, but escape defensively: '' for the path literal, "" for the identifier.
        safe_path = str(path).replace("'", "''")
        safe_name = path.stem.replace('"', '""')
        con.execute(
            f"create view marts.\"{safe_name}\" as select * from read_parquet('{safe_path}')"
        )
    return con


@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a read-only query against the live DB, or the Parquet snapshot in snapshot mode.

    A *missing table* (e.g. ML/AI marts before those steps run) returns an empty frame.
    Connection/auth errors and schema drift (a missing column) are NOT swallowed — they must
    surface rather than silently rendering every panel blank.
    """
    if not db_exists():
        return pd.DataFrame()
    con = _snapshot_connection() if settings.snapshot_mode else connect(read_only=True)
    try:
        return con.execute(sql, list(params) if params else []).df()
    except duckdb.CatalogException:
        return pd.DataFrame()
    finally:
        con.close()


def snapshot_manifest() -> dict | None:
    """Read the snapshot manifest if in snapshot mode, else ``None``.

    ``_manifest.json`` is written by ``mmi snapshot`` alongside the Parquet files.
    It contains ``generated_at`` and per-table ``rows`` counts, so the dashboard can
    display honest provenance even when ``raw.pipeline_runs`` is absent.
    """
    if not settings.snapshot_mode:
        return None
    manifest_path = settings.snapshot_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# Date-range presets for the global chart range selector (Google-Finance style).
RANGE_PRESETS: tuple[str, ...] = ("1M", "6M", "YTD", "1Y", "5Y", "Max")


def range_start(preset: str | None, anchor: str) -> str | None:
    """Map a range preset to an ISO start-date floor relative to ``anchor`` (the latest data date,
    ``"YYYY-MM-DD"``). Returns ``None`` for "Max"/unknown/empty (no floor). Pure + unit-tested; the
    time-series accessors apply it as ``date >= ?`` so one global selector filters every chart."""
    if not anchor or preset in (None, "Max"):
        return None
    try:
        end = date.fromisoformat(anchor[:10])
    except ValueError:
        return None
    if preset == "YTD":
        return f"{end.year:04d}-01-01"
    days = {"1M": 30, "6M": 182, "1Y": 365, "5Y": 365 * 5}.get(preset)
    return (end - timedelta(days=days)).isoformat() if days else None


def assets() -> pd.DataFrame:
    return query("select symbol, asset_class from marts.dim_asset order by asset_class, symbol")


def asset_daily(symbol: str, start: str | None = None) -> pd.DataFrame:
    sql = (
        "select date, close, daily_return, vol_20d, ma_50 "
        "from marts.fct_asset_daily where symbol = ?"
    )
    params: tuple[str, ...] = (symbol,)
    if start:
        sql += " and date >= ?"
        params += (start,)
    return query(sql + " order by date", params)


def all_assets_daily(start: str | None = None) -> pd.DataFrame:
    """Long frame ``[symbol, asset_class, date, close, daily_return]`` for EVERY asset, windowed by
    ``start`` (``date >= start``) when given. One query, ordered by ``symbol, date`` — the
    cross-asset leaderboard, rebased-performance and correlation panels all derive from this.

    Unlike :func:`asset_daily` this omits the precomputed rolling features (``vol_20d``/``ma_50``):
    the cross-asset stats are computed FROM the windowed ``daily_return`` (period return,
    annualised vol, correlation), so the rolling features aren't needed here (the per-asset
    drill-down still uses :func:`asset_daily` for its sliced ``ma_50``/``vol_20d``)."""
    sql = "select symbol, asset_class, date, close, daily_return from marts.fct_asset_daily"
    params: tuple[str, ...] = ()
    if start:
        sql += " where date >= ?"
        params += (start,)
    return query(sql + " order by symbol, date", params)


def portfolio_windows() -> list[str]:
    """The backtest windows actually present in the marts, in canonical enum order.

    Drives the dashboard's window selector — only windows that have been computed are offered, so
    the picker shows one option until D6 lands the full three.
    """
    df = query("select distinct window_id from marts.fct_portfolio_returns")
    present = set(df["window_id"]) if not df.empty else set()
    return [w for w in windows.WINDOWS if w in present]


def portfolio_returns(
    window_id: str = windows.DEFAULT_WINDOW, start: str | None = None
) -> pd.DataFrame:
    sql = (
        "select strategy, date, daily_return, cumulative_return, drawdown, rolling_sharpe_252 "
        "from marts.fct_portfolio_returns where window_id = ?"
    )
    params: tuple[str, ...] = (window_id,)
    if start:
        sql += " and date >= ?"
        params += (start,)
    return query(sql + " order by strategy, date", params)


def portfolio_strategy_stats(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, sharpe, sharpe_lo, sharpe_hi, n_obs, n_boot, ci_pct "
        "from marts.fct_portfolio_strategy_stats where window_id = ? order by sharpe desc",
        (window_id,),
    )


def portfolio_strategy_pairs(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy_a, strategy_b, sharpe_diff, diff_lo, diff_hi, distinguishable "
        "from marts.fct_portfolio_strategy_pairs where window_id = ? "
        "order by strategy_a, strategy_b",
        (window_id,),
    )


def portfolio_attribution(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, symbol, contribution_to_return, contribution_to_risk "
        "from marts.fct_performance_attribution where window_id = ? "
        "order by strategy, contribution_to_return",
        (window_id,),
    )


def portfolio_regime_performance(window_id: str = windows.DEFAULT_WINDOW) -> pd.DataFrame:
    return query(
        "select strategy, regime, n_days, day_share, ann_return, ann_vol, ann_sharpe "
        "from marts.fct_portfolio_regime_performance where window_id = ?",
        (window_id,),
    )


def portfolio_ml_gate(
    window_id: str = windows.DEFAULT_WINDOW, start: str | None = None
) -> pd.DataFrame:
    sql = (
        "select date, forecast_skill, forecast_weight "
        "from marts.fct_portfolio_ml_gate where window_id = ?"
    )
    params: tuple[str, ...] = (window_id,)
    if start:
        sql += " and date >= ?"
        params += (start,)
    return query(sql + " order by date", params)


def portfolio_btc_effect() -> pd.DataFrame:
    """The paired BTC effect per strategy (cross-window; no window filter). Empty until the two
    2015 windows have been computed."""
    return query(
        "select strategy, sharpe_ex, sharpe_inc, sharpe_diff, diff_lo, diff_hi, distinguishable "
        "from marts.fct_portfolio_btc_effect order by sharpe_diff"
    )


def macro_ids() -> list[str]:
    df = query("select distinct series_id from marts.fct_macro_indicator order by series_id")
    return df["series_id"].tolist() if not df.empty else []


def macro_catalog() -> list[dict]:
    """The configured macro series with display metadata — ``[{id, label, category, units}, ...]``
    in config order. Drives the Macro tab's category grouping + friendly labels (the mart only
    stores the raw ``series_id``). Tolerates legacy entries missing category/units."""
    try:
        items = load_assets().get("macro", []) or []
    except Exception:  # noqa: BLE001 — a malformed/absent config must not crash the dashboard
        return []
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict) or "id" not in it:
            continue
        out.append(
            {
                "id": it["id"],
                "label": it.get("label", it["id"]),
                "category": it.get("category", "Other"),
                "units": it.get("units", ""),
            }
        )
    return out


def macro(series_id: str, start: str | None = None) -> pd.DataFrame:
    sql = "select date, value, change from marts.fct_macro_indicator where series_id = ?"
    params: tuple[str, ...] = (series_id,)
    if start:
        sql += " and date >= ?"
        params += (start,)
    return query(sql + " order by date", params)


def market_macro(start: str | None = None) -> pd.DataFrame:
    if start:
        return query("select * from marts.fct_market_macro where date >= ? order by date", (start,))
    return query("select * from marts.fct_market_macro order by date")


def model_metrics() -> pd.DataFrame:
    return query("select model, symbol, metric, value, trained_at from marts.model_metrics")


def ml_forecast() -> pd.DataFrame:
    return query("select * from marts.ml_forecast")


def regimes(symbol: str, start: str | None = None) -> pd.DataFrame:
    sql = "select date, vol_20d, regime from marts.fct_regime where symbol = ?"
    params: tuple[str, ...] = (symbol,)
    if start:
        sql += " and date >= ?"
        params += (start,)
    return query(sql + " order by date", params)


def latest_brief() -> pd.DataFrame:
    try:
        return query(
            "select created_at, engine, brief, data_date "
            "from marts.market_brief order by created_at desc limit 1"
        )
    except Exception:
        return query(
            "select created_at, engine, brief "
            "from marts.market_brief order by created_at desc limit 1"
        )


def pipeline_runs() -> pd.DataFrame:
    return query(
        "select source, rows, status, finished_at from raw.pipeline_runs "
        "order by started_at desc limit 12"
    )


def data_as_of() -> str:
    """The freshest markets date (max `date` in `fct_market_macro`), as an ISO string — `""` if no
    data. Drawn from a mart, so it is identical in live and snapshot mode (unlike `pipeline_runs`,
    whose `raw.pipeline_runs` is never snapshotted)."""
    # Cast to varchar in SQL so DuckDB yields a clean ISO date ("2026-06-23"); going through
    # pandas would coerce the DATE to a Timestamp and str() it as "... 00:00:00".
    df = query("select cast(max(date) as varchar) as d from marts.fct_market_macro")
    if df.empty or pd.isna(df["d"].iloc[0]):
        return ""
    return str(df["d"].iloc[0])


def is_sample_data() -> bool | None:
    """Provenance of the displayed markets data: `True` if ALL synthetic sample data (from
    `mmi seed`), `False` if ALL live (ingested), `None` if there's no data or the provenance is
    mixed/unrecorded. Read from `fct_asset_daily.source` (stamped `"sample"` by `mmi seed`, and
    with the source name by the extractors) so it stays honest in snapshot mode too, where
    `raw.pipeline_runs` is unavailable. Anything short of a clean all-sample or all-live set —
    a mixed set (a partial cron ingest leaving some symbols `"sample"`), OR any unrecorded
    (null/blank) source — is deliberately `None`: never claim "all sample" or "all live" unless
    every row genuinely is."""
    # In snapshot mode, fct_asset_daily may not have a 'source' column
    try:
        df = query("select distinct source from marts.fct_asset_daily")
    except Exception:
        return None  # snapshot mode — no source column
    if df.empty:
        return None
    col = df["source"]
    # Any null/blank source means a row's provenance is unrecorded — we can't claim "all X".
    has_unrecorded = bool(col.isna().any() or (col.dropna() == "").any())
    recorded = set(col.dropna()) - {""}
    if not recorded or has_unrecorded:
        return None  # no provenance signal, or some rows unrecorded
    if recorded == {"sample"}:
        return True  # every row is synthetic
    if "sample" not in recorded:
        return False  # every row is ingested/live
    return None  # mixed sample + live


def recession_risk(start: str | None = None) -> pd.DataFrame:
    """Estrella-Mishkin probit recession-probability time series.

    Returns ``(date, spread_10y_3m, recession_prob, model)`` — one row per date present in the
    mart. ``model`` is ``'10y_3m'`` when the canonical 10Y–3M spread was available, or
    ``'10y_2y_proxy'`` when 10Y–2Y was used as a fallback (e.g. synthetic seed data). Returns an
    empty DataFrame when the mart is not yet built.
    """
    base = "select date, spread_10y_3m, recession_prob, model from marts.fct_recession_risk"
    if start:
        return query(base + " where date >= ? order by date", (start,))
    return query(base + " order by date")


def macro_source_caption(is_sample: bool | None) -> str:
    """The honest source caption for the Macro tab, as a function of provenance — `""` when no
    caption should show. Live FRED data earns the FRED attribution; **sample** data must NOT be
    attributed to FRED (`mmi seed` synthesises the very same FRED series_ids, so a "Source: FRED"
    caption over them is a misattribution); mixed/unknown provenance makes no source claim. Kept
    pure (and unit-tested) because the `make ci` gate never renders the dashboard, so caption
    honesty can't be caught by the smoke test."""
    if is_sample is False:
        return "Source: FRED, Federal Reserve Bank of St. Louis · https://fred.stlouisfed.org/"
    if is_sample is True:
        return "⚠️ Synthetic sample data — not from FRED (live data is sourced from FRED)."
    return ""  # mixed / unknown provenance → make no source claim


# ---------------------------------------------------------------------------
# Source freshness
# ---------------------------------------------------------------------------

# Expected update frequency per FRED series (in days).
# Daily series should update within 3 business days; weekly within 10; monthly within 45.
_FREQUENCY_DAYS = {
    "DGS10": 3,
    "DGS2": 3,
    "DGS3MO": 3,
    "T10Y2Y": 3,  # daily Treasury yields
    "VIXCLS": 3,  # daily VIX
    "DCOILWTICO": 3,  # daily oil
    "DTWEXBGS": 5,  # daily dollar (发布稍晚)
    "ICSA": 10,  # weekly initial claims
    "NFCI": 10,  # weekly financial conditions
    "UNRATE": 45,  # monthly unemployment
    "CPIAUCSL": 45,  # monthly CPI
    "PCEPILFE": 45,  # monthly core PCE
    "FEDFUNDS": 45,  # monthly Fed funds
    "INDPRO": 45,  # monthly industrial production
    "PAYEMS": 45,  # monthly payrolls
    "M2SL": 45,  # monthly M2
    "WALCL": 45,  # monthly Fed balance sheet
    "UMCSENT": 45,  # monthly consumer sentiment
    "SAHMREALTIME": 45,  # monthly Sahm rule
    "RSAFS": 45,  # monthly retail sales
    "A191RL1Q225SBEA": 120,  # quarterly GDP
    "GFDEGDQ188S": 120,  # quarterly debt/GDP
}


def source_freshness() -> pd.DataFrame:
    """Per-series freshness status: latest observation date, expected interval, days stale.

    Returns a DataFrame with columns: series_id, latest_date, expected_days, days_since, status.
    Status is 'fresh', 'stale', or 'unknown' (no data or no frequency defined).
    """
    df = query(
        "select series_id, max(date) as latest_date "
        "from marts.fct_macro_indicator "
        "group by series_id"
    )
    if df.empty:
        cols = ["series_id", "latest_date", "expected_days", "days_since", "status"]
        return pd.DataFrame(columns=cols)

    today = pd.Timestamp.now().normalize()
    # US Federal Reserve holidays — FRED doesn't publish on these.
    # Source: federalreserve.gov/aboutthefed/holiday-calendar
    _fed_holidays = pd.to_datetime(
        [
            # New Year's Day
            "2024-01-01",
            "2025-01-01",
            "2026-01-01",
            "2027-01-01",
            # MLK Day (3rd Monday in January)
            "2024-01-15",
            "2025-01-20",
            "2026-01-19",
            "2027-01-18",
            # Presidents' Day (3rd Monday in February)
            "2024-02-19",
            "2025-02-17",
            "2026-02-16",
            "2027-02-15",
            # Memorial Day (last Monday in May)
            "2024-05-27",
            "2025-05-26",
            "2026-05-25",
            "2027-05-31",
            # Juneteenth
            "2024-06-19",
            "2025-06-19",
            "2026-06-19",
            "2027-06-19",
            # Independence Day
            "2024-07-04",
            "2025-07-04",
            "2026-07-04",
            "2027-07-04",
            # Labor Day (1st Monday in September)
            "2024-09-02",
            "2025-09-01",
            "2026-09-07",
            "2027-09-06",
            # Columbus Day (2nd Monday in October)
            "2024-10-14",
            "2025-10-13",
            "2026-10-12",
            "2027-10-11",
            # Veterans Day
            "2024-11-11",
            "2025-11-11",
            "2026-11-11",
            "2027-11-11",
            # Thanksgiving (4th Thursday in November)
            "2024-11-28",
            "2025-11-27",
            "2026-11-26",
            "2027-11-25",
            # Christmas
            "2024-12-25",
            "2025-12-25",
            "2026-12-25",
            "2027-12-25",
        ]
    )

    rows = []
    for _, row in df.iterrows():
        sid = row["series_id"]
        latest = pd.Timestamp(row["latest_date"])
        expected = _FREQUENCY_DAYS.get(sid)
        if expected is None:
            status = "unknown"
            days_since = None
        else:
            # Count business days excluding weekends AND US federal holidays.
            # A Thursday publication before Thanksgiving is only 1 business day old
            # on the following Monday, not 4 calendar days old.
            bdays = len(pd.bdate_range(latest, today)) - 1
            # Subtract holidays that fall on business days between latest and today
            holidays_in_range = sum(
                1 for h in _fed_holidays if latest < h <= today and h.weekday() < 5
            )
            days_since = max(bdays - holidays_in_range, 0)
            status = "fresh" if days_since <= expected else "stale"
        rows.append(
            {
                "series_id": sid,
                "latest_date": latest.strftime("%Y-%m-%d"),
                "expected_days": expected,
                "days_since": days_since,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def mart_summary() -> pd.DataFrame:
    """Row counts and latest dates for each mart table.

    Returns a DataFrame with: table, rows, latest_date.
    """
    tables = [
        "fct_asset_daily",
        "fct_macro_indicator",
        "fct_market_macro",
        "fct_regime",
        "fct_recession_risk",
        "model_metrics",
        "ml_forecast",
        "market_brief",
    ]
    rows = []
    for table in tables:
        try:
            df = query(f"select count(*) as n from marts.{table}")
            n = int(df["n"].iloc[0]) if not df.empty else 0
            # Get latest date if the table has a date column
            try:
                date_q = f"select cast(max(date) as varchar) as d from marts.{table}"
                date_df = query(date_q)
                latest = (
                    str(date_df["d"].iloc[0])
                    if (not date_df.empty and date_df["d"].iloc[0])
                    else None
                )
            except Exception:
                latest = None
            rows.append({"table": table, "rows": n, "latest_date": latest})
        except Exception:
            rows.append({"table": table, "rows": 0, "latest_date": None})
    return pd.DataFrame(rows)


def pipeline_summary() -> pd.DataFrame:
    """Summary of pipeline run status: source, last status, last run time, row count."""
    df = query(
        "select source, status, rows, finished_at, "
        "row_number() over (partition by source order by started_at desc) as rn "
        "from raw.pipeline_runs"
    )
    if df.empty:
        return pd.DataFrame(columns=["source", "last_status", "last_rows", "last_run"])
    latest = df[df["rn"] == 1].copy()
    col_map = {"status": "last_status", "rows": "last_rows", "finished_at": "last_run"}
    latest = latest.rename(columns=col_map)
    return latest[["source", "last_status", "last_rows", "last_run"]]


def asset_universe() -> pd.DataFrame:
    """Asset universe breakdown: symbol, asset_class, row count, date range."""
    return query(
        "select symbol, asset_class, count(*) as rows, "
        "cast(min(date) as varchar) as first_date, "
        "cast(max(date) as varchar) as last_date "
        "from marts.fct_asset_daily "
        "group by symbol, asset_class "
        "order by asset_class, symbol"
    )


def pipeline_flow() -> pd.DataFrame:
    """Data pipeline flow: source → stage → mart dependencies."""
    return pd.DataFrame(
        [
            {
                "source": "Yahoo Finance",
                "stage": "raw.asset_prices",
                "mart": "fct_asset_daily",
                "assets": "SPY, QQQ, VEA, TLT, TIP, GLD, BTC, EURUSD, GBPUSD",
            },
            {
                "source": "FRED",
                "stage": "raw.macro_series",
                "mart": "fct_macro_indicator",
                "assets": "21 series (yields, VIX, CPI, etc.)",
            },
            {
                "source": "FRED",
                "stage": "raw.macro_series",
                "mart": "fct_market_macro",
                "assets": "SPY + yields (ASOF join)",
            },
            {
                "source": "FRED",
                "stage": "raw.macro_series",
                "mart": "fct_recession_risk",
                "assets": "Yield-curve model",
            },
            {
                "source": "dbt",
                "stage": "fct_asset_daily",
                "mart": "fct_regime",
                "assets": "Vol terciles per symbol",
            },
            {
                "source": "dbt",
                "stage": "fct_asset_daily",
                "mart": "model_metrics",
                "assets": "ML skill metrics",
            },
            {
                "source": "dbt",
                "stage": "fct_asset_daily",
                "mart": "ml_forecast",
                "assets": "Return + vol forecasts",
            },
            {
                "source": "dbt",
                "stage": "fct_asset_daily",
                "mart": "market_brief",
                "assets": "AI narrative",
            },
        ]
    )
