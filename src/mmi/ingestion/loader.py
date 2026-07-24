"""Idempotent loading into DuckDB + a pipeline audit log.

Design goals (data-engineering best practice):
- **Idempotent**: re-running an extractor never duplicates rows (delete-then-insert on keys).
- **Observable**: every run is recorded in ``raw.pipeline_runs`` (rows, duration, status).
- **Incremental-ready**: ``watermark()`` lets extractors fetch only new data.
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone

import pandas as pd

from mmi.utils.db import init_schemas
from mmi.utils.logging import get_logger

log = get_logger(__name__)

_AUDIT_TABLE = "raw.pipeline_runs"

# Canonical schemas for the raw landing tables dbt reads as sources. Created empty up-front
# (idempotently) so a *fresh* DB always has dbt's sources even if an optional extractor fails or
# returns nothing on the first run — dbt then builds empty marts instead of erroring on a missing
# source. Columns mirror what the extractors load (so ``upsert ... BY NAME`` stays clean) and
# cover every column the staging models select.
_RAW_TABLES = {
    "raw.asset_prices": (
        "symbol VARCHAR, asset_class VARCHAR, date TIMESTAMPTZ, open DOUBLE, high DOUBLE, "
        "low DOUBLE, close DOUBLE, volume DOUBLE, daily_return DOUBLE, "
        "source VARCHAR, loaded_at TIMESTAMPTZ"
    ),
    "raw.macro_series": (
        "series_id VARCHAR, date TIMESTAMPTZ, value DOUBLE, source VARCHAR, loaded_at TIMESTAMPTZ"
    ),
    "raw.worldbank": (
        "indicator_id VARCHAR, country VARCHAR, date VARCHAR, value DOUBLE, "
        "source VARCHAR, loaded_at TIMESTAMPTZ"
    ),
    # Portfolio tables are landed by `mmi portfolio` (not an extractor). Pre-created so the dbt
    # portfolio marts always have their sources even on a fresh DB / before a backtest has run, and
    # so a degenerate single-strategy run (which yields no strategy pairs) still leaves an empty
    # table for dbt rather than a missing source.
    # `window` (Phase D) leads each portfolio table so re-running one window's backtest deletes and
    # re-inserts only that window's rows (the upsert keys are window-led in cmd_portfolio).
    "raw.portfolio_returns": (
        "window_id VARCHAR, strategy VARCHAR, date TIMESTAMP, daily_return DOUBLE, "
        "cumulative_return DOUBLE, loaded_at TIMESTAMPTZ"
    ),
    "raw.portfolio_strategy_stats": (
        "window_id VARCHAR, strategy VARCHAR, sharpe DOUBLE, sharpe_lo DOUBLE, sharpe_hi DOUBLE, "
        "n_obs BIGINT, n_boot BIGINT, ci_pct DOUBLE, block_days BIGINT, loaded_at TIMESTAMPTZ"
    ),
    "raw.portfolio_strategy_pairs": (
        "window_id VARCHAR, strategy_a VARCHAR, strategy_b VARCHAR, "
        "sharpe_a DOUBLE, sharpe_b DOUBLE, "
        "sharpe_diff DOUBLE, diff_lo DOUBLE, diff_hi DOUBLE, distinguishable BOOLEAN, "
        "loaded_at TIMESTAMPTZ"
    ),
    "raw.portfolio_attribution": (
        "window_id VARCHAR, strategy VARCHAR, symbol VARCHAR, contribution_to_return DOUBLE, "
        "contribution_to_risk DOUBLE, strategy_gross_return DOUBLE, loaded_at TIMESTAMPTZ"
    ),
    "raw.portfolio_ml_gate": (
        "window_id VARCHAR, date TIMESTAMP, forecast_skill DOUBLE, forecast_weight DOUBLE, "
        "loaded_at TIMESTAMPTZ"
    ),
    # Cross-window comparison (not per-window): the paired BTC effect, one row per strategy.
    "raw.portfolio_btc_effect": (
        "strategy VARCHAR, sharpe_ex DOUBLE, sharpe_inc DOUBLE, sharpe_diff DOUBLE, "
        "diff_lo DOUBLE, diff_hi DOUBLE, distinguishable BOOLEAN, "
        "n_obs BIGINT, n_boot BIGINT, ci_pct DOUBLE, loaded_at TIMESTAMPTZ"
    ),
}


def ensure_raw_tables(con) -> None:
    """Create the raw source tables (empty) if absent, so dbt always has its sources."""
    for table, schema in _RAW_TABLES.items():
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
    with contextlib.suppress(Exception):
        con.execute("ALTER TABLE raw.asset_prices ADD COLUMN IF NOT EXISTS daily_return DOUBLE")


def reset_portfolio_raw_tables(con) -> None:
    """Drop + recreate the portfolio landing tables with the current canonical schema.

    ``mmi portfolio`` re-lands these wholesale every run, so they carry no cross-run state — but a
    schema change (e.g. Phase D's new ``window_id`` column) cannot take effect on an *existing* DB
    via ``CREATE TABLE IF NOT EXISTS`` alone. Dropping and re-ensuring them makes the backtest
    self-healing on a stale DB, while still restoring empty tables so dbt always has its sources
    (including the degenerate no-strategy-pairs case). Only the wholesale-landed portfolio tables
    are touched; the incrementally-loaded data tables (asset_prices, etc.) are left intact.
    """
    for table in (t for t in _RAW_TABLES if t.startswith("raw.portfolio_")):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    ensure_raw_tables(con)


class DuckDBLoader:
    """Loads validated dataframes into the ``raw`` schema, idempotently."""

    def __init__(self, con) -> None:
        self.con = con
        init_schemas(con)
        ensure_raw_tables(con)
        self._ensure_audit_table()

    # --- audit ---------------------------------------------------------------
    def _ensure_audit_table(self) -> None:
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_AUDIT_TABLE} (
                run_id      VARCHAR,
                source      VARCHAR,
                rows        BIGINT,
                started_at  TIMESTAMP,
                finished_at TIMESTAMP,
                status      VARCHAR,
                message     VARCHAR
            );
            """
        )

    def start_run(self, source: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.con.execute(
            f"INSERT INTO {_AUDIT_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
            [run_id, source, 0, datetime.now(timezone.utc), None, "running", ""],
        )
        return run_id

    def finish_run(self, run_id: str, rows: int, status: str, message: str = "") -> None:
        self.con.execute(
            f"""UPDATE {_AUDIT_TABLE}
                SET rows = ?, finished_at = ?, status = ?, message = ?
                WHERE run_id = ?""",
            [rows, datetime.now(timezone.utc), status, message[:500], run_id],
        )

    # --- loading -------------------------------------------------------------
    def upsert(self, table: str, df: pd.DataFrame, keys: list[str]) -> int:
        """Delete-then-insert ``df`` into ``table`` keyed on ``keys``. Returns row count."""
        if df.empty:
            log.warning("upsert: empty dataframe for %s, skipping", table)
            return 0

        df = df.copy()
        if "loaded_at" not in df.columns:
            df["loaded_at"] = datetime.now(timezone.utc)

        self.con.register("_incoming", df)
        # Create the target table from the incoming schema if it does not exist.
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM _incoming LIMIT 0")

        # Idempotent: remove rows whose keys appear in the incoming batch, then insert.
        on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
        self.con.execute(
            f"DELETE FROM {table} t WHERE EXISTS (SELECT 1 FROM _incoming s WHERE {on})"
        )
        # BY NAME: match columns by name so a pre-created (canonical) table can't break on order.
        self.con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming")
        self.con.unregister("_incoming")

        n = len(df)
        log.info("upsert: %s rows -> %s", n, table)
        return n

    def watermark(self, table: str, ts_col: str) -> str | None:
        """Return the max value of ``ts_col`` in ``table`` (for incremental pulls)."""
        exists = self.con.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema || '.' || table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            return None
        val = self.con.execute(f"SELECT max({ts_col}) FROM {table}").fetchone()[0]
        return str(val) if val is not None else None
