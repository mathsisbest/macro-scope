"""DuckDB connection helpers.

DuckDB is an in-process OLAP database: zero infrastructure, a single file on disk,
and fast analytical SQL. Perfect for a free, reproducible data platform.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from mmi.settings import settings

# Schemas follow a medallion layout. dbt also targets `marts` / `staging`.
SCHEMAS = ["raw", "staging", "marts"]


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (and lazily create) the project DuckDB database."""
    db_path = Path(path or settings.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    return con


def init_schemas(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the medallion schemas exist."""
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
