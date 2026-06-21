"""DuckDB connection helpers.

DuckDB is an in-process OLAP database: zero infrastructure, a single file on disk,
and fast analytical SQL. Perfect for a free, reproducible data platform.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from mmi.settings import settings

# Schemas follow a medallion layout. dbt also targets `marts` / `staging`.
SCHEMAS = ["raw", "staging", "marts"]


def connect(path: Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the project database.

    Local DuckDB file by default (dev / CI / offline demo). When MotherDuck is configured
    (``MMI_MOTHERDUCK_DATABASE`` + ``MOTHERDUCK_TOKEN``) and no explicit ``path`` is given,
    connect to MotherDuck instead. The token is passed via the ``motherduck_token`` env var
    the extension reads — never embedded in a connection string we build, log or display.
    """
    if path is None and settings.use_motherduck:
        os.environ.setdefault("motherduck_token", settings.motherduck_token)
        # MotherDuck manages concurrency server-side; the local read_only flag
        # does not apply to ``md:`` connections.
        return duckdb.connect(f"md:{settings.motherduck_database}")
    db_path = Path(path or settings.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def init_schemas(con: duckdb.DuckDBPyConnection) -> None:
    """Ensure the medallion schemas exist."""
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
