# 2. DuckDB as the warehouse

**Status:** Accepted

**Context.** We need an analytical (OLAP) store that is free, requires zero infrastructure,
runs identically on a laptop and in CI, and is supported by dbt.

**Decision.** Use **DuckDB** as the analytical engine, shared by the Python layers and dbt (via
`dbt-duckdb`). Local DuckDB file for dev/CI/offline demo; the **public deploy reads committed
Parquet snapshots** (`data/public/`). **MotherDuck** is an optional live store for private dev only,
not the public path. The `.duckdb` binary is **not** committed to git
(superseding the earlier "commit data to the private repo" idea — see Update below).

**Consequences.**
- Zero cost, zero ops, fast columnar SQL, trivial reproducibility (`make demo`).
- Single-writer model: fine for scheduled batch jobs, not for high-concurrency writes.
- No binary in git: the cron commits Parquet snapshots to data/public/ and the public dashboard reads from them.

**Update (P0, owner-confirmed).** Deployed state lives in committed `data/public/*.parquet` snapshots,
not a MotherDuck database — this removes the need for a shared DB, binary-history noise, and the
`contents: write` permission in the cron. MotherDuck remains an optional private dev store only.
Note the MotherDuck free-tier **fees addendum** limits free accounts to internal use; before any
public portfolio deploy, switch to sample-data/Parquet/screenshots or an upgraded backend.
