# 2. DuckDB as the warehouse

**Status:** Accepted

**Context.** We need an analytical (OLAP) store that is free, requires zero infrastructure,
runs identically on a laptop and in CI, and is supported by dbt.

**Decision.** Use **DuckDB** as the analytical engine, shared by the Python layers and dbt (via
`dbt-duckdb`). Local DuckDB file for dev/CI/offline demo; **MotherDuck** free tier is the deployed
shared store for the scheduled cron + dashboard. The `.duckdb` binary is **not** committed to git
(superseding the earlier "commit data to the private repo" idea — see Update below).

**Consequences.**
- Zero cost, zero ops, fast columnar SQL, trivial reproducibility (`make demo`).
- Single-writer model: fine for scheduled batch jobs, not for high-concurrency writes.
- No binary in git: the cron writes to MotherDuck and the dashboard reads from it.

**Update (P0, owner-confirmed).** Deployed state lives in MotherDuck (free tier), not a committed
`.duckdb` file — this removes binary-history noise and the need for `contents: write` in the cron.
Note the MotherDuck free-tier **fees addendum** limits free accounts to internal use; before any
public portfolio deploy, switch to sample-data/Parquet/screenshots or an upgraded backend.
