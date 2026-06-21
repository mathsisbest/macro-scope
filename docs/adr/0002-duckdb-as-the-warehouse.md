# 2. DuckDB as the warehouse

**Status:** Accepted

**Context.** We need an analytical (OLAP) store that is free, requires zero infrastructure,
runs identically on a laptop and in CI, and is supported by dbt.

**Decision.** Use **DuckDB** as a single-file warehouse, shared by the Python layers and dbt
(via `dbt-duckdb`). Data is small enough to commit to the private repo; **MotherDuck** (free
500 MB) is the optional cloud upgrade.

**Consequences.**
- Zero cost, zero ops, fast columnar SQL, trivial reproducibility (`make demo`).
- Single-writer model: fine for scheduled batch jobs, not for high-concurrency writes.
- Binary file in git adds history noise at high refresh cadence — mitigated by a modest cron
  schedule and the MotherDuck alternative.
