# 5. Orchestration & scaling strategy (right-sizing + escalation triggers)

**Status:** Accepted

**Context.** The data is small (MBs — daily macro plus a handful of assets), the pipeline is linear
(ingest → dbt transform → ML → publish), the cadence is daily, it's a solo project, and the hard
constraint is £0. The obvious reviewer question is "why not Airflow?" — this ADR records the
right-sizing as a deliberate, defensible judgment rather than leaving it implicit. It extends
[ADR-0002](0002-duckdb-as-the-warehouse.md) (DuckDB as the warehouse) and
[ADR-0003](0003-scheduled-microbatch-over-streaming.md) (scheduled micro-batch over streaming).

**Decision.** Keep the right-sized modern small-data stack:
- **DuckDB** (single-node OLAP) for storage/compute;
- **dbt** for the medallion transform (staging → marts), tests, and the intra-pipeline DAG + lineage;
- **GitHub Actions cron** as the scheduler;
- **Parquet snapshot + Streamlit Community Cloud** for serving.

Do **not** adopt Airflow / Dagster / Spark / Kafka / a cloud warehouse for this workload.

**Rationale.** The data fits one machine many times over; the cadence is daily batch with no
real-time need (see ADR-0003); the budget is £0; the team is one person. DuckDB + dbt *is* the
correct modern tool at this scale — not a stand-in for a "real" stack. Heavier orchestration or
compute would be over-engineering and would break £0. Demonstrating that you know *when* to reach
for the big tools — and chose not to here on purpose — is the senior signal; bolting Airflow onto a
toy pipeline reads as the opposite.

**Tradeoffs accepted (be honest).**
- GitHub Actions is a *scheduler*, not a full *orchestrator* — no DAG branching, backfill,
  task-level retries/SLAs, or a run-history/lineage UI. Mitigated: the pipeline is linear, and dbt
  already provides the transform-layer DAG + lineage + tests.
- Loads are full-refresh, not incremental/watermarked (acceptable at this volume).
- Parquet-in-git for serving is a £0 hack, not a production object-store/warehouse pattern.

**Escalation triggers.** Adopt the heavier tool only when the constraint actually bites:

| Adopt | When |
|---|---|
| Airflow / Dagster / Prefect | many interdependent pipelines, real backfill/SLA/retry needs, or a team needing shared observability. NB: Airflow breaks £0 (always-on scheduler + webserver + metadata DB; managed Airflow is paid). Dagster is the more dbt-native / modern choice if/when this is wanted. |
| Spark / Dask | data that no longer fits one machine (~10s of GB+). |
| Kafka / streaming | genuine real-time event streams (explicitly out of scope per ADR-0003). |
| Snowflake / BigQuery / Postgres warehouse | scale, multi-user, or governance needs. NB: the dbt models are warehouse-portable — the same models run on Postgres/Snowflake — so the AE skill already transfers without adopting one now. |

**Consequences.** The stack choice is an explicit, defensible decision and the escalation path is
documented, so a reviewer sees right-sizing + judgment rather than an unexamined default. If
hands-on orchestration is later wanted for the portfolio, prefer a clearly-labelled **local Dagster
demo** of the same pipeline over bolting Airflow onto the prod path (keeps £0; keeps GitHub Actions
as the real scheduler).
