# 3. Scheduled micro-batch instead of true streaming

**Status:** Accepted

**Context.** The brief asks to "stream" data at zero cost. True streaming infrastructure
(Kafka/Kinesis/Flink) is neither free at scale nor warranted by these data volumes (crypto
updates per-minute at most; macro is monthly).

**Decision.** Implement **scheduled, incremental, idempotent micro-batch** ingestion via
GitHub Actions cron — the pattern most real analytics platforms actually run. Incremental
watermarks + delete-then-insert give exactly-once-ish semantics; `raw.pipeline_runs` and dbt
source freshness provide observability.

**Consequences.**
- £0, simple, reliable, and inside the unlimited free Actions minutes/month for public repos.
- Not sub-second fresh — acceptable here. A genuine Kafka demo (e.g. Redpanda free tier) is
  documented as optional Phase-4 work for CV breadth, off the critical path.
