# Architecture

```
free APIs ──▶ ingestion (src/mmi/ingestion) ──▶ DuckDB raw schema
                                                    │
                                          dbt (transform/) builds
                                          staging ▶ intermediate ▶ marts
                                                    │
                        ┌───────────────────────────┼───────────────────────────┐
                        ▼                           ▼                            ▼
                 ml (src/mmi/ml)            ai (src/mmi/ai)               dashboard/
            forecast + regimes        GenAI market brief             Streamlit + Plotly
                        └───────────── all read/write marts ─────────────┘
                                                    ▲
                              GitHub Actions cron (ingest.yml) orchestrates the loop
```

## Layers
- **Ingestion (Data Engineering).** One `Extractor` per source enforcing
  `fetch → validate → load`. Loads are idempotent (delete-then-insert on natural keys) and
  audited in `raw.pipeline_runs`.
- **Transform (Analytics Engineering).** dbt-core + dbt-duckdb. Medallion layout
  (staging → intermediate → marts), schema + singular tests, source freshness, generated docs.
- **ML/AI.** scikit-learn forecasting with a leakage-free, walk-forward backtest and explicit
  baselines; volatility-regime labelling. Metrics persisted to `marts.model_metrics`.
- **GenAI.** Provider-agnostic LLM client (Gemini/Groq/Claude) writes a daily market brief;
  deterministic template fallback keeps it free and always-working.
- **BI.** Streamlit app; all charts/theme defined in code (Plotly).

## Storage
A single DuckDB file is the warehouse; the Python and dbt layers share it. Small enough to
live in the (private) repo or a free MotherDuck database.

## Orchestration
GitHub Actions `schedule:` cron runs the ingest→transform→ml→ai loop and commits refreshed
data back, which auto-redeploys the Streamlit app. This is "streaming" done the free,
free-tier-appropriate way: scheduled, incremental micro-batches.

See `docs/adr/` for the reasoning behind the key choices.
