# Markets & Macro Intelligence 📈🌍

[![CI](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ci.yml)
[![Ingest](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ingest.yml/badge.svg)](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ingest.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![Cost](https://img.shields.io/badge/cost-%C2%A30%2Fmo-brightgreen)](#cost)

A **zero-cost, code-first** data platform that streams live markets + macro data, models it with
**dbt**, scores it with **ML**, explains it with a **GenAI** layer, and serves it through a
**Streamlit** dashboard. One repo, four skill areas:
**Data Engineering · Analytics Engineering · ML/AI · BI.**

> Full design rationale, dataset choices and roadmap live in **[PLAN.md](./PLAN.md)**.

---

## What it does

```
free APIs ──> ingestion (DE) ──> DuckDB raw ──> dbt (AE) ──> marts
                                                              │
                              ┌───────────────────────────────┤
                              ▼                               ▼
                       ML forecast + regime            GenAI daily brief
                              └───────────────┬───────────────┘
                                              ▼
                                   Streamlit dashboard (BI)
```

- **Streaming, for free:** GitHub Actions cron does scheduled, incremental, idempotent
  micro-batch ingestion (the realistic, free-tier way to "stream").
- **Sources:** CoinGecko (crypto), Stooq/yfinance (equities & FX), FRED + World Bank (macro).
- **Everything in code:** charts, theme and layout are defined in Python — no Power BI/Tableau.

## Quickstart (runs with bundled sample data, no API keys)

```bash
# 1. install (editable) with dashboard + ml extras
make install-dev

# 2. seed a tiny synthetic dataset + build marts, then launch the dashboard
make demo

# open http://localhost:8501
```

Want live data? Copy `.env.example` → `.env`, add free API keys (FRED, CoinGecko, an LLM key),
then:

```bash
make ingest      # pull live data into DuckDB
make dbt-build   # transform raw -> marts
make ml          # train/score forecast + regime models
make ai          # generate the GenAI market brief
make dashboard   # serve
```

## Tech stack (all free tier)

| Layer | Tooling |
|---|---|
| Ingestion | Python · httpx · pydantic · pandas |
| Storage | DuckDB + Parquet (optional MotherDuck free) |
| Transform | dbt-core + dbt-duckdb |
| ML/AI | scikit-learn · numpy |
| GenAI | provider-agnostic → Gemini/Groq free (Claude optional) |
| BI | Streamlit + Plotly |
| Orchestration | GitHub Actions cron |
| Quality | ruff · pytest · mypy · pre-commit |

## Cost

**£0 / month.** See the breakdown and the honest "Claude API isn't free" note in
[PLAN.md §10](./PLAN.md#10-cost-breakdown--and-one-honesty-note).

## Repo layout

See [PLAN.md §6](./PLAN.md#6-repository-structure). TL;DR: `src/mmi/` (ingestion, ml, ai),
`transform/` (dbt), `dashboard/` (Streamlit), `.github/workflows/` (CI + cron).

## License

MIT — see [LICENSE](./LICENSE).
