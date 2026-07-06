# Macro Scope 📈🌍

### Markets & Macro Intelligence

[![CI](https://github.com/mathsisbest/macro-scope/actions/workflows/ci.yml/badge.svg)](https://github.com/mathsisbest/macro-scope/actions/workflows/ci.yml)
[![Weekly refresh](https://github.com/mathsisbest/macro-scope/actions/workflows/weekly.yml/badge.svg)](https://github.com/mathsisbest/macro-scope/actions/workflows/weekly.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![Cost](https://img.shields.io/badge/cost-%C2%A30%2Fmo-brightgreen)](#cost)

**Live demo: <https://macro-scope.streamlit.app>** &nbsp;·&nbsp; _free, public, auto-refreshing (see [docs/RUNBOOK.md](./docs/RUNBOOK.md))_

> **Responsive dashboard.** The Streamlit app is designed for laptop, tablet, and phone
> viewports, with horizontally scrollable controls where dense financial tables need them.

> **Not investment advice.** Nothing here constitutes financial advice or a
> recommendation to buy, sell, or hold any security.

A **zero-cost, code-first** data platform that streams live markets + macro data, models it with
**dbt**, scores it with **ML**, explains it with a **GenAI** layer, and serves it through a
**Streamlit** dashboard — an end-to-end pipeline spanning
**Data Engineering · Analytics Engineering · ML/AI · BI.**

> Full design rationale, dataset choices and roadmap live in **[PLAN.md](./PLAN.md)**.

---

## Public-path overview

The deployed app runs entirely from **committed Parquet snapshots** — no MotherDuck token, no API
keys, no live database connection. The pipeline works as follows:

1. Two GitHub Actions cron workflows (`daily.yml` / `weekly.yml`) pull prices, macro, and crypto
   from free-tier APIs into an **ephemeral local DuckDB**, run `dbt build`, and call `mmi snapshot`
   to export the marts to `data/public/*.parquet`. A **daily** run does the cheap refresh (prices/
   macro); a **weekly** run additionally runs ML, the portfolio backtest, and the GenAI brief.
2. The Action commits the Parquet files back to the repo. Streamlit Community Cloud detects the
   push and auto-redeploys.
3. The public Streamlit app reads those committed Parquet files in-process. It **auto-detects**
   snapshot mode when no live database is configured (the deploy case), so **no secrets are
   required at serve time** — no MotherDuck, no API keys, no network connection. (Set
   `MMI_SNAPSHOT_MODE=1` to pin it explicitly if you ever need to.)

Because this is a **public** repo, GitHub Actions gives free unlimited minutes and a 6-hour job cap,
so even the heavy portfolio backtest (24 years × 3 windows × MVO + bootstrap, `n_boot=2000`) runs in
the **weekly** Actions job — no laptop, no local data. The **daily** run is cheap (~3–5 min) and
preserves the committed portfolio + brief Parquet between weekly runs. (`make refresh-full` remains
available for an optional local run, but it isn't required.)

---

## Data sources & attribution

| Source | Data | Tier |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com/) (`yfinance`) | Equities, ETF, FX & BTC (BTC-USD) daily OHLCV | Free, no key |
| [FRED](https://fred.stlouisfed.org/) (St Louis Fed) | Macro series: DGS10, DGS2, DGS3MO, CPIAUCSL, UNRATE, FEDFUNDS, T10YIE, BAMLH0A0HYM2 | Free, key required |
| [World Bank](https://data.worldbank.org/) | GDP growth, current-account balance | Free, no key |

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
- **Sources:** Yahoo Finance (equities, FX, BTC daily), FRED (macro), World Bank (macro).
- **Everything in code:** charts, theme and layout are defined in Python — no Power BI/Tableau.

## Quickstart (runs with generated sample data, no API keys)

```bash
# 1. install (editable) with dashboard + ml extras
make install-dev

# 2. seed a tiny synthetic dataset + build marts, then launch the dashboard
make demo

# open http://localhost:8501
```

Want live data? Copy `.env.example` → `.env`, add free API keys (FRED, an LLM key),
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
| Storage | DuckDB (local dev/CI) · MotherDuck free tier (private dev only) |
| Transform | dbt-core + dbt-duckdb |
| ML/AI | scikit-learn · numpy |
| GenAI | provider-agnostic → Gemini/Groq free (Claude optional) |
| BI | Streamlit + Plotly |
| Orchestration | GitHub Actions cron |
| Quality | ruff · pytest · mypy · pre-commit |

## Cost

**£0 / month.** See the breakdown and the honest "Claude API isn't free" note in
[PLAN.md §10](./PLAN.md#10-cost-breakdown--and-one-honesty-note).

## Go-live runbook

See **[docs/RUNBOOK.md](./docs/RUNBOOK.md)** for the step-by-step GUI click-path to deploy the
live app (GitHub Actions secrets, Streamlit Community Cloud setup, branch-protection exception).

## Repo layout

See [PLAN.md §6](./PLAN.md#6-repository-structure). TL;DR: `src/mmi/` (ingestion, ml, ai),
`transform/` (dbt), `dashboard/` (Streamlit), `.github/workflows/` (CI + cron).

## License

**Source-available, not open source.** © 2026 mathsisbest — all rights reserved.
No reuse license is granted. See [LICENSE](./LICENSE).
