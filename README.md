# Markets & Macro Intelligence 📈🌍

[![CI](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ci.yml)
[![Ingest](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ingest.yml/badge.svg)](https://github.com/mathsisbest/markets-macro-intelligence/actions/workflows/ingest.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![Cost](https://img.shields.io/badge/cost-%C2%A30%2Fmo-brightgreen)](#cost)

**Live demo: <https://macro-scope.streamlit.app>** &nbsp;·&nbsp; _free, public, auto-refreshing (see [docs/RUNBOOK.md](./docs/RUNBOOK.md))_

> **Desktop layout only.** The dashboard is optimised for a laptop or wider screen;
> narrow/mobile viewports are not supported.

> **Not investment advice.** This project is a demonstration of data engineering,
> analytics engineering, ML, and BI skills. Nothing here constitutes financial advice or a
> recommendation to buy, sell, or hold any security.

A **zero-cost, code-first** data platform that streams live markets + macro data, models it with
**dbt**, scores it with **ML**, explains it with a **GenAI** layer, and serves it through a
**Streamlit** dashboard. One repo, four engineering layers:
**Data Engineering · Analytics Engineering · ML/AI · BI.**

> Full design rationale, dataset choices and roadmap live in **[PLAN.md](./PLAN.md)**.

---

## Public-path overview

The deployed app runs entirely from **committed Parquet snapshots** — no MotherDuck token, no API
keys, no live database connection. The pipeline works as follows:

1. A daily GitHub Actions cron (`ingest.yml`) pulls prices, macro, and crypto from free-tier APIs
   into an **ephemeral local DuckDB**, runs `dbt build`, and calls `mmi snapshot` to export every
   mart to `data/public/*.parquet`.
2. The Action commits the Parquet files back to the repo. Streamlit Community Cloud detects the
   push and auto-redeploys.
3. The public Streamlit app starts with `MMI_SNAPSHOT_MODE=1`, opens the committed Parquet files
   in-process, and never touches a network connection — no MotherDuck, no API secrets required at
   serve time.

The heavy portfolio backtest (24 years × 3 windows × MVO + bootstrap) is too slow for a
60-minute Actions run, so it is run **locally** by the owner via `make refresh-full` and its
output committed directly. The daily cron preserves but never regenerates those files.

---

## Data sources & attribution

| Source | Data | Tier |
|---|---|---|
| [Yahoo Finance](https://finance.yahoo.com/) (`yfinance`) | Equities & ETF daily OHLCV (SPY, TLT, GLD, QQQ, EFA, TIP, IWM) | Free, no key |
| [FRED](https://fred.stlouisfed.org/) (St Louis Fed) | Macro series: DGS10, DGS2, DGS3MO, CPIAUCSL, UNRATE, FEDFUNDS, T10YIE, BAMLH0A0HYM2 | Free, key required |
| [CoinGecko](https://www.coingecko.com/) | Bitcoin daily price/volume | Free tier, key optional |
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
- **Sources:** Yahoo Finance (equities), CoinGecko (crypto), FRED (macro), World Bank (macro).
- **Everything in code:** charts, theme and layout are defined in Python — no Power BI/Tableau.

## Quickstart (runs with generated sample data, no API keys)

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
This repository is public for portfolio and demonstration purposes only; no reuse
license is granted. See [LICENSE](./LICENSE).
