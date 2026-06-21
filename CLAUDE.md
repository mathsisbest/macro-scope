# CLAUDE.md — project context & handoff

> Read this first. Full design detail is in **PLAN.md**; architecture decisions in **docs/adr/**.
> This file is the condensed brief so you know *what we're building and why* before changing code.

## Mission
A **zero-cost, code-first data platform** ("Markets & Macro Intelligence") that ingests live
markets + macro data, transforms it with dbt, scores it with ML, explains it with a GenAI layer,
and serves it through a Streamlit dashboard. It's a **portfolio project** for the owner
(GitHub `mathsisbest`) to showcase four skills in one coherent repo:
**Data Engineering · Analytics Engineering · ML/AI · BI.**

## Owner & constraints (do not violate)
- **£0 / $0 forever.** Free tiers only — no paid hosting, DBs, or APIs.
- **High-code, not BI tools.** No Power BI / Tableau; all charts/layout defined in Python.
- **Private GitHub repo.** Future-proof; deliberately uses GenAI.
- Owner is a product designer (SaaS/AI) in London — values clarity and good SWE hygiene.

## Key decisions (and why)
1. **Domain = Markets & Macro** (chosen on free-data availability):
   - Crypto via **CoinGecko** (free, 100 calls/min) → the genuine high-frequency "streaming" story.
   - Equities/ETFs/FX via **Stooq** (no key) + yfinance fallback → deep history for ML.
   - Macro via **FRED** (free key) + **World Bank** (no key) → analytical backbone.
   - **Sports betting is an OPTIONAL Phase-2 module** (PLAN §13), not core: The Odds API free
     tier (~500 credits/mo) is too thin to anchor streaming.
2. **"Streaming" = scheduled micro-batch** via GitHub Actions cron (2,000 free private-repo
   mins/mo). Incremental, idempotent loads. True Kafka is out of scope/cost (ADR-0003).
3. **Stack:** Python 3.10+, **DuckDB** (single-file warehouse) + Parquet, **dbt-duckdb**
   (medallion: staging→intermediate→marts), **scikit-learn**, **Streamlit + Plotly**.
4. **GenAI is provider-agnostic** (`src/mmi/ai/llm.py`): `LLM_PROVIDER` = gemini|groq|claude.
   Defaults to **free Gemini/Groq**; falls back to a deterministic template if no key.
   ⚠️ **The Claude API is metered/not free** — the owner's Claude subscription does NOT cover it.
   Keep the free default; Claude is an opt-in switch.

## Current state (verified in the scaffolding session)
- ✅ `ruff check` clean, `ruff format --check` clean.
- ✅ `pytest` → 5/5 pass, 72% coverage.
- ✅ Full pipeline ran end-to-end on synthetic sample data: `mmi seed` → marts → `mmi ml`
  → `mmi ai` (offline template brief). All marts populate.
- ⚠️ **dbt itself was NOT run yet** — only the Python SQL fallback (`mmi build` /
  `src/mmi/transform_fallback.py`) was exercised. **TODO: install `dbt-duckdb` and verify
  `cd transform && dbt build` + `dbt test` actually compile/pass.** The fallback mirrors the
  dbt marts so the demo runs without dbt, but dbt is the canonical transform layer.
- ⚠️ **Not pushed to GitHub.** It was scaffolded in an isolated sandbox with no GitHub auth.
  A local commit exists but the sandbox left stale `.git` lock files. **Start git fresh:**
  ```bash
  rm -rf .git && git init -b main && git add -A
  git commit -m "feat: initial scaffold — markets & macro intelligence"
  gh repo create mathsisbest/markets-macro-intelligence --private --source=. --remote=origin --push
  ```

## How to run
```bash
make install-dev   # editable install + ml, dashboard, dev extras
make demo          # seed sample data (+ build marts) and launch the dashboard
# live data path (needs free keys in .env — see .env.example):
make ingest && make dbt-build && make ml && make ai && make dashboard
```

## Conventions
- Package code under `src/mmi/` (installable, `mmi` CLI). No loose scripts.
- Typed config via `pydantic-settings` (`src/mmi/settings.py`); secrets via `.env` (gitignored)
  / GH Actions secrets / Streamlit secrets. Never commit keys.
- Ingestion: one `Extractor` per source (`fetch → validate → load`), idempotent upserts
  (delete-then-insert on natural keys), audited in `raw.pipeline_runs`.
- dbt: medallion layout + tests + source freshness; custom schema-name macro keeps schemas
  clean (`staging`, `marts`). Asset universe is declarative in `config/assets.yml`.
- ML: leakage-free features, **walk-forward** backtest, **explicit baselines** (honesty over
  leaderboard-chasing). Metrics persisted to `marts.model_metrics`.
- Lint/format: ruff (line length 100). Tests: pytest. Pre-commit configured.

## Repo map (see PLAN.md §6 for full tree)
`src/mmi/{ingestion,ml,ai,utils}` · `transform/` (dbt) · `dashboard/` (Streamlit) ·
`config/` · `tests/` · `.github/workflows/` (ci.yml + ingest.yml cron) · `docs/` (+ ADRs).

## Immediate next steps / roadmap
1. Push to GitHub (above), then the owner will review with **Codex** and bring back suggestions.
2. **Verify dbt** (`dbt build` + `dbt test`) and reconcile any drift vs the fallback marts.
3. Get free keys (FRED, CoinGecko, Gemini); run live `make ingest`; deploy to **Streamlit
   Community Cloud** (deploys from private repos; auto-redeploys on push). See
   `.github/workflows/deploy-note.md`.
4. Work PLAN §11 phases (DE → AE → ML → GenAI → polish) as clean per-skill PR sets.

## Likely review talking points (be ready to discuss/improve)
- **ML baseline:** on synthetic sample data the model *trails* the naive baseline — expected
  (no signal). On real data, re-evaluate; consider classification (direction) + proper CV,
  and don't oversell predictive power.
- **Data-in-git:** the cron commits the DuckDB binary back to the repo to feed Streamlit.
  Tidy but noisy; the documented cleaner option is **MotherDuck** (free) — consider switching.
- **Secrets & freshness:** ensure no keys leak; surface dbt source-freshness in the UI.
- **yfinance/Stooq** are unofficial — treat as best-effort; FRED/World Bank are the reliable core.
