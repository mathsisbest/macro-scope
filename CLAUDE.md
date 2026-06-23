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
- **Commit identity:** author commits as `mathsisbest` (`33107428+mathsisbest@users.noreply.github.com`).
  Never let an unrelated work email author.

## Dev workflow
- **Small, single-concern PRs.** One concern per PR (~1–5 files), branch `pNN-slug`, with a
  structured body (concern / what changed / risk / `make ci` result / questions). Move fast.
- **Local-first testing is the gate — NOT GitHub Actions.** Run `make ci` (ruff, ruff format,
  mypy, seed, `dbt build`+tests, dashboard smoke, pytest) before every PR and paste the result in
  the PR body. One-time setup: `make setup` (needs `brew install python@3.11`).
- **GitHub Actions is disabled by default** (`ci.yml` is `workflow_dispatch`-only) to preserve the
  free tier; enable a workflow only to run the scheduled MotherDuck ingest, with the owner's say-so.

## Roles & session kickoff (three-Claude)
Three separate Claude Code sessions, each a fresh `/clear`ed terminal. `CLAUDE.md` + `MEMORY.md`
auto-load, so a kickoff only needs the role + the task.
- **Claude 1 — Implementer.** Builds one small single-concern PR (branch `pNN-slug`, `make ci`,
  structured body). Kickoff: `Implementer: build <issue #N / task>.`
- **Claude 2 — Reviewer.** Separate session; runs `/review-pr <n>` (follows `docs/REVIEW_GUIDE.md`,
  posts the verdict via `gh pr review`). Interacts with the implementer only through the repo +
  PR comments — never chat relay. Kickoff: `/review-pr <n>`.
- **Claude 3 — Architect.** Read-only strategy / Q&A; **no commits**. Catches up via open issues +
  `gh pr list` + recent `git log` before answering, then hands specs to #1. Kickoff: `Architect: <question>.`

## Key decisions (and why)
1. **Domain = Markets & Macro** (chosen on free-data availability):
   - Crypto via **CoinGecko** (free, 100 calls/min) → the genuine high-frequency "streaming" story.
   - Equities/ETFs/bonds/commodities/FX/daily-crypto via the **Yahoo v8 chart** extractor (no key,
     adjusted close = total return) → deep daily history for ML. (Stooq retired — its free endpoint
     now returns a JS challenge, not CSV.)
   - Macro via **FRED** (free key) + **World Bank** (no key) → analytical backbone.
   - **Sports betting is an OPTIONAL Phase-2 module** (PLAN §13), not core: The Odds API free
     tier (~500 credits/mo) is too thin to anchor streaming.
2. **"Streaming" = scheduled micro-batch** via GitHub Actions cron (2,000 free private-repo
   mins/mo). Incremental, idempotent loads. True Kafka is out of scope/cost (ADR-0003).
3. **Stack:** Python 3.10+, **DuckDB** (local dev/CI) + **MotherDuck** free tier (deployed),
   **dbt-duckdb** (medallion: staging→intermediate→marts), **scikit-learn**, **Streamlit + Plotly**.
4. **GenAI is provider-agnostic** (`src/mmi/ai/llm.py`): `LLM_PROVIDER` = gemini|groq|claude.
   Defaults to **free Gemini/Groq**; falls back to a deterministic template if no key.
   ⚠️ **The Claude API is metered/not free** — the owner's Claude subscription does NOT cover it.
   Keep the free default; Claude is an opt-in switch.

## Current status — read the live sources, not this file
Status drifts, so it's NOT hand-maintained here. For where things stand, read: **open issues**,
`gh pr list`, recent `git log`, and the milestone plan in **PLAN.md §11 / issue #7**.

## How to run
```bash
make setup         # one-time: create the .venv + install everything (needs `brew install python@3.11`)
make ci            # the local gate: lint, types, dbt build+tests, dashboard smoke, pytest
make demo          # seed sample data (+ build dbt marts) and launch the dashboard
# live data path (needs free keys in .env — see .env.example):
make ingest && make dbt-build && make ml && make ai && make dashboard
```

## Conventions
- Package code under `src/mmi/` (installable, `mmi` CLI). No loose scripts.
- Typed config via `pydantic-settings` (`src/mmi/settings.py`); secrets via `.env` (gitignored)
  / GH Actions secrets / Streamlit secrets. Never commit keys. **MotherDuck:** enable via
  `MMI_MOTHERDUCK_DATABASE` + `MOTHERDUCK_TOKEN`; the token goes through env only and must never
  appear in a connection string, log, or the dashboard UI.
- Ingestion: one `Extractor` per source (`fetch → validate → load`), idempotent upserts
  (delete-then-insert on natural keys), audited in `raw.pipeline_runs`.
- dbt: medallion layout + tests + source freshness; custom schema-name macro keeps schemas
  clean (`staging`, `marts`). Asset universe is declarative in `config/assets.yml`.
  **dbt is the canonical transform layer**; the Python SQL fallback (`transform_fallback.py` /
  `mmi build`) is **demo-only** — it mirrors the marts so `make demo` runs without dbt.
- ML: leakage-free features, **walk-forward** backtest, **explicit baselines** (honesty over
  leaderboard-chasing). Metrics persisted to `marts.model_metrics`.
- Lint/format: ruff (line length 100). Tests: pytest. Pre-commit configured.

## Repo map (see PLAN.md §6 for full tree)
`src/mmi/{ingestion,ml,ai,utils}` · `transform/` (dbt) · `dashboard/` (Streamlit) ·
`config/` · `tests/` · `.github/workflows/` (ci.yml — manual; ingest.yml — scheduled refresh, disabled by default) · `docs/` (+ ADRs).

## Likely review talking points (be ready to discuss/improve)
- **ML baseline:** on synthetic sample data the model *trails* the naive baseline — expected
  (no signal). On real data, re-evaluate; consider classification (direction) + proper CV,
  and don't oversell predictive power.
- **No data in git:** the scheduled cron writes to **MotherDuck**; the `.duckdb` binary and any
  ingested data are never committed.
- **Secrets & freshness:** ensure no keys leak; surface dbt source-freshness in the UI.
- **Yahoo v8** is an unofficial endpoint — treat as best-effort; **FRED / World Bank** are the
  reliable core.
