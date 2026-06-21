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

## Dev workflow (two-Claude: implementer + reviewer)
- **Small, single-concern PRs.** One concern per PR (~1–5 files), branch `pNN-slug`, with a
  structured body (concern / what changed / risk / `make ci` result / questions). Move fast.
- **Local-first testing is the gate — NOT GitHub Actions.** Run `make ci` (ruff, ruff format,
  mypy, seed, `dbt build`+tests, dashboard smoke, pytest) before every PR and paste the result in
  the PR body. One-time setup: `make setup` (needs `brew install python@3.11`).
- **GitHub Actions is disabled** (`ci.yml` is `workflow_dispatch`-only) to preserve the free tier.
  Don't re-enable auto-runs without the owner's say-so.
- **Reviewer = a separate Claude session** that runs `/review-pr <n>` and follows
  `docs/REVIEW_GUIDE.md` (skeptical checklist + hard constraints; posts the verdict via
  `gh pr review`). Implementer and reviewer interact only through the repo + PR comments — never
  via chat relay.

## Key decisions (and why)
1. **Domain = Markets & Macro** (chosen on free-data availability):
   - Crypto via **CoinGecko** (free, 100 calls/min) → the genuine high-frequency "streaming" story.
   - Equities/ETFs/FX via **Stooq** (no key); yfinance fallback is roadmap → deep history for ML.
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

## Current state (verified in the scaffolding session)
- ✅ `ruff check` clean, `ruff format --check` clean.
- ✅ `pytest` → 5/5 pass, 72% coverage.
- ✅ Full pipeline ran end-to-end on synthetic sample data: `mmi seed` → marts → `mmi ml`
  → `mmi ai` (offline template brief). All marts populate.
- **dbt is the canonical transform layer** and runs in CI (`dbt build` + tests on seeded data,
  `--target dev`). The Python SQL fallback (`mmi build` / `src/mmi/transform_fallback.py`) is
  **demo-only** — it mirrors the dbt marts so `make demo` works without dbt; it is not canonical.
  (The first green CI run verifies dbt compiles/passes end-to-end.)
- ✅ **Pushed to GitHub:** `mathsisbest/markets-macro-intelligence` (private), default branch `main`.
- ✅ **Storage (owner-confirmed):** DuckDB locally (dev/CI) + **MotherDuck** free tier for the
  deployed/scheduled path; the `.duckdb` binary is **not** committed to git.

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
- ML: leakage-free features, **walk-forward** backtest, **explicit baselines** (honesty over
  leaderboard-chasing). Metrics persisted to `marts.model_metrics`.
- Lint/format: ruff (line length 100). Tests: pytest. Pre-commit configured.

## Repo map (see PLAN.md §6 for full tree)
`src/mmi/{ingestion,ml,ai,utils}` · `transform/` (dbt) · `dashboard/` (Streamlit) ·
`config/` · `tests/` · `.github/workflows/` (ci.yml — manual; ingest.yml — scheduled refresh, disabled by default) · `docs/` (+ ADRs).

## Immediate next steps / roadmap
1. ✅ Pushed + reviewed by Codex (issue #1). **P0 hygiene** (this branch): honest token-free CI
   (incl. dbt), MotherDuck storage plumbing, precise cron failure semantics, doc alignment.
2. Wire `MOTHERDUCK_TOKEN` (see deploy-note.md) + run the scheduled refresh; then P1: reconcile any
   dbt-vs-fallback drift, source-specific watermarks, macro ML features, dashboard polish.
3. **Capstone = issue #7** (portfolio backtesting / analytics / AI), sequenced **after** P1–P3 and
   built in slices 0–D (see PLAN §11). Phase 0 — Yahoo *adjusted-close* ingestion replacing the broken
   Stooq path — is the prerequisite and is in progress. The phased plan + critical review live on issue #7.
3. Get free keys (FRED, CoinGecko, Gemini); run live `make ingest`; deploy to **Streamlit
   Community Cloud** (deploys from private repos; auto-redeploys on push). See
   `.github/workflows/deploy-note.md`.
4. Work PLAN §11 phases (DE → AE → ML → GenAI → polish) as clean per-skill PR sets.

## Likely review talking points (be ready to discuss/improve)
- **ML baseline:** on synthetic sample data the model *trails* the naive baseline — expected
  (no signal). On real data, re-evaluate; consider classification (direction) + proper CV,
  and don't oversell predictive power.
- **Data-in-git:** RESOLVED in P0 — the cron writes to **MotherDuck** instead of committing the
  `.duckdb` binary; nothing data-related is pushed back to the repo.
- **Secrets & freshness:** ensure no keys leak; surface dbt source-freshness in the UI.
- **yfinance/Stooq** are unofficial — treat as best-effort; FRED/World Bank are the reliable core.
