# CLAUDE.md — project context & handoff

> Read this first. Full design detail is in **PLAN.md**; architecture decisions in **docs/adr/**.
> This file is the condensed brief so you know *what we're building and why* before changing code.

## Mission
A **zero-cost, code-first data platform** ("Markets & Macro Intelligence") that ingests live
markets + macro data, transforms it with dbt, scores it with ML, explains it with a GenAI layer,
and serves it through a Streamlit dashboard — an end-to-end pipeline spanning
**Data Engineering · Analytics Engineering · ML/AI · BI.**

## Owner & constraints (do not violate)
- **£0 / $0 forever.** Free tiers only — no paid hosting, DBs, or APIs.
- **High-code, not BI tools.** No Power BI / Tableau; all charts/layout defined in Python.
- **Public GitHub repo** (source-available, all rights reserved). Deliberately uses GenAI.
- Values clarity and good SWE hygiene.
- **Commit identity:** author commits as `mathsisbest` (`33107428+mathsisbest@users.noreply.github.com`).
  Never let an unrelated work email author.

## Dev workflow
- **Small, single-concern PRs.** One concern per PR (~1–5 files), branch `pNN-slug`, with a
  structured body (concern / what changed / risk / `make ci` result / questions). Move fast.
- **`make ci` is the gate — run it locally before every PR; CI re-runs it on the PR.** `make ci`
  (ruff, ruff format, mypy, seed, `dbt build`+tests, dashboard smoke, pytest) is your local
  pre-flight — paste the result in the PR body. One-time setup: `make setup` (needs `brew install python@3.11`).
- **GitHub Actions runs the same gate on every PR** (`ci.yml` triggers on `pull_request` to main,
  mirroring `make ci`; still `workflow_dispatch`-able). Stays within the free private-repo tier.
  The scheduled MotherDuck **`ingest.yml` stays disabled** (cron commented) — enable only with the owner's say-so.

## Roles & session kickoff
Use the generic roles (Planner / Builder / Reviewer — see `~/.claude/velocity-playbook.md`). Run them
as **separate sessions** with a repo-only handoff (no chat relay) — the anti-rubber-stamp rule.
- **Build:** `Builder: implement <issue #N / task>` → one small single-concern PR (`make ci`, structured body).
- **Review:** `/review-pr <n>` → separate session; loads `docs/REVIEW_GUIDE.md`, posts the verdict via `gh pr review`.
- **Plan / architect:** `Planner: <question>` → read-only strategy/Q&A, **no commits**; hands specs to the builder via the repo.

## Key decisions (full rationale in PLAN.md + docs/adr/)
- **Stack:** Python 3.10+, **DuckDB** (local dev/CI) + **MotherDuck** free tier (deployed),
  **dbt-duckdb** (medallion staging→marts), **scikit-learn**, **Streamlit + Plotly**.
- **GenAI is provider-agnostic** (`src/mmi/ai/llm.py`): `LLM_PROVIDER` = gemini|groq|claude,
  default **free Gemini/Groq**, deterministic-template fallback if no key.
  ⚠️ **The Claude API is metered/not free — the owner's subscription does NOT cover it.**
  Keep the free default; Claude is opt-in.
- Domain & data-source choices (CoinGecko / Yahoo v8 / FRED / World Bank), the micro-batch
  "streaming" model (ADR-0003), and Sports-betting Phase-2 → **see PLAN.md + docs/adr/**.

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
`config/` · `tests/` · `.github/workflows/` (ci.yml — runs the gate on PRs + manual; ingest.yml — scheduled refresh, disabled by default) · `docs/` (+ ADRs).

## Review focus
Project-specific watch-items live in **docs/REVIEW_GUIDE.md** (§7), loaded by `/review-pr`.
