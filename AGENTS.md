# AGENTS.md — project context & handoff

> Read this first. Full design detail is in **PLAN.md**; architecture decisions in **docs/adr/**.
> Keep this file lean — the hard rules + what isn't derivable from the repo. Push rationale, specs,
> and data-source detail to PLAN.md / docs and link them; don't inline them here.

## Mission
A **zero-cost, code-first data platform** ("Markets & Macro Intelligence"): live markets + macro →
dbt → ML → a GenAI brief → a Streamlit dashboard, spanning **DE · AE · ML/AI · BI**. Full rationale,
datasets, and roadmap in **PLAN.md**.

## Owner & project nature
- **£0 / $0 forever** — free tiers only; no paid hosting, DBs, or APIs.
- **High-code, not BI tools** — no Power BI / Tableau; all charts/layout in Python.
- **Public** GitHub repo (source-available, all rights reserved). Deliberately uses GenAI.
- Values clarity and good SWE hygiene.

## Dev workflow
- **Small, single-concern PRs** (~1–5 files), branch `pNN-slug`, structured body
  (concern / what changed / risk / `make ci` result / questions). Move fast.
- **`make ci` is the gate** (ruff, ruff format, mypy, seed, `dbt build`+tests, dashboard smoke,
  pytest) — run it locally before every PR and paste the result in the body. One-time: `make setup`
  (needs `brew install python@3.11`).
- **GitHub Actions runs the same gate on every PR** (`ci.yml` on `pull_request` to main; also
  `workflow_dispatch`). Public repo → unlimited free Actions minutes.
- Two GitHub Actions cron workflows — **`daily.yml`** (weekdays 06:00 UTC, refresh prices/macro and
  snapshot) and **`weekly.yml`** (Mon 04:00 UTC, full pipeline incl. ML, portfolio backtest, and
  GenAI brief) — commit refreshed `data/public/*.parquet` back to the repo. See `Never` before
  changing their cadence.

## Roles & session kickoff
Generic roles (Planner / Builder / Reviewer — see `~/.claude/velocity-playbook.md`), run as
**separate sessions** with a repo-only handoff (no chat relay) — the anti-rubber-stamp rule.
- **Plan:** Outline high-level goals, task breakdown, and technical rationale before starting implementation.
- **Build:** Implement <issue #N / task> in small, single-concern branches (`pNN-slug`). All changes must pass `make ci`.
- **Review (External / DeepSeek / Opencode):** The user performs adversarial review of PR branches using DeepSeek or other opencode models. Every review checks:
  (1) Unit tests for uncovered modules with adversarial edge-case depth,
  (2) `pytest --cov=mmi --cov-report=term-missing` proves test coverage target (>=70%) is met,
  (3) Extracted pure-functions pattern for dashboard/utility components,
  (4) Compliance with £0 cost, free-tier limits, and no main branch direct pushes.

## Key decisions (full rationale in PLAN.md + docs/adr/)
- **Stack:** Python 3.10+, **DuckDB** (local dev/CI), **dbt-duckdb** (medallion staging→marts),
  **scikit-learn**, **Streamlit + Plotly**. The **public deploy reads committed Parquet snapshots**
  (`data/public/`, snapshot mode); **MotherDuck** is an optional live store for private dev only,
  not the public path.
- **GenAI is provider-agnostic** (`src/mmi/ai/llm.py`): `LLM_PROVIDER` = gemini|groq|claude, default
  **free Gemini/Groq**, deterministic-template fallback with no key. The Claude API is metered and
  **not** covered by the owner's subscription — see `Never`.
- Data sources (Yahoo v8 / FRED / World Bank / **Shiller CAPE**), the micro-batch "streaming" model (ADR-0003), and
  the Sports-betting Phase-2 plan → **PLAN.md + docs/adr/**.
- **Per-symbol ML config** (PR #65): each asset has its own model/horizon/features selected by systematic
  sweep (192 configs). SPY uses Gradient Boosting at 10yr horizon with vol_macro + CAPE features (R²=+0.58);
  TLT uses LightGBM at 2yr (R²=+0.40); GLD uses GB at 1yr with short rolling window. See `_SYMBOL_ML_CONFIG`
  in `src/mmi/ml/pipeline.py` and the full sweep at `PLAN.md §7.3`.

## Current status — read the live sources, not this file
Status drifts, so it's not hand-maintained here: read **open issues**, `gh pr list`, recent
`git log`, and **PLAN.md §11 / issue #7**.

## How to run
```bash
make setup         # one-time: create .venv + install everything (needs `brew install python@3.11`)
make ci            # the local gate: lint, types, dbt build+tests, dashboard smoke, pytest
make demo          # seed sample data (+ build dbt marts) and launch the dashboard
# live data path (needs free keys in .env — see .env.example):
make ingest && make dbt-build && make ml && make ai && make dashboard
```

## Viewing the dashboard — local-first vs production
Always verify the dashboard **locally** during development and PR review; the deployed Streamlit Cloud app is for post-merge verification only.

```bash
make dashboard      # → http://localhost:8501 (add --server.headless to run quietly without browser pop)
```

**Why local-first during Build & Review:**
- **Instant feedback & logs:** Errors, stack traces, and stdout stream directly to terminal.
- **Pre-merge testing:** Tests unmerged PR branches (`pNN-slug`) locally before committing/merging (Streamlit Cloud only builds `main`).
- **No browser disruption:** Headless execution avoids popping OS browser windows during agentic coding or CI runs.

With no local `data/mmi.duckdb` and no MotherDuck token, `dashboard/snapshot_boot.py` auto-enables
snapshot mode, serving the committed real-data Parquet in `data/public/` keyless. Use `make dashboard`, **not** `make demo` (`demo` seeds *synthetic* sample data).

**Deployed App (`macro-scope.streamlit.app`):**
Use only for **post-merge verification** on `main` to confirm Streamlit Cloud auto-deployment succeeded.

## Conventions
- Package code under `src/mmi/` (installable, `mmi` CLI). No loose scripts.
- Typed config via `pydantic-settings` (`src/mmi/settings.py`). Secrets via `.env` (gitignored) /
  GH Actions / Streamlit secrets — see `Never`. **MotherDuck** (optional): enable via
  `MMI_MOTHERDUCK_DATABASE` + `MOTHERDUCK_TOKEN`, env only.
- Ingestion: one `Extractor` per source (`fetch → validate → load`), idempotent upserts
  (delete-then-insert on natural keys), audited in `raw.pipeline_runs`.
- dbt: medallion layout + tests + source freshness; a custom schema-name macro keeps schemas clean
  (`staging`, `marts`); asset universe is declarative in `config/assets.yml`. **dbt is the canonical
  transform layer**; the Python SQL fallback (`transform_fallback.py` / `mmi build`) is **demo-only**.
- ML: leakage-free features, **walk-forward** backtest, **explicit baselines** (honesty over
  leaderboard-chasing). Metrics → `marts.model_metrics`.
- Lint/format: ruff (line length 100). Tests: pytest. Pre-commit configured.

## Repo map (full tree in PLAN.md §6)
`src/mmi/{ingestion,ml,ai,utils}` · `transform/` (dbt) · `dashboard/` (Streamlit) · `config/` ·
`tests/` · `.github/workflows/` (`ci.yml` — gate on PRs; `daily.yml` + `weekly.yml` — scheduled snapshot refresh) ·
`docs/` (+ ADRs).

## Review focus
Project-specific watch-items live in **docs/REVIEW_GUIDE.md** (§7), loaded by `/review-pr`.

## Boundaries — the hard rules (single source of truth)
**Always (do without asking):** read any file, run `make ci` / tests, search the codebase; make the
smallest single-concern change that satisfies the task; fix a failing gate at its root cause.

**Ask first (stop and check):**
- A data **schema** / **dbt contract** / mart-shape change, a public-API change, or a migration.
- Adding a dependency, or **anything that spends money** (£0 project — no paid APIs/hosting/DBs).
- Deleting or rewriting a file you didn't create; touching auth or secrets.

**Never:**
- Merge your own work, or push to `main` directly.
- Author commits as anyone but **mathsisbest** (`33107428+mathsisbest@users.noreply.github.com`) —
  never let an unrelated work email author.
- Commit secrets/tokens or the **MotherDuck token** / `.env` contents; the token must never appear in
  a connection string, log, or the dashboard UI. Keys live in env / GH Actions / Streamlit secrets.
- Flip the LLM default to **Claude** — it's metered and not covered by the owner's subscription; the
  default stays free Gemini/Groq, Claude opt-in only.
- Change the `daily.yml` / `weekly.yml` cron cadence or add scheduled jobs without the owner's say-so.
- Open `macro-scope.streamlit.app` to view/verify the dashboard — it's share-only and pops a browser
  on whichever machine the session runs on; view locally via `make dashboard`.
- Suppress/skip a failing test, or weaken the gate, to push a change through.

## When you compact this session
Preserve verbatim: every modified file path; the gate command (`make ci`) + any one-off task
commands; any in-progress task and its next step (e.g. an open PR awaiting review/merge). Drop
exploratory reasoning you no longer need; mid-task near the limit, checkpoint before summarizing.
