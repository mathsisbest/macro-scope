# Deploying the dashboard (free) — local-first architecture

> **Architecture:** the heavy portfolio backtest runs **locally** (no time cap) via
> `make refresh-full`; the owner commits the resulting Parquet snapshot. GitHub Actions' role is
> the **cheap daily refresh** of prices / macro / crypto / ML, which **preserves** the committed
> portfolio + brief Parquet (it never rebuilds them). The heavy backtest is never run in CI.
>
> ⚠️ **Current status (pre-go-live):** the scheduled cron in `ingest.yml` is **PAUSED** (both
> `schedule:` lines are commented out — see [#76](https://github.com/mathsisbest/macro-scope/pull/76)).
> Today, refreshes happen **manually** (the **Run workflow** button / `workflow_dispatch`, which
> still carries a `full` toggle) or **locally**. **Re-enabling the daily-cheap cron is a go-live
> step** (Step 3 below + [docs/RUNBOOK.md](../../docs/RUNBOOK.md)); the heavy refresh stays local
> regardless of cron state.
>
> For the full GUI click-path to go live, see [docs/RUNBOOK.md](../../docs/RUNBOOK.md).

---

## Step 1 — Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (private is fine).
2. Go to <https://share.streamlit.io>, click **New app**, pick this repo and `dashboard/app.py`.
3. In **Advanced settings → Secrets**, set just:
   ```
   MMI_SNAPSHOT_MODE = "1"
   ```
   That is the only secret the public app needs. It reads the committed Parquet snapshot in
   `data/public/*.parquet`; no MotherDuck token, no API keys, no LLM keys are required or
   permitted here (Contract B). Optionally set `MMI_SNAPSHOT_DIR` if you move the snapshot
   off the default `data/public`.
4. Click **Deploy**. Streamlit registers a webhook; every push to `main` — including a
   snapshot-refresh commit — auto-redeploys the app.

> The repo already ships a committed **sample** snapshot in `data/public/`, so the app renders
> immediately on deploy. While that sample snapshot is live, the provenance badge honestly reads
> **"sample"** (synthetic — not FRED/live); it flips to **"live"** only after Step 2.

---

## Step 2 — Seed the real portfolio + brief Parquet locally (required once)

The portfolio backtest (24 years × 3 windows × MVO + a 2 000-draw bootstrap) is too heavy for the
60-minute GitHub Actions cap, so it runs **locally on your machine** where there is no time limit.

**The daily cron preserves whatever is committed to `data/public/` but never creates the
portfolio/brief Parquet from scratch** — so you seed the real data once, here, before the daily
cron has anything to preserve.

```bash
# From the repo root, with real keys in .env:
make refresh-full
```

`make refresh-full` runs the full local pipeline in order:

```
mmi ingest → dbt build → mmi portfolio → dbt build → mmi ml → mmi ml-gate (STRICT) → mmi ai → mmi snapshot
```

The **`mmi ml-gate` STRICT** step sits between `mmi ml` and `mmi snapshot`: the HAR
realized-volatility model must clear the minimum skill threshold (OOS R² ≥ 0.10 **and**
QLIKE-ratio < 0.99 **and** ≥ 3/5 walk-forward folds) before a snapshot can be produced. If it
**fails**, the run exits non-zero and no snapshot is committed — and the honest response is **not**
to re-tune to pass: either ship with the baseline-only state active (the ML tab then truthfully
shows *"no demonstrated out-of-sample edge"*) or keep refining the features locally. The
margins/seed are fixed and never adjusted to flip a verdict.

After a successful run, confirm `data/public/` holds one `.parquet` per mart (including
`fct_portfolio_returns.parquet` and `market_brief.parquet`) and that each file is under the 2 MB
cap, then commit and push:

```bash
git add data/public/
git commit -m "chore(data): seed real public snapshot"
git push
```

The push auto-redeploys Streamlit, and the provenance badge flips to **"live"** once every source
in the snapshot is real.

> **To refresh the heavy backtest later:** re-run `make refresh-full` locally and recommit
> `data/public/`. The committed public artifact always uses the default `n_boot=2000` —
> `make refresh-full-fast` (low `n_boot`) is for local iteration only, never for the commit.

---

## Step 3 — Daily cron (re-enable at go-live; cheap refresh only)

> **Currently PAUSED.** Re-enabling the daily-cheap cron is a go-live step — do it **after** the
> Step 2 real-data seed so the first run has the real portfolio/brief Parquet to preserve.

At go-live, `ingest.yml` is switched to a **single daily-cheap schedule** (the weekly auto-schedule
is deleted so CI can never take the 60-minute-timeout path; the manual **Run workflow** /
`workflow_dispatch` button — with its `full` toggle — remains for on-demand use, but the real
heavy refresh is always the local `make refresh-full`, not the Actions `full` path). The daily run
executes the cheap path:

```
mmi ingest → dbt build --exclude tag:portfolio → mmi ml → mmi snapshot
```

What this does:

- Refreshes prices, macro series, crypto, and the ML rows in the snapshot.
- **Does not** run the portfolio backtest or regenerate the AI brief.
- **Preserves** any `fct_portfolio_returns.parquet` and `market_brief.parquet` already committed
  to `data/public/` — `mmi snapshot` exports only marts present in the ephemeral DuckDB, and the
  portfolio marts were excluded from the build, so they are absent from the DB and therefore left
  untouched on disk.

### Cron setup

The cron pushes the refreshed snapshot commit, and the data sources read these GitHub Actions
secrets (all optional — without them the keyless core still runs and `mmi ingest` exits 0):

| Secret name | Purpose | Required? |
|---|---|---|
| `FRED_API_KEY` | Real macro data (FRED) | Recommended |
| `GEMINI_API_KEY` | AI brief (else deterministic offline template) | Optional |

The job already has `contents: write` permission to push the snapshot commit. If `main` has
branch-protection rules that block Action pushes, add an exception for `github-actions[bot]` under
**Settings → Branches → Branch protection rules → main** (see
[docs/RUNBOOK.md](../../docs/RUNBOOK.md) Step D).

---

## MotherDuck is private-dev only

MotherDuck's fees addendum restricts free accounts to *internal business use* — not for delivering
the service to third parties — so it is deliberately **out of the public path**. It stays plumbed
for private local/dev use only (`MMI_MOTHERDUCK_DATABASE` + `MOTHERDUCK_TOKEN`). The public app and
the snapshot cron never touch it.

---

## Free-tier notes

- **Streamlit Community Cloud:** one app from a private repo, ~1 GB RAM — fine for this dataset.
- **GitHub Actions:** 2 000 free private-repo minutes/month. Once the daily cron is re-enabled,
  ~30 cheap runs/month at ~3–5 min each ≈ 90–150 min/month, well inside the free tier. The heavy
  portfolio backtest never runs in CI.
