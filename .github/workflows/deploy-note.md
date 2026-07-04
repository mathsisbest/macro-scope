# Deploying the dashboard (free) — CI-native architecture

> **Architecture:** the repo is **public**, so GitHub Actions has **free unlimited minutes + a
> 6-hour job cap**. Both the cheap daily refresh AND the heavy weekly portfolio backtest run in
> CI — no laptop required. The pipeline commits a Parquet snapshot to `data/public/` on every
> run; Streamlit Community Cloud auto-redeploys on the push.

---

## Step 1 — Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub.
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

> The repo ships a committed snapshot in `data/public/`, so the app renders immediately on
> deploy. The provenance badge reads **"live"** once every source in the snapshot is real data
> (not sample).

---

## Step 2 — Initial data seed (one-time)

The first time, trigger a **manual full run** to seed the portfolio backtest and brief:

1. Go to **Actions → Refresh public snapshot (scheduled) → Run workflow**
2. Check **Full refresh incl. the portfolio backtest**
3. Click **Run workflow**

This runs the full pipeline in CI (~5–15 min on the 6-hour cap):

```
mmi ingest → dbt build → mmi portfolio → dbt build → mmi ml → mmi ml-gate (STRICT) → mmi ai → mmi snapshot
```

The **`mmi ml-gate` STRICT** step sits between `mmi ml` and `mmi snapshot`: the HAR
realized-volatility model must clear the minimum skill threshold (OOS R² ≥ 0.10 **and**
QLIKE-ratio < 0.99 **and** ≥ 3/5 walk-forward folds) before a snapshot can be produced. If it
**fails**, the run exits non-zero and no snapshot is committed — and the honest response is **not**
to re-tune to pass: either ship with the baseline-only state active (the ML tab then truthfully
shows *"no demonstrated out-of-sample edge"*) or keep refining the features locally.

After the run completes, confirm `data/public/` holds one `.parquet` per mart (including
`fct_portfolio_returns.parquet` and `market_brief.parquet`). The push auto-redeploys Streamlit.

> **To refresh the heavy backtest later:** re-trigger a manual full run, or wait for the weekly
> Monday cron. The committed public artifact always uses `n_boot=2000`.

---

## Step 3 — Automated cron schedules

Both schedules are active in `ingest.yml`:

| Schedule | Path | What it does |
|---|---|---|
| **Weekdays 06:00 UTC** | Cheap daily | Refreshes prices, macro, crypto, ML, and regenerates the brief. **Preserves** the committed portfolio Parquet. |
| **Monday 04:00 UTC** | Full weekly | Full refresh including the portfolio backtest (`n_boot=2000`). |

Each schedule runs in its own concurrency slot (`snapshot-<schedule>`) so they never block each
 other.

### Daily (weekdays) — cheap path

```
mmi ingest → dbt build --exclude tag:portfolio → mmi ml → mmi ai → mmi snapshot
```

- Refreshes prices, macro series, crypto, ML, and the AI brief.
- **Does not** run the portfolio backtest.
- **Preserves** any `fct_portfolio_returns.parquet` and `market_brief.parquet` already committed
  to `data/public/` — `mmi snapshot` exports only marts present in the ephemeral DuckDB, and the
  portfolio marts were excluded from the build, so they are absent from the DB and therefore left
  untouched on disk.

### Weekly (Monday) — full path

```
mmi ingest → dbt build → mmi portfolio → dbt build → mmi ml → mmi ml-gate --warn-only → mmi ai → mmi snapshot
```

- Full refresh including the portfolio backtest.
- Runs with `timeout-minutes: 300` (well under the 6-hour public-repo cap).

### Secrets (all optional)

| Secret name | Purpose | Required? |
|---|---|---|
| `FRED_API_KEY` | Real macro data (FRED) | Recommended |
| `GEMINI_API_KEY` | AI brief (else deterministic offline template) | Optional |

The job has `contents: write` permission to push the snapshot commit. No branch protection is
configured; if it is added later, allow `github-actions[bot]` to push under
**Settings → Branches → Branch protection rules → main**.

---

## MotherDuck is private-dev only

MotherDuck's fees addendum restricts free accounts to *internal business use* — not for delivering
the service to third parties — so it is deliberately **out of the public path**. It stays plumbed
for private local/dev use only (`MMI_MOTHERDUCK_DATABASE` + `MOTHERDUCK_TOKEN`). The public app and
the snapshot cron never touch it.

---

## Free-tier notes

- **Streamlit Community Cloud:** one app, ~1 GB RAM — fine for this dataset.
- **GitHub Actions (public repo):** free unlimited minutes + 6-hour job cap. The daily cron
  runs ~22 times/month at ~1–2 min each ≈ 20–40 min/month. The weekly full backtest runs
  ~4 times/month at ~5–15 min each ≈ 20–60 min/month. Total: ~40–100 min/month, well within
  the free tier.
