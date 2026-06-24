# Deploying the dashboard (free)

## Streamlit Community Cloud (recommended, free, supports private repos)
1. Push this repo to GitHub (private is fine).
2. Go to https://share.streamlit.io , "New app", pick this repo + `dashboard/app.py`.
3. In **Advanced settings → Secrets**, set just:
   ```
   MMI_SNAPSHOT_MODE = "1"
   ```
   That's it — the public app reads the committed Parquet snapshot in-process (`data/public/*.parquet`)
   and carries **no secrets**: no MotherDuck token, no API keys. (Optionally `MMI_SNAPSHOT_DIR` if you
   move the snapshot off the default `data/public`.)
4. Deploy. Streamlit sets a webhook, so every push (including the daily snapshot refresh) auto-redeploys.

## Keeping data fresh — scheduled Parquet snapshot (no secrets in the public app)
The scheduled `ingest.yml` workflow ("Refresh public snapshot") runs the pipeline against an **ephemeral
local DuckDB**, exports the marts to `data/public/*.parquet`, and **commits the Parquet** back to the repo.
The push auto-redeploys the Streamlit app. MotherDuck is **not** in this path.

It runs at **two cadences** because the portfolio backtest is a ~static, multi-decade computation that's
too heavy (and pointless) to re-run daily:
- **Daily (06:00 UTC):** cheap refresh of prices / macro / crypto / ML. Skips the portfolio backtest and
  the brief; their committed Parquet from the last weekly run is preserved untouched (the daily build
  `--exclude tag:portfolio` and `mmi snapshot` only exports marts present in the DB).
- **Weekly (Mon 04:00 UTC):** full refresh incl. `mmi portfolio` + the portfolio-grounded brief.

> **Seed the portfolio marts once before relying on the daily run:** the daily refresh preserves but never
> creates the portfolio/brief Parquet. On a fresh repo, run a full pass first via the Actions tab →
> "Refresh public snapshot" → "Run workflow" → tick **full** (or just wait for the first Monday).

Setup:
1. (Optional, for the keyed sources) add the data-source API keys as GitHub Actions **secrets** so the
   refresh folds them in: repo → Settings → Secrets and variables → Actions → New repository secret, or
   `gh secret set FRED_API_KEY` / `gh secret set COINGECKO_API_KEY` (don't paste keys into chat).
   Without them, the keyless core (Yahoo + World Bank) still lands and `mmi ingest` exits 0 (#51 scope 1).
   `GEMINI_API_KEY` / `GROQ_API_KEY` are also optional — the GenAI brief falls back to a deterministic
   template offline.
2. Trigger on demand any time via the Actions tab → "Refresh public snapshot" → "Run workflow"
   (tick **full** to force the heavy portfolio refresh).
3. The job needs `contents: write` (already set) to push the snapshot to `main`. If `main` has branch
   protection that blocks Action pushes, add an exception for the workflow (or relax it for the snapshot
   commit) — otherwise the build succeeds but the push fails.

## MotherDuck is private-dev only
MotherDuck's **fees addendum** restricts free accounts to *internal business use* — not for delivering the
service to third parties — so it is deliberately **out of the public path**. It stays plumbed for private
local/dev use only (`MMI_MOTHERDUCK_DATABASE` + `MOTHERDUCK_TOKEN`); the public app and the snapshot cron
never touch it.
(Pricing: https://motherduck.com/product/pricing/ · Fees: https://motherduck.com/fees-addendum/)

## Free-tier notes
- Streamlit Community Cloud: one app from a private repo, ~1 GB RAM — fine for this dataset.
- GitHub Actions: 2,000 free private-repo minutes/month. ~30 cheap daily runs (~3-5 min) + ~4 heavy weekly
  runs (the portfolio backtest, tens of minutes) stays well inside — that's the whole point of the split.
