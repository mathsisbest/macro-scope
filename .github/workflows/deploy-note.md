# Deploying the dashboard (free)

## Streamlit Community Cloud (recommended, free, supports private repos)
1. Push this repo to GitHub (private is fine).
2. Go to https://share.streamlit.io , "New app", pick this repo + `dashboard/app.py`.
3. In **Advanced settings → Secrets**, paste the same keys as `.env`
   (FRED_API_KEY, COINGECKO_API_KEY, LLM_PROVIDER, GEMINI_API_KEY, ...).
4. Deploy. Streamlit sets a webhook, so **every push auto-redeploys**.

## Keeping data fresh
The `ingest.yml` cron refreshes data and commits the DuckDB file back to the repo;
that push triggers a Streamlit redeploy with fresh data — no server needed.

### Notes / trade-offs
- Committing the `.duckdb` binary every run adds history noise. Cadence is set to every
  6 hours to stay tidy and inside the free Actions quota.
- **Cleaner alternative:** point both the pipeline and dashboard at **MotherDuck** (free
  500 MB). Then the cron writes to MotherDuck and the dashboard reads from it — no data in git.
- Free tier: one private-repo app; ~1 GB RAM. Plenty for this dataset.
