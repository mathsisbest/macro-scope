# Deploying the dashboard (free)

## Streamlit Community Cloud (recommended, free, supports private repos)
1. Push this repo to GitHub (private is fine).
2. Go to https://share.streamlit.io , "New app", pick this repo + `dashboard/app.py`.
3. In **Advanced settings → Secrets**, set the same keys as `.env` — including
   `MMI_MOTHERDUCK_DATABASE` and `MOTHERDUCK_TOKEN` so the app reads the live MotherDuck store
   (plus FRED_API_KEY, COINGECKO_API_KEY, LLM_PROVIDER, GEMINI_API_KEY, ...).
4. Deploy. Streamlit sets a webhook, so every push auto-redeploys.

## Keeping data fresh — MotherDuck (no data in git)
The scheduled `ingest.yml` workflow writes to **MotherDuck** (the shared store); the dashboard
reads from it. Nothing is committed back to the repo.

Setup:
1. Create a free MotherDuck account and a token: MotherDuck UI → Settings → Access Tokens.
2. Add the token as a GitHub Actions **secret** named `MOTHERDUCK_TOKEN`
   (repo → Settings → Secrets and variables → Actions → New repository secret), or via
   `gh secret set MOTHERDUCK_TOKEN` from your terminal (don't paste it into chat).
3. Add the same token (and `MMI_MOTHERDUCK_DATABASE=mmi`) to Streamlit secrets.
4. In `ingest.yml`, uncomment the `schedule:` block to enable the 6-hourly refresh. Until then,
   trigger it manually via the Actions tab → Ingest → "Run workflow".

## ⚠️ MotherDuck free tier + public portfolio
The MotherDuck **fees addendum** restricts free accounts to *internal business use* — not for
delivering the service to third parties. That's fine while the repo/app are private. **Before
making the dashboard public**, switch the public app to one of: committed/generated **sample
data only**, static **Parquet** artifacts, a **screenshot/demo-only** page, or an
upgraded/changed backend.
(Pricing: https://motherduck.com/product/pricing/ · Fees: https://motherduck.com/fees-addendum/)

## Free-tier notes
- Streamlit Community Cloud: one app from a private repo, ~1 GB RAM — fine for this dataset.
- GitHub Actions: 2,000 free private-repo minutes/month; the 6-hourly cron stays well inside.
