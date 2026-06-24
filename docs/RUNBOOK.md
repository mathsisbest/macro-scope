# mmi Go-Live Runbook

Step-by-step GUI click-path to take the app from "code-complete" to "live on the internet."

For the automated deployment notes see [`.github/workflows/deploy-note.md`](../.github/workflows/deploy-note.md).

---

## Prerequisites

- You have push access to `mathsisbest/markets-macro-intelligence` on GitHub.
- You have free-tier accounts for the data sources you want to use (see below).
- Python 3.10+ is installed locally for the heavy refresh step.

---

## Step A — Add GitHub Actions secrets

The daily cron is keyless-safe (it exits 0 if keys are absent), but real macro and crypto data
require the optional keys. Add them once; the Action picks them up automatically on every run.

1. Open the repo on GitHub: https://github.com/mathsisbest/markets-macro-intelligence
2. Go to **Settings → Secrets and variables → Actions**.
3. Click **New repository secret** for each key you have:

   | Secret name | Where to get it | Required? |
   |---|---|---|
   | `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html (free) | Recommended |
   | `COINGECKO_API_KEY` | https://www.coingecko.com/en/api (free demo key) | Optional |
   | `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey (free tier) | Optional |

   Do **not** paste key values into chat, PR bodies, or commit messages — the secrets UI is the
   only safe path. Alternatively from a local terminal (never from a shared machine):
   ```bash
   gh secret set FRED_API_KEY
   gh secret set COINGECKO_API_KEY   # optional
   gh secret set GEMINI_API_KEY      # optional
   ```

4. Confirm: the Secrets list shows the three names (values are always hidden).

---

## Step B — Run the local heavy refresh and commit data/public

The portfolio backtest (24 years × 3 windows × MVO + 2 000 bootstrap draws) exceeds the
60-minute GitHub Actions cap, so it runs **locally** on your uncapped machine. The daily cron
**preserves** the committed output but never regenerates it — you must seed it once.

1. Make sure your `.env` holds real API keys:
   ```
   FRED_API_KEY=...
   COINGECKO_API_KEY=...   # optional
   GEMINI_API_KEY=...      # optional
   ```
2. From the repo root, run the full refresh (this may take 20–60 minutes):
   ```bash
   make refresh-full
   ```
   This command (once built in Wave 6 task D5) runs: ingest → dbt build → portfolio backtest →
   ML train → AI brief → `mmi snapshot`. It will write Parquet files to `data/public/`.

   While `make refresh-full` is being built, you can run the steps manually:
   ```bash
   mmi ingest
   mmi build
   mmi portfolio
   mmi ml
   mmi ai
   mmi snapshot
   ```
3. Check the output:
   ```bash
   ls -lh data/public/
   ```
   You should see one `.parquet` file per mart (e.g. `fct_asset_daily.parquet`,
   `fct_portfolio_returns.parquet`, `market_brief.parquet`, etc.).
4. Commit and push:
   ```bash
   git add data/public/
   git commit -m "chore: seed real data/public snapshot"
   git push
   ```

---

## Step C — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in (free account, supports private repos).
2. Click **New app**.
3. Fill in the form:
   - **Repository:** `mathsisbest/markets-macro-intelligence`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Expand **Advanced settings** and add the following secret (the only one the public app needs):
   ```
   MMI_SNAPSHOT_MODE = "1"
   ```
   Optionally also set:
   ```
   MMI_SNAPSHOT_DIR = "data/public"   # only needed if you move the snapshot
   ```
   Do **not** add any API keys here — the public app must run without them (Contract B).
5. Click **Deploy**. Streamlit builds the app from `requirements.txt` (which resolves to
   `.[dashboard]` — no ML or ingestion extras).
6. Wait for the build to complete (~2–3 minutes). Open the URL Streamlit provides and confirm:
   - The dashboard loads without errors.
   - The provenance badge shows **"sample"** if you deployed before the real-data seed, or
     **"live"** after Step B.
   - No "Source: FRED" caption appears on the macro tab when showing sample data.
7. Copy the URL (format: `https://<slug>.streamlit.app`). Paste it into the
   `README.md` live-demo line:
   ```markdown
   **Live demo: https://<slug>.streamlit.app**
   ```

---

## Step D — Allow the Action to push to main (branch protection)

If `main` has branch-protection rules that block direct pushes, the daily snapshot commit from
the Action will succeed locally but fail to push — you will see a "push rejected" error in the
Actions log.

To grant the exception:

1. Go to **Settings → Branches → Branch protection rules → main**.
2. Under **"Restrict who can push to matching branches"**, add the Actions bot identity, **or**
3. Under **"Allow specified actors to bypass required pull requests"**, add
   `github-actions[bot]`.
4. Save. Trigger a manual run to verify (see Step E).

If you prefer not to change branch protection, an alternative is to set up a deploy key with
write access and pass it as a secret — but the bot-exception approach above is simpler for a
solo repo.

---

## Step E — Verify the first daily cron

After Steps A–D are complete, verify the automation end-to-end:

1. Go to **Actions → Refresh public snapshot → Run workflow** (leave **full** unticked for the
   cheap daily path).
2. Watch the run (~3–5 minutes). It should end green.
3. Check that a new commit appeared on `main` with updated timestamps on `data/public/*.parquet`.
4. Open the Streamlit URL. The provenance badge should advance: the **"Data as of"** date should
   match the new snapshot.
5. Confirm the portfolio/brief Parquet files are **byte-identical** to your Step B seed (the
   daily run excludes the portfolio tag, so they must be preserved untouched).

---

## Checklist summary

- [ ] A: FRED_API_KEY secret added (+ optional COINGECKO / GEMINI)
- [ ] B: `make refresh-full` completed locally; `data/public/` committed and pushed
- [ ] C: Streamlit app deployed with `MMI_SNAPSHOT_MODE=1`; live URL captured and added to README
- [ ] D: Branch-protection exception granted (if applicable)
- [ ] E: First daily cron run verified green; data_as_of badge advancing; portfolio Parquets preserved

Once all five boxes are checked, close EPIC #51 and issue #50.
