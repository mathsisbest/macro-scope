# mmi go-live — consolidated plan (PLAN-ONLY)

> Status: **plan-only — nothing built, no branches.** This is the single source of truth consolidating
> four planning rounds: (1) contract-first deep-plan, (2) local-first + skill-gate delta after owner
> decisions, (3) evidence-based ML re-target to volatility, (4) verified-factor additions.
> Generated 2026-06-24. Supersedes the scattered task notes in EPIC #51 / issue #50.

---

## 1. Executive summary

mmi is feature-complete but **not live**: `data/public/` is empty on `origin/main`, so the public app
renders "No database yet," and PR #76 paused both refresh crons (the weekly FULL backtest exceeded the
60-min Actions cap — cancelled at 1h0m22s). This plan takes it to a **polished, live, honest** product.

Two owner decisions shape the architecture:

1. **Live refresh = local-first, daily-cron-only.** The heavy portfolio+ML backtest runs **locally**
   (uncapped) via `make refresh-full`; the owner commits the snapshot. GitHub re-enables **only** the
   cheap daily cron (preserves the committed portfolio/brief Parquet). The weekly schedule is **deleted**,
   so CI can never hit the 60-min timeout.
2. **ML go-live bar = minimum skill threshold, designed honestly.** A model must beat its baseline
   out-of-sample before the live snapshot is committed — never re-tuned to pass.

The ML target was re-pointed on **verified empirical evidence** (deep-research): from next-day SPY
direction (near-noise) to a **HAR-style next-week realized-volatility forecast** (the one target free
daily data can clear honestly). Verified factor work also adds a **yield-curve recession-risk panel**
(macro context) and a **bond-return honesty note** on TLT/TIP; a **12-month time-series-momentum overlay**
ships as a gated experiment. See §8 for the evidence basis.

**Honest 24h verdict:** all **code** can land in 24h with parallel builders; **live-on-real-data is gated
on the owner's single local `make refresh-full` run** (real keys + the heavy uncapped backtest) plus GUI
steps. Full UI polish is a stretch — deploy on the sample bootstrap first (it's de-gated), polish as a
fast-follow. See §6.

---

## 2. Locked contracts

Build against these verbatim. Changing one requires a contract-change PR that updates every consumer in
the same wave.

### Contract A — snapshot / Parquet layout, bootstrap recipe, size
- `mmi snapshot` exports the **whole** `marts` schema, one `<table>.parquet` per mart under
  `settings.snapshot_dir` (default `data/public`, env `MMI_SNAPSHOT_DIR`). No hand-maintained table list;
  never rename the `<table>.parquet` pattern. The `.duckdb` binary is never committed.
- **Bootstrap recipe mirrors `make ci` exactly**, including the load-bearing
  `DROP schema marts/staging cascade` before `dbt build` — so the committed `fct_asset_daily` is
  dbt-model-shaped, not the `transform_fallback` 5-mart shape.
- Snapshot stays **whole-schema** + a **fail-loud size cap** `MMI_SNAPSHOT_MAX_BYTES` (default 2_000_000,
  the pre-commit `--maxkb=2000` limit): export, sum bytes, exit non-zero if exceeded. The remedy for an
  oversized real snapshot is a **new downsampled dbt equity mart** the dashboard reads (WS-D9) — *not*
  excluding marts (the dashboard reads full-daily `fct_portfolio_returns`).
- A **committed sample bootstrap** (`source='sample'` → `is_sample_data()==True`) is the **last code
  artifact** (depends on every mart/metric/brief shape-changer) so the public app renders immediately and
  honestly. The **real** snapshot is produced **locally** by `make refresh-full` and committed by the owner.
- **Daily-cron preservation invariant:** a daily snapshot (marts present minus portfolio/brief) into a dir
  already holding portfolio/brief `.parquet` leaves those byte-identical. `data/public/` has exactly one
  code owner (the sample-bootstrap task) + the owner real-data seed.

### Contract B — public installability & data-source flags
- Public deploy sets **only** `MMI_SNAPSHOT_MODE=1` (optional `MMI_SNAPSHOT_DIR`). **No secrets** in the
  public app — no MotherDuck token, no API/LLM keys.
- `requirements.txt` stays exactly `.[dashboard]`. **No module-scope `sklearn`/`scipy`/`dbt`/
  `portfolio.compute` import may be transitively reachable from `app.py` or `data.py`** at import time
  (CI installs all extras, so only the clean-room import guard catches this).
- `query()` swallows **only** `duckdb.CatalogException` → empty frame; connection/auth/missing-column
  errors still surface.

### Contract C — `data.py` accessor API + UI honesty
- The existing accessors are frozen (signatures + returned columns). Render tasks **consume** them. New
  accessors (`recession_risk()`, and a vol-forecast read) are added **only** in their data/ML pillar task,
  never in a render task.
- **UI honesty (regression-locked):** macro source claims route only through
  `macro_source_caption(is_sample)`; the provenance badge routes only through `is_sample_data()` tri-state
  (`True`/`False`/`None`). Never hardcode "Source: FRED"; never upgrade `None` to "live."
- The ML tab carries an explicit scope caption; the recession panel and the not-cleared ML state carry
  their honest captions (see Contracts E, G).

### Contract D — marts schema touchpoints
- `marts.model_metrics` is **long** `(model, symbol, metric, value, trained_at)`. New metrics are **new
  rows**, never new columns; the `pipeline.py` persist loop is extended in the **same** PR that adds them.
- `ml_forecast` `(symbol, as_of, predicted_next_return, model)` — the vol forecast adds rows with a
  distinct `model` tag (e.g. `rv_har`); `fct_regime` labels stay `Low|Medium|High`; `market_brief` stays
  3 columns (see Contract G).
- **New mart `fct_recession_risk`** (append-only): `(date, spread_10y_3m, recession_prob, model)` — the
  Estrella-Mishkin probit output. Carries no new contract footguns; must be tagged correctly for the
  daily-vs-weekly snapshot cadence (it's a cheap daily mart, **not** `portfolio`-tagged).
- `fct_asset_daily` keeps `source` (drives `is_sample_data()`) and `open/high/low/close` (drives the
  Garman-Klass vol estimator). Windows/strategies enums unchanged.

### Contract E — ML: volatility headline target, return forecast, skill gate, non-goals
- **Return forecast model:** per-symbol GradientBoosting / LightGBM with Shiller CAPE + dividend/earnings
  yield features (`vol_macro` feature set). Horizon varies by asset (SPY 2520d, TLT 504d, GLD 252d). Strict
  walk-forward OOS with no lookahead. Metrics (R², IC, direction accuracy) persist to `marts.model_metrics`.
  The active configs were selected via a systematic sweep of 192 combinations (3 symbols × 6 feature sets ×
  2 models × 7 horizons) — see `data/ml_full_sweep.csv` and `PLAN.md §7.3`.
- **Headline certified volatility model = forward next-week (5-trading-day) realized-volatility forecast for SPY**,
  HAR-style. Vol proxy = **Garman-Klass** range estimator from OHLC; features = HAR cascade (1d/5d/22d) +
  macro (yield curve, rate level). Walk-forward `TimeSeriesSplit(5)`, leakage-free (features use only past
  data; label = forward realized vol).
- **Metrics** (appended as long rows, `model='rv_har'`): `oos_r2` (Mincer-Zarnowitz vs realized), `qlike`,
  `baseline_qlike`, `qlike_skill_ratio`, `n_folds`, `folds_passed`. Honest baseline = **persistence /
  EWMA (RiskMetrics λ=0.94)**.
  - **QLIKE vol floor:** QLIKE floors realised vol at `_VOL_FLOOR` (= 0.002 daily ≈ 3.2% annualised)
    before squaring, applied **identically to the model and the baseline**, so a near-flat
    Garman-Klass day can't send the predicted/realised variance ratio to ~1e6 and dominate the loss.
    Economically motivated and **fixed — never tuned to clear the gate**.
- **Go-live skill gate** (one pure helper `skill_verdict()` in `src/mmi/ml/skill_gate.py`, read by the CLI
  gate, the dashboard badge, and the brief): `cleared` = `oos_r2 ≥ 0.10` AND `qlike_skill_ratio < 0.99`
  AND `folds_passed ≥ ceil(0.6·n_folds)` AND `n_obs ≥ 250`. Fixed module constants, **never re-tuned to
  pass**. Kept **out** of `make ci` (sample data has no real edge).
  - **`n_obs` semantics (locked-holdout change):** since the locked-holdout work, `n_obs` is the **CV-sample
    (dev) count** — the rows the walk-forward CV actually trained/scored on — **not** the full valid count.
    The dev portion is everything before the carved holdout, minus the last `horizon` rows whose forward
    target window falls outside dev (they get a NaN label and drop, exactly as at the series end). The gate's
    `n_obs ≥ 250` check is therefore evaluated on the dev count; on real data (≫ 250 obs) this is unaffected,
    and the check can only flip versus the old full-count behaviour in the narrow **~250–311-row band**
    (where carving ~20% drops the count below 250). Real-data runs are far above this band.
- **Locked holdout = honest extra OOS readout, REPORTED not GATED.** Both the vol (`rv_har`) and direction
  (`return_gb`) models carve the **last** `min(252, ⌊0.2·n⌋)` time-ordered rows as a locked holdout,
  final-fit on dev only, and score it once — emitting `holdout_*` long rows (vol: `holdout_oos_r2`,
  `holdout_qlike`, `holdout_qlike_skill_ratio`, `holdout_n_obs`; direction: `holdout_dir_acc`,
  `holdout_baseline_dir_acc`, `holdout_n_obs`). The forward **target is built per-slice** (dev and holdout
  separately) so no dev label reads a holdout-period value — the gated CV metrics stay leakage-free at the
  label level. `skill_verdict()` is **unchanged** and never reads `holdout_*`; the holdout is never used to
  tune the model, features, or thresholds. Skipped (no rows) when carving would leave `< 60` dev rows.
- **Escape hatch (honest):** if not cleared, do not overfit — commit with the
  "no demonstrated out-of-sample edge — baseline-only" UI/brief state; no beats/outperforms phrasing.
- **Next-day direction model** (`model='return_gb'`) is **retained, demoted** to an honestly-labelled
  "no demonstrated short-horizon edge" secondary — **not** the gate. Shares the regularized
   regularized factory + frozen `SEED` with the 21-day `mvo_ml` model, but each is a distinct
  prediction problem; `mvo_ml` keeps its own `fct_portfolio_ml_gate`.
- **Macro is volatility/regime context, never a return input.** The recession-risk panel is macro context.
- **Confirmed non-goals (honest, recorded):** macro→return-level/direction prediction (Goyal-Welch
  pattern); weekly reversal (Lehmann); cross-sectional momentum (only ~6 assets); CP forward-rate bond
  predictor (not constructible from DGS10/DGS2). The TSMOM overlay is a **portfolio overlay gated on
  bootstrap CI**, not the ML headline gate.

### Contract F — theme tokens & Streamlit config
- `theme.PALETTE` is the single colour source; existing keys + hexes **immutable**, additions only. Every
  figure goes through `style_fig()`; no inline hex. `.streamlit/config.toml [theme]` mirrors PALETTE (planned — not yet created);
  `[client] showErrorDetails` off (no stack traces in the public app). WCAG-AA documented.

### Contract G — brief record, timestamp, redaction
- `market_brief` stays **3 columns** `(created_at, engine, brief)`. `created_at` = wall-clock generation
  time; the **data date lives in the body**, and the app caption is reworded so the data date is never
  labelled "Generated." Offline deterministic template is the floor + only fallback. `redact()` is
  mandatory on every except→log path **and on every persisted brief body**. The brief reflects the
  vol-skill verdict via `skill_verdict()` (honest "no edge" copy when not cleared).

### Contract H — branch / PR / gate conventions
- One concern per PR (~1–5 files), branch `pNN-slug`, structured body with `make ci` pasted green, author
  `mathsisbest <33107428+mathsisbest@users.noreply.github.com>`. UI PRs additionally require a Streamlit
  render smoke + screenshot.
- **Live-sync workflow guaranteed not to fail CI:** only the cheap daily branch auto-runs (weekly schedule
  **deleted**, FULL only via manual `workflow_dispatch full=true`), under `timeout-minutes: 15`, with
  warn-only freshness and offline-safe `mmi ai`. `mmi ml-gate` is not in `make ci`.
- **Contended-file serialization** (one owner per file per wave) — see §4.

---

## 3. Task graph

Legend: **[code]** = code-PR · **[owner]** = owner-only GUI/local step. Workstream tags:
A=test/CI harness · B=UI · C=ML · D=live-refresh infra · E=macro evidence · F=portfolio experiment ·
G=#50 hygiene · H=snapshot/bootstrap · O=owner go-live.

### Wave 1 — independent foundations (no deps)
| id | ws | kind | task | owns |
|---|---|---|---|---|
| A1 | A | code | App-render smoke harness (`AppTest`: populated + empty-snapshot paths) + Makefile step | `scripts/dashboard_app_smoke.py`, `Makefile` |
| A2 | A | code | Clean-room `.[dashboard]`-only import guard | `scripts/public_import_smoke.py`, `Makefile` |
| B1 | B | code | Theme semantic tokens (additive) + WCAG-AA audit | `dashboard/theme.py` |
| C1 | C/G | code | Shared regularized `make_regressor()` factory + frozen `SEED` (#50 item 1) | `src/mmi/ml/forecast.py`, `src/mmi/ml/forecast_panel.py`, `tests/test_features.py` |
| C2 | C | code | Volatility features: Garman-Klass + HAR cascade + macro, behind a flag, leakage-checked | `src/mmi/ml/features.py`, `tests/test_features_vol.py` |
| E1 | E | code | Add `DGS3MO` to the keyless FRED ingest (enables canonical 10Y–3M) | `config/assets.yml` (or FRED config), `tests/test_ingestion_fred.py` |
| G1 | G | code | `base.run()` audit-mask fix (#50 item 2) | `src/mmi/ingestion/base.py`, `tests/test_cli_ingest.py` |
| G2 | G | code | `btc_aligned_returns` interior-NaN warning (#50 item 3) | `src/mmi/portfolio/compute.py`, `tests/test_portfolio_compute.py` |
| G3 | G | code | Delete dormant Stooq dead code (#50 item 5) | `src/mmi/ingestion/stooq.py`, `src/mmi/ingestion/__init__.py` |
| H0 | H/G | code | Seed a deterministic offline brief inside `mmi seed` (root-cause of empty AI tab) | `src/mmi/cli.py`, `tests/test_pipeline_offline.py` |
| H1 | H | code | Snapshot round-trip schema guard (asset/macro/portfolio accessors) | `tests/test_snapshot_roundtrip.py` |

### Wave 2 — config, polish, models, infra
| id | ws | kind | task | owns | deps |
|---|---|---|---|---|---|
| A1b | A | code | Wire app-render smoke into `.github/workflows/ci.yml` | `ci.yml` | A1 |
| B2 | B | code | `.streamlit/config.toml` mirroring PALETTE + hide error details (planned — not yet created) | `.streamlit/config.toml` | B1, A1 |
| B3 | B | code | Chart styling polish via `style_fig` | `dashboard/components/charts.py` | B1, A1 |
| B4 | B | code | KPI card refine (empty/oversized guards, theme delta colours) | `dashboard/components/kpi.py` | B1, A1 |
| B5 | B | code | App shell: hero + methodology expander + per-source attribution + "not investment advice" + favicon + per-tab empty states + **bond-return note (E/t61)** | `dashboard/app.py`, `dashboard/assets/favicon.png` | B1, A1 |
| C3 | C | code | HAR realized-vol model (`ml/volatility.py`) + persist `model='rv_har'` metrics/forecast | `src/mmi/ml/volatility.py`, `src/mmi/ml/pipeline.py`, `tests/test_volatility.py` | C1, C2 |
| C4 | C | code | Append skill-metric rows (vol: oos_r2/qlike/ratio/folds; direction: mae_skill_ratio/dir_acc_edge as honest secondary) | `src/mmi/ml/forecast.py`, `src/mmi/ml/pipeline.py` | C1 (after C3 on pipeline.py) |
| D1 | D | code | `MMI_PORTFOLIO_N_BOOT` env knob threaded through `cmd_portfolio` | `src/mmi/cli.py`, `tests/test_cli_snapshot.py` | H0 (cli.py order) |
| D7 | D | code | Snapshot manifest + per-file atomicity + daily-cron preservation test | `src/mmi/cli.py`(*serialize*), `tests/test_cli_snapshot.py` | H0 |
| E2 | E | code | `fct_recession_risk` mart (Estrella-Mishkin probit from 10Y–3M) + `recession_risk()` accessor | `transform/models/marts/fct_recession_risk.sql`, `dashboard/data.py`, `tests/...` | E1 |
| G4 | C | code | Brief honest "data as of" (data date in body) | `src/mmi/ai/narrative.py`, `tests/test_ai_narrative.py` | H0 |
| D2 | D | code | Re-enable **daily** cron only (`daily.yml`, weekdays 06:00 UTC); **delete** weekly schedule (`weekly.yml`, Mon 04:00 UTC); `timeout-minutes:15`; FULL only via dispatch | `.github/workflows/daily.yml`, `.github/workflows/weekly.yml` | — |
| DOC1 | D | code | README live-demo + `docs/RUNBOOK.md` (GUI go-live click-paths) | `README.md`, `docs/RUNBOOK.md` | — |

### Wave 3 — honesty surfaces, tests, experiment
| id | ws | kind | task | owns | deps |
|---|---|---|---|---|---|
| B6 | B | code | Portfolio tab collapsible sections | `dashboard/app.py`(*serialize*) | B5 |
| C5 | C | code | Pure `skill_verdict()` helper (vol-scoped) | `src/mmi/ml/skill_gate.py`, `tests/test_skill_gate.py` | C4 |
| C6 | C | code | `mmi ml-gate` CLI (blocks the live snapshot on skill failure) | `src/mmi/cli.py`(*serialize*), `tests/test_cli_ml_gate.py` | C5 |
| C7 | C | code | Honest escape-hatch UI/brief state when not cleared | `dashboard/components/charts.py`, `src/mmi/ai/narrative.py`(*serialize*), `tests/...` | C5 |
| C8 | C | code | Leakage re-check test (vol + direction) | `tests/test_forecast_leakage.py` | C4 |
| C9 | C/G | code | Noise-must-fail gate test + ML/regime edge cases (#50 item 6) | `tests/test_ml_edge_cases.py` | C5 |
| GB | C | code | Brief post-gen validate + body redact (+`llm-rejected` tag) | `src/mmi/ai/narrative.py`(*serialize*), `tests/test_ai_narrative.py` | G4 |
| H2 | H | code | Brief snapshot round-trip test | `tests/test_dashboard_snapshot_read.py` | H0 |
| H3 | H | code | ML-marts round-trip test (incl. `rv_har` rows) | `tests/test_ml_snapshot_roundtrip.py` | C4 |
| B7 | B | code | ML-tab honest **vol-skill** render (OOS R² vs persistence, regime prob, SPY scope) | `dashboard/components/charts.py`(*serialize*), `scripts/dashboard_smoke.py` | B3, C5 |
| E3 | E | code | Macro-tab recession-risk panel (probability + caveats: term-premium + 2022–23 false positive) | `dashboard/components/charts.py`(*serialize*) | E2, C5 |
| D3 | D | code | Keep FULL-branch `mmi ai` offline-safe (`weekly.yml`); daily path never invokes `mmi ai` (`daily.yml`) | `.github/workflows/daily.yml`(*serialize*), `.github/workflows/weekly.yml`(*serialize*) | D2 |
| D4 | D/G | code | Warn-only `dbt source freshness` in the daily cron + prune unenforced config (#50 item 4) | `daily.yml`/`weekly.yml`(*serialize*), `transform/models/staging/_sources.yml` | D3 |
| F1 | F | code | 12-month TSMOM overlay as a **gated experiment** strategy (must beat 1/N + buy-and-hold on bootstrap CI; labelled experiment otherwise) | `src/mmi/portfolio/compute.py`, `src/mmi/portfolio/backtest.py`, `dashboard/app.py`(*serialize*), `tests/...` | — |

### Wave 4 — render polish, snapshot cap, secrets
| id | ws | kind | task | owns | deps |
|---|---|---|---|---|---|
| GC | C | code | Brief deterministic ordering + distinct engine tags + facts TypedDict | `src/mmi/ai/narrative.py`, `src/mmi/ai/llm.py`, `tests/...` | GB |
| B8 | B | owner | Production-scale render check: ML + AI tabs against the real local DB in **both** cleared/not-cleared states (honesty, single-source agreement) | — | B6, B7, C7 |
| D6 | D | code | Whole-schema snapshot + fail-loud size cap `MMI_SNAPSHOT_MAX_BYTES` | `src/mmi/cli.py`(*serialize*), `tests/test_cli_snapshot.py` | D1 |
| O1 | O | owner | Add free Actions secrets (`FRED_API_KEY`, optional `GEMINI_API_KEY`) | — | DOC1 |

### Wave 5 — owner visual sign-off (sample)
| id | ws | kind | task | deps |
|---|---|---|---|---|
| O2 | O | owner | Local visual sign-off on assembled UI + honesty surfaces (sample data) | B8, B5, B4, B3, B7, E3 |

### Wave 6 — final code artifact
| id | ws | kind | task | owns | deps |
|---|---|---|---|---|---|
| H4 | H | code | Snapshot-contract closure in `cmd_snapshot` (knob + cap + `rv_har` rows reconciled) | `src/mmi/cli.py`(*serialize, last*), `tests/test_cli_snapshot.py` | D6 |
| H6 | H | code | Commit **sample** bootstrap snapshot (all marts incl. `fct_recession_risk` + `rv_har`; `source='sample'`; each <2MB) | `data/public/*.parquet` | C3,C4,C5,C7,G4,GB,GC,E2,D7,H4,A1 |
| D8 | D | code | Rewrite `deploy-note.md`: local-first FULL refresh, daily-cron-only | `.github/workflows/deploy-note.md` | D2 |
| D5 | D | code | `make refresh-full` (+ `refresh-full-fast`) local target with the skill gate inline | `Makefile`, `scripts/live_refresh.sh` | D1, C6 |

### Wave 7 — bootstrap test + deploy on sample
| id | ws | kind | task | deps |
|---|---|---|---|---|
| H5 | H | code | Bootstrap completeness + honesty + size test (every mart by exact name; `source='sample'`; <2MB) | H6 |
| O3 | O | owner | Deploy Streamlit Cloud (`MMI_SNAPSHOT_MODE=1` only) on the sample bootstrap; capture live URL | H6, B2, A2 |

### Wave 8 — owner local real-data seed (the binding constraint)
| id | ws | kind | task | deps |
|---|---|---|---|---|
| O4 | O | owner-local | Run `make refresh-full` locally (real keys, `n_boot=2000`); record `mmi ml-gate` verdict; if cleared commit real `data/public`; if not, commit with the honest baseline-only state | D5, D6, C6, H4, O1 |

### Wave 9 — verify live
| id | ws | kind | task | deps |
|---|---|---|---|---|
| O5 | O | owner | Verify the first **daily** cron preserves portfolio/brief byte-identical and the live `data_as_of` badge advances | O4, O3 |

### Wave 10 — close
| id | ws | kind | task | deps |
|---|---|---|---|---|
| O6 | O | owner | Tick + close #50; confirm + close EPIC #51 | (all code + O4 + O5) |
| D9 | D | code (conditional) | Downsampled equity-curve dbt mart — **only if** the real snapshot exceeds the size cap | D6, O4 |

---

## 4. Contended-file serialization (one owner per file per wave)

These files are touched by multiple tasks; the build orchestrator serializes them in this order so no two
tasks in the same wave own the same file:

- `src/mmi/cli.py`: **H0 → D1 → D7 → C6 → D6 → H4** (one per wave)
- `dashboard/app.py`: **B5 → B6 → (B8 owner check)** (+ F1's app.py touch serialized after B6)
- `dashboard/components/charts.py`: **B3 → B7 → E3** (one per wave)
- `src/mmi/ai/narrative.py`: **G4 → GB → C7 → GC**
- `src/mmi/ml/forecast.py`: **C1 → C4 → C8** ; `src/mmi/ml/pipeline.py`: **C3 → C4**
- `.github/workflows/daily.yml`: **D2** ; `.github/workflows/weekly.yml`: **D3 → D4**
- `data/public/`: **H6 (sample) → O4 (real)** — single owner only.

---

## 5. Risks

1. **The binding 24h constraint is O4** — only the owner can run `make refresh-full` locally (real keys +
   the heavy uncapped backtest). Until then `data/public` holds the sample bootstrap and the daily cron has
   nothing real to preserve.
2. **Skill bar may not clear even on volatility with daily-data proxies** — Corsi's R²~0.7 uses intraday
   RV; Garman-Klass daily proxy will be lower (~0.3–0.5 expected). Still clears `oos_r2 ≥ 0.10` comfortably
   in the literature, but if the real run doesn't, the honest escape hatch ships (no overfitting).
3. **Public-install regression is invisible to CI** (CI installs all extras). A2 + O3's boot log are the
   only guards against a stray `sklearn` import crashing Streamlit Cloud.
4. **Real snapshot vs 2 MB cap** — the 3-window × ~24yr daily `fct_portfolio_returns` may breach the
   pre-commit cap; D6 fails loud, D9 (downsampled mart) is the conditional remedy.
5. **Recession panel is in a live OOS failure** — the 2022–23 inversion produced no NBER recession through
   mid-2026; the panel must surface this + the term-premium critique, or it reads as overconfident.
6. **TSMOM is LOW-confidence** — ships labelled "experiment" and must clear the bootstrap-CI gate vs 1/N +
   buy-and-hold before any non-experimental claim; do not headline it.
7. **`timeout-minutes:15` on the daily job** is a judgement call (~3–5 min observed); loosen toward ~20 if
   a cold runner + API retries trip it — never toward 60 (that reopens the FULL-times-out hole).
8. **Branch protection**: if `main` is protected, the daily Action's push must be allowed (owner GUI).

---

## 6. 24-hour feasibility & recommended order

**All code lands in 24h** with parallel builders; **live-on-real-data is gated on O4** + GUI steps; full
UI polish is a stretch. Recommended order if review/merge capacity is the bottleneck:

1. **Unblockers first:** A1 + A1b (render smoke wired into Makefile *and* ci.yml) + A2 (import guard) —
   they de-risk every later UI/deploy gate.
2. **Hygiene + ML + brief + infra knobs:** the Wave-1/2 code (C/D/E/G/H foundations).
3. **Honesty surfaces + tests + experiment:** Wave 3.
4. **Shape closure → sample bootstrap (H6) → bootstrap test (H5).**
5. **Owner go-live:** O1 secrets → O3 deploy on sample (live URL immediately, honest sample badge) →
   **O4 local `make refresh-full` → commit real snapshot** → O5 verify daily cron → O6 close.

If the clock runs short: **deploy on the sample bootstrap (O3 is de-gated; Streamlit auto-redeploys)** so
the URL is live-and-honest, defer deeper UI polish (B6/B8) and the TSMOM experiment (F1) to a fast-follow,
and still close #50/#51 on the real-data path.

---

## 7. Deferred / explicit non-goals

**Deferred:** multi-symbol ML; dbt-managed `market_brief`; mobile-responsive layout; CI bootstrap
regeneration; CSV/PNG download affordances; ML hyperparameter tuning beyond regularization; daily-cron
skill-drift auto-handling (warn-only log recommended).

**Confirmed honest non-goals (evidence-based — §8):** macro→equity-return-level/direction prediction
(Goyal-Welch); weekly reversal (Lehmann); cross-sectional momentum (only ~6 assets); a Cochrane-Piazzesi
forward-rate bond predictor (not constructible from DGS10/DGS2, and OOS-fragile).

---

## 8. Evidence basis (deep-research, adversarially verified)

| Factor | Confidence | Finding | mmi use |
|---|---|---|---|
| Realized volatility (HAR) | **HIGH** (3-0) | OOS R² ~0.63–0.70 intraday; beats persistence; edge widens at 1-week (Corsi 2009; ABDL NBER w8160) | **Headline ML target** |
| Yield-curve → recession | **HIGH** ✅ | AUC 0.85–0.89 @12mo (SF Fed 2018); probit pseudo-R² ~0.29; live 2022–23 false positive | **Recession-risk panel** (context) |
| Bond-return predictability | **HIGH** ✅ | In-sample strong (Fama-Bliss ~15%, CP ≤0.44) but OOS-fragile (Thornton-Valente, Hodrick-Tomunen, Bauer-Hamilton) | **Honesty note** on TLT/TIP |
| Macro → return level | **HIGH** ✅ | Single predictors fail OOS; only combination/sign-restricted recover small gains (Campbell-Thompson, Rapach) | **Stays out** |
| Weekly reversal | MEDIUM | Real in-sample, turnover/microstructure-driven (Lehmann 1990) | **Stays out** |
| Time-series momentum (12mo) | **LOW** ⚠️ | Canonical Sharpe >1 but primary source inaccessible this pass | **Gated experiment** (F1) |
| Cross-sectional momentum | **LOW** ⚠️ | Source inaccessible; needs many assets | **Stays out** (~6 assets) |

Verification caveats: Goyal-Welch itself was paywalled (corroborated by its confirmed follow-ups); the
momentum papers were genuinely inaccessible (403 / image-only PDFs) → graded LOW, numbers directional only.
The volatility, recession-AUC, and bond-OOS-critique findings were directly fetched and confirmed.
