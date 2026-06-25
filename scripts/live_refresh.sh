#!/usr/bin/env bash
# live_refresh.sh — local heavy-refresh pipeline for mmi
#
# Orchestrates the full local pipeline in dependency order:
#   1. mmi ingest          — pull live data from free APIs
#   2. dbt build (first)   — staging → marts (all except portfolio)
#   3. mmi portfolio       — heavy walk-forward backtest (uncapped, uses real n_boot)
#   4. dbt build (second)  — rebuild marts including portfolio marts
#   5. mmi ml              — train + score HAR realized-vol + direction models
#   6. mmi ml-gate STRICT  — skill gate (exits non-zero if model does NOT clear the bar)
#                            This BLOCKS steps 7-8; a failing model can never produce a
#                            committed artifact. Omit --warn-only intentionally.
#   7. mmi ai              — generate the GenAI market brief (uses LLM_PROVIDER)
#   8. mmi snapshot        — export marts.* → data/public/*.parquet
#
# Usage:
#   make refresh-full              # full run with default n_boot=2000
#   make refresh-full-fast         # fast run with low n_boot for local tuning
#
# WARNING — n_boot note:
#   refresh-full-fast sets MMI_PORTFOLIO_N_BOOT to a LOW value (e.g. 200) for fast local
#   tuning / iteration. The committed public artifact MUST always use the default n_boot=2000.
#   NEVER commit a data/public snapshot produced by refresh-full-fast — the bootstrap
#   confidence intervals will be undersampled and statistically invalid for publication.
#
# Environment variables:
#   MMI_PORTFOLIO_N_BOOT  — bootstrap resamples for portfolio (default: 2000)
#                           Set to a low value by refresh-full-fast (NOT for committing)
#   MMI_DUCKDB_PATH       — path to local DuckDB file (default: data/mmi.duckdb)
#   All standard mmi env vars (FRED_API_KEY, COINGECKO_API_KEY, GEMINI_API_KEY, etc.)
#   are read from the environment / .env — this script does NOT source .env automatically.
#
# Dry-run note (ml-gate STRICT):
#   When run against keyless / sample data (no FRED_API_KEY / COINGECKO_API_KEY),
#   the HAR realized-vol model will train on synthetic data and is very unlikely to
#   clear the skill gate (oos_r2 >= 0.10 AND qlike_skill_ratio < 0.99 AND
#   folds_passed >= ceil(0.6*n_folds)). In that case `mmi ml-gate` exits non-zero
#   and the script halts BEFORE `mmi ai` and `mmi snapshot`. The committed data/public
#   snapshot is therefore never produced from a model that failed the skill bar.
#   To test the full pipeline on sample data, use `make ci` (which uses --warn-only
#   semantics implicitly via the seeded offline path) — not this script.
#
# Requires: bash >= 3.2, Python venv at .venv (run `make setup` first)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Pin the DuckDB path to an ABSOLUTE location so `mmi` and `dbt` use the SAME database. Without
# this, dbt resolves its profile default ('../data/mmi.duckdb') relative to the repo-root CWD ->
# one directory ABOVE the repo (e.g. .../Projects/data/mmi.duckdb), so it cannot find the DB that
# `mmi ingest` just wrote to <repo>/data/mmi.duckdb. Respect a caller-supplied override.
export MMI_DUCKDB_PATH="${MMI_DUCKDB_PATH:-$REPO_ROOT/data/mmi.duckdb}"

# Resolve Python and dbt from the project venv if present.
VENV="$REPO_ROOT/.venv"
if [[ -f "$VENV/bin/python" ]]; then
    PY="$VENV/bin/python"
    DBT="$VENV/bin/dbt"
else
    PY="python3"
    DBT="dbt"
fi

# Respect caller-supplied MMI_PORTFOLIO_N_BOOT; default to 2000 (the committed-artifact value).
: "${MMI_PORTFOLIO_N_BOOT:=2000}"
export MMI_PORTFOLIO_N_BOOT

echo "=== mmi live_refresh.sh ==="
echo "    REPO_ROOT           : $REPO_ROOT"
echo "    MMI_PORTFOLIO_N_BOOT: $MMI_PORTFOLIO_N_BOOT"
if [[ "$MMI_PORTFOLIO_N_BOOT" -lt 2000 ]]; then
    echo ""
    echo "WARNING: MMI_PORTFOLIO_N_BOOT=$MMI_PORTFOLIO_N_BOOT is below 2000."
    echo "         This run is for LOCAL TUNING ONLY. Do NOT commit the resulting"
    echo "         data/public snapshot — bootstrap confidence intervals will be"
    echo "         undersampled and statistically invalid for publication."
    echo ""
fi
echo ""

# Step 1 — ingest live data from free APIs
echo "--- Step 1/8: mmi ingest ---"
"$PY" -m mmi.cli ingest
echo ""

# Step 2 — first dbt build: staging → marts (non-portfolio marts)
echo "--- Step 2/8: dbt build (first pass) ---"
"$DBT" build --project-dir transform --profiles-dir transform --target dev
echo ""

# Step 3 — heavy portfolio backtest (walk-forward, uncapped)
echo "--- Step 3/8: mmi portfolio (n_boot=$MMI_PORTFOLIO_N_BOOT) ---"
"$PY" -m mmi.cli portfolio
echo ""

# Step 4 — second dbt build: rebuild all marts including portfolio marts
echo "--- Step 4/8: dbt build (second pass — including portfolio marts) ---"
"$DBT" build --project-dir transform --profiles-dir transform --target dev
echo ""

# Step 5 — train + score HAR realized-vol and direction models
echo "--- Step 5/8: mmi ml ---"
"$PY" -m mmi.cli ml
echo ""

# Step 6 — skill gate STRICT (no --warn-only): exits non-zero if model does not clear the bar.
#           This BLOCKS steps 7 and 8. A model that fails the skill bar can NEVER produce
#           a committed snapshot artifact.
echo "--- Step 6/8: mmi ml-gate (STRICT — blocks snapshot on failure) ---"
if ! "$PY" -m mmi.cli ml-gate; then
    echo ""
    echo "ERROR: mmi ml-gate STRICT failed — the HAR realized-vol model did not clear the"
    echo "       skill bar (oos_r2 >= 0.10 AND qlike_skill_ratio < 0.99 AND"
    echo "       folds_passed >= ceil(0.6*n_folds)). Steps 7 (mmi ai) and 8 (mmi snapshot)"
    echo "       were BLOCKED. No snapshot was written; data/public is unchanged."
    echo ""
    echo "       If running against sample/keyless data this is expected — see the dry-run"
    echo "       note at the top of this script."
    echo ""
    echo "       Next steps:"
    echo "         - Check model metrics: mmi ml-gate --symbol SPY"
    echo "         - Inspect logs for feature/fold diagnostics from mmi ml"
    echo "         - Iterate on features/hyperparameters locally, re-run refresh-full-fast"
    echo "         - Once cleared locally, run make refresh-full (n_boot=2000) to produce"
    echo "           the committed artifact"
    exit 1
fi
echo ""

# Step 7 — generate the GenAI market brief
echo "--- Step 7/8: mmi ai ---"
"$PY" -m mmi.cli ai
echo ""

# Step 8 — export marts.* → data/public/*.parquet
echo "--- Step 8/8: mmi snapshot ---"
"$PY" -m mmi.cli snapshot
echo ""

echo "=== live_refresh.sh: COMPLETE ==="
echo "    Snapshot written to data/public/ with n_boot=$MMI_PORTFOLIO_N_BOOT"
if [[ "$MMI_PORTFOLIO_N_BOOT" -lt 2000 ]]; then
    echo ""
    echo "WARNING: This snapshot was produced with n_boot=$MMI_PORTFOLIO_N_BOOT (< 2000)."
    echo "         Do NOT commit data/public — re-run 'make refresh-full' (default n_boot=2000)"
    echo "         before committing the public artifact."
fi
