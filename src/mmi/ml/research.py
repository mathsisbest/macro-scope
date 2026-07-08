"""ML research sweep — systematic comparison of model × features × horizon configs.

Run as a script: ``python -m mmi.ml.research``
Or import and call ``run_research(con)``.

The sweep is READ-ONLY against the DB — it never writes to model_metrics or ml_forecast.
Results are printed to stdout and returned as a DataFrame for manual review.
"""

from __future__ import annotations

import io
import itertools

import pandas as pd

from mmi.ml.volatility import train_and_backtest_vol
from mmi.settings import REPO_ROOT
from mmi.utils.db import connect
from mmi.utils.logging import get_logger

log = get_logger("ml.research")

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

MODELS = ["rv_har", "rv_ridge", "rv_lasso", "rv_gb", "rv_har_regime"]
FEATURE_SETS = ["vol", "vol_macro", "vol_rich"]
HORIZONS = [5, 10]
N_SPLITS_LIST = [5, 10]

# Ridge / Lasso alpha grid (only used for those models)
_ALPHA_GRID = [0.01, 0.1, 1.0]

# GB hyperparameter grid
_GB_GRID = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 3},
]


def _model_params(model_name: str) -> list[dict]:
    """Return the hyperparameter combos to sweep for a given model."""
    if model_name == "rv_har":
        return [{}]  # OLS has no hyperparams
    if model_name == "rv_ridge":
        return [{"alpha": a} for a in _ALPHA_GRID]
    if model_name == "rv_lasso":
        return [{"alpha": a} for a in _ALPHA_GRID]
    if model_name == "rv_gb":
        return list(_GB_GRID)
    return [{}]


def _load_macro_data(con) -> pd.DataFrame:
    """Load FRED series + Shiller CAPE, pivoted to a wide daily DataFrame.

    Returns one row per date with a column per series_id plus ``cape`` and
    ``excess_cape_yield``.  The feature builder ASOF-merges this onto trading dates.
    """
    try:
        df = con.execute(
            "select date, series_id, value from marts.fct_macro_indicator order by date"
        ).df()
        if df.empty:
            return pd.DataFrame()
        wide = df.pivot_table(index="date", columns="series_id", values="value", aggfunc="first")
        wide = wide.reset_index().sort_values("date")
        for col in wide.columns:
            if col != "date":
                wide[col] = wide[col].ffill()

        # Merge Shiller CAPE data — monthly, ASOF-merged onto daily grid
        cape_df = _load_cape_data()
        if not cape_df.empty:
            wide["date"] = pd.to_datetime(wide["date"])
            cape_df["date"] = pd.to_datetime(cape_df["date"])
            wide = pd.merge_asof(
                wide.sort_values("date"),
                cape_df.sort_values("date"),
                on="date",
                direction="backward",
            )
            # Forward-fill CAPE-derived features (monthly) to daily dates
            for col in ["cape", "excess_cape_yield", "div_yield", "earn_yield"]:
                if col in wide.columns:
                    wide[col] = wide[col].ffill()

        return wide
    except Exception:
        return pd.DataFrame()


def _load_cape_data() -> pd.DataFrame:
    """Download Shiller CAPE data and return as a (date, cape, excess_cape_yield) DataFrame.

    The source is Robert Shiller's public spreadsheet at Yale.  Data is monthly from 1881.
    Also extracts dividend yield and earnings yield computed from raw P/D/E columns.
    Returns an empty DataFrame if download or parse fails.
    """
    import urllib.request

    cache_path = REPO_ROOT / "data/raw/shiller_cape.parquet"
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read()
        df = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=None, skiprows=8)
    except Exception:
        log.warning("Failed to download Shiller CAPE data — skipping")
        return pd.DataFrame()

    # Columns: 0=date, 1=P, 2=D, 3=E, 12=CAPE, 16=excess_CAPE_yield
    out = df[[0, 1, 2, 3, 12, 16]].copy()
    out.columns = ["date_str", "P", "D", "E", "cape", "excess_cape_yield"]

    def _parse(d):
        if pd.isna(d):
            return pd.NaT
        parts = str(d).split(".")
        return pd.Timestamp(
            year=int(parts[0]),
            month=int(parts[1]) if len(parts) > 1 else 1,
            day=1,
        )

    out["date"] = out["date_str"].apply(_parse)
    for col in ["P", "D", "E", "cape", "excess_cape_yield"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Compute yields: dividend yield and earnings yield as decimals (not %)
    out["div_yield"] = out["D"] / out["P"]
    out["earn_yield"] = out["E"] / out["P"]

    out = out.dropna(subset=["cape"])
    cols = ["date", "cape", "excess_cape_yield", "div_yield", "earn_yield"]
    out = out[cols].sort_values("date").reset_index(drop=True)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
    except Exception:
        pass

    return out


def _load_asset_data(con, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Load per-symbol daily data for cross-asset vol features."""
    out = {}
    for sym in symbols:
        try:
            df = con.execute(
                "select date, daily_return from marts.fct_asset_daily "
                "where symbol = ? order by date",
                [sym],
            ).df()
            if not df.empty:
                out[sym] = df
        except Exception:
            pass
    return out


def run_research(
    con,
    symbol: str = "SPY",
    models: list[str] | None = None,
    feature_sets: list[str] | None = None,
    horizons: list[int] | None = None,
    n_splits_list: list[int] | None = None,
) -> pd.DataFrame:
    """Run the full combinatorial sweep and return a comparison DataFrame.

    Parameters
    ----------
    con:
        DuckDB connection.
    symbol:
        Asset to model (default SPY).
    models:
        Subset of MODELS to sweep (default: all).
    feature_sets:
        Subset of FEATURE_SETS to sweep (default: all).
    horizons:
        Subset of HORIZONS to sweep (default: all).
    n_splits_list:
        Subset of N_SPLITS_LIST to sweep (default: all).
    """
    models = models or MODELS
    feature_sets = feature_sets or FEATURE_SETS
    horizons = horizons or HORIZONS
    n_splits_list = n_splits_list or N_SPLITS_LIST

    macro_df = _load_macro_data(con)
    asset_dfs = _load_asset_data(con, ["GLD", "TLT"])

    results: list[dict] = []
    combos = list(itertools.product(models, feature_sets, horizons, n_splits_list))
    total = len(combos)

    for i, (model_name, feat_set, horizon, n_splits) in enumerate(combos, 1):
        # Skip vol_macro if macro data unavailable
        if feat_set == "vol_macro" and macro_df.empty:
            log.info("skip %s/%s: no macro data", model_name, feat_set)
            continue

        params_list = _model_params(model_name)
        for params in params_list:
            param_str = str(params) if params else ""
            log.info(
                "[%d/%d] %s feat=%s h=%d splits=%d %s",
                i,
                total,
                model_name,
                feat_set,
                horizon,
                n_splits,
                param_str,
            )

            try:
                metrics, _forecast = train_and_backtest_vol(
                    con,
                    symbol=symbol,
                    model_name=model_name,
                    feature_set=feat_set,
                    horizon=horizon,
                    n_splits=n_splits,
                    model_params=params if params else None,
                    macro_df=macro_df if feat_set in ("vol_macro", "vol_rich") else None,
                    asset_dfs=asset_dfs if feat_set in ("vol_macro", "vol_rich") else None,
                )
            except Exception as e:
                log.warning("FAILED %s/%s: %s", model_name, feat_set, e)
                continue

            if not metrics:
                log.info("skip %s/%s: small sample", model_name, feat_set)
                continue

            results.append(
                {
                    "model": model_name,
                    "feature_set": feat_set,
                    "horizon": horizon,
                    "n_splits": n_splits,
                    "params": param_str,
                    "oos_r2": metrics.get("oos_r2"),
                    "qlike_skill_ratio": metrics.get("qlike_skill_ratio"),
                    "folds_passed": metrics.get("folds_passed"),
                    "n_folds": metrics.get("n_folds"),
                    "n_obs": metrics.get("n_obs"),
                    "holdout_oos_r2": metrics.get("holdout_oos_r2"),
                    "holdout_qlike_skill_ratio": metrics.get("holdout_qlike_skill_ratio"),
                }
            )

    if not results:
        log.warning("no results — all combos failed or were skipped")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Sort by OOS R² descending (best first)
    df = df.sort_values("oos_r2", ascending=False).reset_index(drop=True)

    # Print summary
    print("\n" + "=" * 80)
    print("ML RESEARCH SWEEP RESULTS")
    print("=" * 80)
    print(f"Symbol: {symbol} | Combos tested: {len(df)}")
    print()

    # Skill gate thresholds
    R2_MIN = 0.10
    QLIKE_MAX = 0.99

    for _, row in df.iterrows():
        cleared = (
            row["oos_r2"] is not None
            and row["oos_r2"] >= R2_MIN
            and row["qlike_skill_ratio"] is not None
            and row["qlike_skill_ratio"] < QLIKE_MAX
            and row["folds_passed"] is not None
            and row["folds_passed"] >= 3
        )
        tag = " CLEARED" if cleared else ""
        print(
            f"  {row['model']:12s} | {row['feature_set']:10s} | h={row['horizon']} "
            f"| splits={row['n_splits']} | R²={row['oos_r2']:.4f} "
            f"| QR={row['qlike_skill_ratio']:.4f} "
            f"| folds={row['folds_passed']}/{row['n_folds']}{tag}"
        )

    print()
    best = df.iloc[0]
    print(
        f"BEST: {best['model']} / {best['feature_set']} / h={best['horizon']} "
        f"/ R²={best['oos_r2']:.4f} / QR={best['qlike_skill_ratio']:.4f}"
    )
    print("=" * 80 + "\n")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the research sweep on live data."""
    con = connect()
    try:
        df = run_research(con)
        if not df.empty:
            out_path = "data/research_sweep.csv"
            df.to_csv(out_path, index=False)
            print(f"Results saved to {out_path}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
