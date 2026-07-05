"""Research sweep for return forecast: systematically search hyperparameter + config space.

Compares models (GB vs LGB) × feature sets (default/vol/vol_medium) × target types
(raw/vol_adjusted/excess) × horizons × regime vs global × loss functions × n_estimators.

Usage:
    python -m mmi.ml.research_forecast
    python -m mmi.ml.research_forecast --db-path /path/to/mmi.duckdb --output sweep.csv
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from .forecast import evaluate_forecast

# Lazy import for DB
_mm_db: object = None


def _get_db(db_path: str | None = None):
    global _mm_db
    if _mm_db is not None:
        return _mm_db
    from ..db import get_macro_db

    _mm_db = get_macro_db(db_path=db_path)
    return _mm_db


def run_sweep(
    db_path: str | None = None,
    symbol: str = "SPY",
    train_size: int = 250,
    test_size: int = 20,
    min_rows: int = 100,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run a full sweep over model × feature_set × target_type × horizon × regime × loss × n_estimators.

    Parameters
    ----------
    db_path: Optional path to mmi.duckdb.
    symbol: Ticker to forecast (default SPY).
    train_size: Walk-forward training window.
    test_size: Walk-forward test window.
    min_rows: Minimum required rows after feature engineering.
    output_path: If given, save results CSV here.

    Returns
    -------
    DataFrame with columns: model, feature_set, target_type, horizon, regime_aware,
    loss, n_estimators, ic, ic_pvalue, direction_accuracy, prediction_count, sharpe,
    r2, mean_train_rows.
    """
    db = _get_db(db_path)
    df = db.prices_df(symbol)
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")

    print(f"Loaded {len(df)} rows for {symbol}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    models = ["gb", "lgb"]
    feature_sets = ["default", "vol", "vol_medium"]
    target_types = ["raw", "vol_adjusted", "excess"]
    horizons = [1, 5, 10]
    losses = ["squared_error", "huber"]
    n_estimators_list = [250, 500]
    regime_options = [False, True]  # False = global, True = regime-aware (lagged vol)

    results = []

    total = (
        len(models)
        * len(feature_sets)
        * len(target_types)
        * len(horizons)
        * len(losses)
        * len(n_estimators_list)
        * len(regime_options)
    )
    count = 0

    for model, fset, ttype, horizon, loss, nest, regime in itertools.product(
        models, feature_sets, target_types, horizons, losses, n_estimators_list, regime_options
    ):
        count += 1
        desc = f"[{count}/{total}] {model} / {fset} / {ttype} / h={horizon} / {'regime' if regime else 'global'} / loss={loss} / n={nest}"
        print(desc, end=" ... ")

        try:
            if fset in ("vol", "vol_macro", "vol_medium", "vol_rich"):
                macro_df = getattr(db, "macro_df", None)
                asset_dfs = getattr(db, "asset_dfs", None)
            else:
                macro_df = None
                asset_dfs = None

            res = evaluate_forecast(
                df=df,
                train_size=train_size,
                test_size=test_size,
                horizon=horizon,
                model=model,
                feature_set=fset,
                macro_df=macro_df,
                asset_dfs=asset_dfs,
                target_type=ttype,
                ensemble_method="mean",
                loss=loss,
                n_estimators=nest,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if res.get("prediction_count", 0) < min_rows:
            print(f"too few preds ({res.get('prediction_count', 0)})")
            continue

        results.append({
            "model": model,
            "feature_set": fset,
            "target_type": ttype,
            "horizon": horizon,
            "regime_aware": regime,
            "loss": loss,
            "n_estimators": nest,
            "ic": res.get("ic", np.nan),
            "ic_pvalue": res.get("ic_pvalue", np.nan),
            "direction_accuracy": res.get("direction_accuracy", np.nan),
            "prediction_count": res.get("prediction_count", 0),
            "sharpe": res.get("sharpe", np.nan),
            "r2": res.get("r2", np.nan),
            "mean_train_rows": res.get("mean_train_rows", 0),
        })
        print(f"IC={res.get('ic', np.nan):.4f}  DA={res.get('direction_accuracy', np.nan):.3f}")

    out = pd.DataFrame(results)
    if output_path and not out.empty:
        out.to_csv(output_path, index=False)
        print(f"\nSaved {len(out)} results to {output_path}")

    return out


def summarize(results: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Print a summary of sweep results, sorted by IC."""
    top = results.sort_values("ic", ascending=False).head(top_k)
    print("\n=== Top Results (by IC) ===")
    cols = ["model", "feature_set", "target_type", "horizon", "ic",
            "direction_accuracy", "prediction_count", "sharpe"]
    print(top[cols].to_string(index=False))
    return top


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research sweep for return forecast")
    parser.add_argument("--db-path", type=str, default=None, help="Path to mmi.duckdb")
    parser.add_argument("--output", type=str,
                        default="data/research_forecast_sweep.csv",
                        help="Output CSV path")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--train-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=100)
    args = parser.parse_args()

    results_df = run_sweep(
        db_path=args.db_path,
        symbol=args.symbol,
        train_size=args.train_size,
        test_size=args.test_size,
        min_rows=args.min_rows,
        output_path=args.output,
    )
    if not results_df.empty:
        summarize(results_df)
