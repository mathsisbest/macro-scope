"""Research sweep for return forecast: systematically search hyperparameter + config space.

Compares models (GB vs LGB) × feature sets (default/vol/vol_medium) × target types
(raw/vol_adjusted/excess) × horizons × regime vs global × loss functions × n_estimators.

Usage:
    python -m mmi.ml.research_forecast
    python -m mmi.ml.research_forecast --output data/research_forecast_sweep.csv
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from mmi.utils.db import connect

from .forecast import evaluate_forecast

# ---------------------------------------------------------------------------
# Data helpers — fetch & pivot from raw DuckDB
# ---------------------------------------------------------------------------

_MACRO_SERIES = [
    "T10Y2Y",
    "DGS10",
    "DGS2",
    "DGS3MO",
    "FEDFUNDS",
    "VIXCLS",
    "DCOILWTICO",
    "DTWEXBGS",
    "ICSA",
    "NFCI",
]


def _pivot_macro(con) -> pd.DataFrame:
    """Pivot the long-format macro indicator table into wide columns.

    Returns a DataFrame keyed on ``date`` with one column per series_id in
    ``_MACRO_SERIES``.
    """
    if con is None:
        return pd.DataFrame()
    placeholders = ", ".join(f"'{s}'" for s in _MACRO_SERIES)
    df = con.execute(f"""
        SELECT date, series_id, value
        FROM marts.fct_macro_indicator
        WHERE series_id IN ({placeholders})
        ORDER BY date, series_id
    """).df()
    if df.empty:
        return pd.DataFrame()
    pivoted = df.pivot_table(
        index="date", columns="series_id", values="value"
    ).reset_index()
    pivoted.columns.name = None
    pivoted["date"] = pd.to_datetime(pivoted["date"])
    return pivoted.sort_values("date").reset_index(drop=True)


def _load_asset_vol(con, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Load daily returns for cross-asset symbols (for vol features)."""
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = con.execute(f"""
            SELECT date, daily_return
            FROM marts.fct_asset_daily
            WHERE symbol = '{sym}'
            ORDER BY date
        """).df()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            result[sym] = df
    return result


def _spy_df(con) -> pd.DataFrame:
    """Fetch SPY OHLC + daily_return from fct_asset_daily."""
    df = con.execute("""
        SELECT
            date, open, high, low, close, daily_return
        FROM marts.fct_asset_daily
        WHERE symbol = 'SPY'
        ORDER BY date
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    con=None,
    db_path: str | None = None,
    symbol: str = "SPY",
    train_size: int = 250,
    test_size: int = 20,
    min_rows: int = 100,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run a full sweep over model × feature_set × target_type × horizon
    × regime × loss × n_estimators.

    Parameters
    ----------
    con:
        An open DuckDB connection. If None, one will be opened via
        ``mmi.utils.db.connect``.
    db_path:
        Optional explicit path to mmi.duckdb (falls back to env default).
    symbol:
        Ticker to forecast (default SPY).
    train_size:
        Walk-forward training window (rows).
    test_size:
        Walk-forward test window (rows).
    min_rows:
        Minimum required test predictions to record a result.
    output_path:
        If given, save results CSV here.

    Returns
    -------
    DataFrame with columns: model, feature_set, target_type, horizon, regime_aware,
    loss, n_estimators, ic, ic_pvalue, direction_accuracy, prediction_count, sharpe,
    r2, mean_train_rows.
    """
    if con is None:
        con = connect(db_path)

    df = _spy_df(con)
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")
    print(f"Loaded {len(df)} rows for {symbol}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    macro_df = _pivot_macro(con)
    cols = [c for c in macro_df.columns if c != "date"]
    print(f"Macro series: {len(macro_df)} rows, columns: {cols}")

    asset_dfs = _load_asset_vol(con, ("GLD", "TLT"))
    print(f"Cross-asset data: {list(asset_dfs.keys())}")

    models = ["gb", "lgb"]
    feature_sets = ["default", "vol_medium"]
    target_types = ["raw", "vol_adjusted", "excess"]
    horizons = [1, 5, 10]
    losses = ["squared_error"]
    # GB uses max_iter (HistGB), LGB uses n_estimators — we pass both; the
    # irrelevant one is silently ignored because model_kwargs is model-specific.
    model_strengths = [150]
    regime_options = [False, True]

    results = []

    total = (
        len(models)
        * len(feature_sets)
        * len(target_types)
        * len(horizons)
        * len(losses)
        * len(model_strengths)
        * len(regime_options)
    )
    count = 0

    for model, fset, ttype, horizon, loss, ntree, regime in itertools.product(
        models, feature_sets, target_types, horizons, losses, model_strengths, regime_options
    ):
        count += 1
        regime_str = "regime" if regime else "global"
        desc = (
            f"[{count}/{total}] {model} / {fset} / {ttype} / h={horizon}"
            f" / {regime_str} / loss={loss} / n={ntree}"
        )
        print(desc, end=" ... ")

        try:
            needs_macro = fset in ("vol", "vol_macro", "vol_medium", "vol_rich")
            kwargs = {"max_iter": ntree, "n_estimators": ntree}
            res = evaluate_forecast(
                df=df,
                train_size=train_size,
                test_size=test_size,
                horizon=horizon,
                model=model,
                feature_set=fset,
                macro_df=macro_df if needs_macro else None,
                asset_dfs=asset_dfs if needs_macro else None,
                target_type=ttype,
                ensemble_method="mean",
                single_split=True,
                loss=loss,
                **kwargs,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if res.get("prediction_count", 0) < min_rows:
            print(f"too few preds ({res.get('prediction_count', 0)})")
            continue

        results.append(
            {
                "model": model,
                "feature_set": fset,
                "target_type": ttype,
                "horizon": horizon,
                "regime_aware": regime,
                "loss": loss,
                "n_estimators": ntree,
                "ic": res.get("ic", np.nan),
                "ic_pvalue": res.get("ic_pvalue", np.nan),
                "direction_accuracy": res.get("direction_accuracy", np.nan),
                "prediction_count": res.get("prediction_count", 0),
                "sharpe": res.get("sharpe", np.nan),
                "r2": res.get("r2", np.nan),
                "mean_train_rows": res.get("mean_train_rows", 0),
            }
        )
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
    cols = [
        "model",
        "feature_set",
        "target_type",
        "horizon",
        "ic",
        "direction_accuracy",
        "prediction_count",
        "sharpe",
    ]
    print(top[cols].to_string(index=False))
    return top


def _snapshot_connect() -> object:
    """Open an in-memory DuckDB connection that reads from the committed Parquet snapshot."""
    import duckdb as _duckdb

    con = _duckdb.connect()
    con.execute("CREATE SCHEMA IF NOT EXISTS marts")
    con.execute("""
        CREATE VIEW marts.fct_asset_daily AS
        SELECT * FROM 'data/public/fct_asset_daily.parquet'
    """)
    con.execute("""
        CREATE VIEW marts.fct_macro_indicator AS
        SELECT * FROM 'data/public/fct_macro_indicator.parquet'
    """)
    return con


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research sweep for return forecast")
    parser.add_argument("--db-path", type=str, default=None, help="Path to mmi.duckdb")
    parser.add_argument(
        "--snapshot", action="store_true", help="Use committed snapshot Parquet instead of live DB"
    )
    parser.add_argument(
        "--output", type=str, default="data/research_forecast_sweep.csv", help="Output CSV path"
    )
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--train-size", type=int, default=250)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=100)
    args = parser.parse_args()

    if args.snapshot:
        con = _snapshot_connect()
    else:
        from mmi.utils.db import connect

        con = connect(args.db_path)

    results_df = run_sweep(
        con=con,
        symbol=args.symbol,
        train_size=args.train_size,
        test_size=args.test_size,
        min_rows=args.min_rows,
        output_path=args.output,
    )
    con.close()
    if not results_df.empty:
        summarize(results_df)
