"""Portfolio-level return forecasts and signal generation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..ml import features as feat
from ..ml.forecast import evaluate_forecast


def compute_all_predictions(
    db,
    universe: Sequence[str] | None = None,
    train_size: int = 1260,
    test_size: int = 20,
    model: str = "lgb",
    feature_set: str = "vol_medium",
    roll_window: int = 1260,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    max_workers: int = 4,
    target_type: str = "raw",
    ensemble_method: str = "mean",
    loss: str = "squared_error",
    **model_kwargs,
) -> pd.DataFrame:
    """Compute out-of-sample ensemble predictions for each ticker across horizons.

    Parameters
    ----------
    db:
        Database accessor with ``prices_df(symbol)``, ``macro_df``, ``asset_dfs``.
    universe:
        Ticker list.  Defaults to S&P 500 constituents.
    train_size:
        Walk-forward training window rows.
    test_size:
        Walk-forward test window rows.
    model:
        ``'gb'`` or ``'lgb'``.
    feature_set:
        Feature group (default/vol/vol_macro/vol_rich/vol_medium).
    roll_window:
        Not used directly (legacy).
    horizons:
        Forecast horizons to ensemble across.
    max_workers:
        Max parallel workers (not yet implemented).
    target_type:
        ``'raw'``, ``'vol_adjusted'``, or ``'excess'``.
    ensemble_method:
        ``'mean'``, ``'median'``, or ``'ic_weighted'``.
    loss:
        Loss function for GB: ``'squared_error'`` or ``'huber'``.
    **model_kwargs:
        Passed to the regressor.

    Returns
    -------
    pd.DataFrame with columns ``date, symbol, pred_ret, ue, pos_signal, pred_vol``.
    """
    if universe is None:
        universe = ["SPY"]
    output_rows = []

    macro_df = getattr(db, "macro_df", None)
    asset_dfs = getattr(db, "asset_dfs", None)

    for sym in universe:
        df = db.prices_df(sym)
        if df is None or df.empty or len(df) < train_size + 1:
            continue

        res = evaluate_forecast(
            df=df,
            train_size=train_size,
            test_size=test_size,
            horizon=None,
            model=model,
            feature_set=feature_set,
            macro_df=macro_df,
            asset_dfs=asset_dfs if sym == "SPY" else None,
            target_type=target_type,
            ensemble_method=ensemble_method,
            loss=loss,
            **model_kwargs,
        )

        if res.get("prediction_count", 0) == 0:
            continue

        dates = res.get("dates", pd.Series(dtype="object"))
        preds = res.get("predictions", pd.Series(dtype=float))

        if isinstance(dates, pd.Series) and isinstance(preds, pd.Series):
            for d, p in zip(dates.values, preds.values):
                output_rows.append({
                    "date": pd.Timestamp(d),
                    "symbol": sym,
                    "pred_ret": float(p) if pd.notna(p) else np.nan,
                })

    if not output_rows:
        return pd.DataFrame(columns=["date", "symbol", "pred_ret", "ue", "pos_signal", "pred_vol"])

    out = pd.DataFrame(output_rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    out["ue"] = np.nan
    out["pos_signal"] = 0
    out["pred_vol"] = np.nan
    return out
