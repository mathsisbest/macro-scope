"""Panel-level forecast backtest: combine predictions across tickers + horizons.

Provides the ``ForecastBacktest`` class that:
  - iterates over a configurable panel of tickers,
  - runs the walk-forward forecast for each,
  - optionally ensembles across horizons,
  - returns performance metrics.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import features as feat
from .forecast import evaluate_forecast, _feasible_date_range

_FORECAST_MODEL = "gb"
_HORIZONS = (1, 5, 10, 20)
_TRAIN_ROWS = 1260
_TEST_ROWS = 20
_ENSEMBLE_METHODS = ("mean", "median", "ic_weighted")
_DEFAULT_ENSEMBLE = "mean"
_DEFAULT_TARGET_TYPE = "raw"
_DEFAULT_LOSS = "squared_error"
_FEATURE_SET = "vol_macro"


class ForecastBacktest:
    """Panel forecast backtest.

    Parameters
    ----------
    universe:
        List of tickers.  Each ticker must have a corresponding key in
        ``macro_db.asset_dfs[ticker]``.
    macro_db:
        A database accessor (e.g. DuckDBMacroDB) bound to this backtest.
        Must expose ``prices_df(symbol)``, ``macro_df`` and a dict-like ``asset_dfs``.
    run_date:
        Optional cutoff date.  Only data <= run_date is used.
    """

    def __init__(
        self,
        universe: list[str],
        macro_db: object,
        run_date: str | None = None,
    ):
        self.universe = universe
        self.macro_db = macro_db
        self.run_date = pd.Timestamp(run_date) if run_date else None

    @staticmethod
    def rolling_window_split(
        df: pd.DataFrame,
        train_size: int,
        test_size: int,
    ):
        """Yield (train_idx, test_idx) tuples over ``df``."""
        total = len(df)
        for start in range(0, total - train_size, test_size):
            train_end = start + train_size
            test_end = min(train_end + test_size, total)
            yield list(range(start, train_end)), list(range(train_end, test_end))

    def run_forecast(
        self,
        symbol: str,
        model: str = _FORECAST_MODEL,
        feature_set: str = _FEATURE_SET,
        horizons: tuple[int, ...] = _HORIZONS,
        train_size: int = _TRAIN_ROWS,
        test_size: int = _TEST_ROWS,
        target_type: str = _DEFAULT_TARGET_TYPE,
        ensemble_method: str = _DEFAULT_ENSEMBLE,
        loss: str = _DEFAULT_LOSS,
    ) -> dict:
        """Run the forecast backtest for a single symbol.

        Returns a dict of results (keys: dates, predictions, ic, direction_accuracy,
        horizon_map, etc.).

        When multiple horizons are given, each horizon's predictions are computed
        independently; they are then combined into a single ensemble prediction.

        The ensemble method is one of:
          - ``'mean'``: equal-weight average of horizon predictions.
          - ``'median'``: equal-weight median of horizon predictions.
          - ``'ic_weighted'``: weight each horizon by its historical IC (from the
            validation portion of each walk-forward window).  The initial 5 windows
            use ``'mean'`` as a warm-up; after that, each horizon's weight is its
            trailing-IC over the N most recent out-of-sample prediction batches.
        """
        df = self.macro_db.prices_df(symbol)
        if df is None or df.empty:
            return {"error": f"No data for {symbol}"}

        required = {"date", "daily_return"}
        if feature_set in ("vol", "vol_macro", "vol_rich", "vol_medium"):
            required |= {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            return {"error": f"{symbol}: missing columns {required - set(df.columns)}"}

        df = df.sort_values("date").reset_index(drop=True)
        if self.run_date:
            df = df[df["date"] <= self.run_date].reset_index(drop=True)

        macro_df = getattr(self.macro_db, "macro_df", None)
        asset_dfs = getattr(self.macro_db, "asset_dfs", None)

        horizon_results: dict[int, dict] = {}
        for h in horizons:
            h_res = evaluate_forecast(
                df=df,
                train_size=train_size,
                test_size=test_size,
                horizon=h,
                model=model,
                feature_set=feature_set,
                macro_df=macro_df,
                asset_dfs=asset_dfs,
                target_type=target_type,
                ensemble_method=ensemble_method,
                loss=loss,
            )
            horizon_results[h] = h_res

        ens_res = self._combine_horizons(horizon_results, ensemble_method)
        ens_res["horizon_results"] = horizon_results
        return ens_res

    def _combine_horizons(
        self,
        horizon_results: dict[int, dict],
        ensemble_method: str,
    ) -> dict:
        """Combine predictions across horizons into a single ensemble.

        For ``ic_weighted``, uses the trailing IC from each horizon's OOS
        predictions as weights.
        """
        first = next(iter(horizon_results.values()))
        if "error" in first:
            return first

        valid_horizons = {
            h: r
            for h, r in horizon_results.items()
            if "predictions" in r and r.get("prediction_count", 0) > 0
        }
        if not valid_horizons:
            return _empty_panel_result()

        if ensemble_method == "ic_weighted":
            weight_map = _calc_ic_weights(valid_horizons)
        else:
            weight_map = {h: 1.0 / len(valid_horizons) for h in valid_horizons}

        all_dates = set()
        for r in valid_horizons.values():
            if "dates" in r and hasattr(r["dates"], "values"):
                all_dates.update(pd.to_datetime(r["dates"].values))
            elif "dates" in r and isinstance(r["dates"], (list, np.ndarray)):
                all_dates.update(r["dates"])
        all_dates = sorted(all_dates)
        if not all_dates:
            return _empty_panel_result()

        pred_df = pd.DataFrame({"date": all_dates})
        for h, r in valid_horizons.items():
            if "dates" in r and "predictions" in r:
                h_df = pd.DataFrame({
                    "date": pd.to_datetime(r["dates"].values),
                    f"pred_h{h}": r["predictions"].values,
                })
                h_df = h_df.dropna(subset=[f"pred_h{h}"])
                pred_df = pred_df.merge(h_df, on="date", how="left")

        weight_cols = [f"pred_h{h}" for h in valid_horizons]
        present = [c for c in weight_cols if c in pred_df.columns]
        if ensemble_method == "mean":
            pred_df["ensemble_pred"] = pred_df[present].mean(axis=1)
        elif ensemble_method == "median":
            pred_df["ensemble_pred"] = pred_df[present].median(axis=1)
        else:
            w = np.array([weight_map.get(h, 1.0 / len(valid_horizons)) for h in valid_horizons])
            w = w / w.sum()
            pred_df["ensemble_pred"] = pred_df[present].dot(w)

        combined = pred_df.dropna(subset=["ensemble_pred"])
        if combined.empty:
            return _empty_panel_result()

        full = first.get("dates", pd.Series())
        if isinstance(full, pd.Series) and not full.empty:
            first_res = next(iter(valid_horizons.values()))
            y_true = pd.Series(index=full.values, dtype=float)
        else:
            y_true = pd.Series(dtype=float)

        return _compute_panel_metrics(combined, y_true, valid_horizons, ensemble_method)

    def run_universe(
        self,
        model: str = _FORECAST_MODEL,
        feature_set: str = _FEATURE_SET,
        horizons: tuple[int, ...] = _HORIZONS,
        train_size: int = _TRAIN_ROWS,
        test_size: int = _TEST_ROWS,
        target_type: str = _DEFAULT_TARGET_TYPE,
        ensemble_method: str = _DEFAULT_ENSEMBLE,
        loss: str = _DEFAULT_LOSS,
        progress: bool = True,
        forecast_suffix: str | None = None,
    ) -> dict:
        """Run forecast for every ticker in the universe.

        Returns a dict keyed by ticker, each value is the result of ``run_forecast``.
        """
        results: dict = {}
        tickers = self.universe
        if not tickers:
            tickers = list(self.macro_db.prices_df_cache.keys())
        for i, sym in enumerate(tickers):
            if progress:
                print(f"  [{i + 1}/{len(tickers)}] {sym} ...")
            res = self.run_forecast(
                symbol=sym,
                model=model,
                feature_set=feature_set,
                horizons=horizons,
                train_size=train_size,
                test_size=test_size,
                target_type=target_type,
                ensemble_method=ensemble_method,
                loss=loss,
            )
            results[sym] = res
        return results


def _empty_panel_result() -> dict:
    return {
        "ic": np.nan,
        "direction_accuracy": np.nan,
        "prediction_count": 0,
        "sharpe": np.nan,
        "r2": np.nan,
        "median_model_count": 0,
        "mean_train_rows": 0,
    }


def _calc_ic_weights(horizon_results: dict[int, dict]) -> dict[int, float]:
    """Weight each horizon by its historical IC, with exponential decay."""
    MIN_WARMUP = 5
    ic_vals = {}
    for h, r in horizon_results.items():
        ic = r.get("ic", np.nan)
        if pd.notna(ic) and r.get("prediction_count", 0) > MIN_WARMUP * 20:
            ic_vals[h] = max(ic, 0.01)
    if not ic_vals:
        return {h: 1.0 / len(horizon_results) for h in horizon_results}
    total = sum(ic_vals.values())
    if total <= 0:
        return {h: 1.0 / len(ic_vals) for h in ic_vals}
    return {h: v / total for h, v in ic_vals.items()}


def _compute_panel_metrics(
    combined: pd.DataFrame,
    y_true: pd.Series,
    valid_horizons: dict[int, dict],
    ensemble_method: str,
) -> dict:
    """Compute final performance metrics from ensemble predictions."""
    if len(combined) < 5:
        return _empty_panel_result()

    ens_pred = combined["ensemble_pred"]
    y_true_series = pd.Series(dtype=float)
    ic = np.nan
    direction_accuracy = np.nan
    sharpe = np.nan
    r2 = np.nan

    from scipy.stats import pearsonr

    try:
        ic_val, ic_pval = pearsonr(
            ens_pred.values.astype(float),
            y_true_series.values.astype(float),
        )
        ic = float(ic_val) if not pd.isna(ic_val) else 0.0
    except Exception:
        ic = 0.0

    first = next(iter(valid_horizons.values()))
    return {
        "ic": ic,
        "direction_accuracy": direction_accuracy,
        "prediction_count": len(combined),
        "sharpe": sharpe,
        "r2": r2,
        "median_model_count": first.get("median_model_count", 0),
        "mean_train_rows": first.get("mean_train_rows", 0),
        "ensemble_pred": combined["ensemble_pred"],
        "ensemble_method": ensemble_method,
    }
