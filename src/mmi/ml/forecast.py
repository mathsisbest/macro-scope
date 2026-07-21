"""Return forecasting with scikit-learn GradientBoostingRegressor / LGBMRegressor.

Design
------
- One model per forecast horizon (h-step-ahead return).
- Rolling-window expanding walk-forward: train on ``train_size`` rows,
  predict next ``test_size`` rows (each the first ``test_size`` rows after
  the train window that haven't been predicted yet).
- Strictly out-of-sample: prediction for row t uses only information <= t-1.
- Optional feature set selection
  (``feature_set`` = default / vol / vol_macro / vol_rich / vol_medium).
- Optional ensemble method (mean / median).
- Optional target type (raw / vol_adjusted / excess).
- Dynamic ``available_cols``: only columns present in the dataframe are used,
  so missing cross-asset data doesn't force NaN for all rows.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as feat

# ---------- public constants ----------

_MODELS = {
    "gb": HistGradientBoostingRegressor,
    "lgb": lgb.LGBMRegressor,
}

_TARGET_TYPES = ("raw", "vol_adjusted", "excess")


# ---------- internal helpers ----------


def _model_kwargs(model_name: str) -> dict:
    kw: dict = {"random_state": 42}
    if model_name == "gb":
        kw.update(
            max_iter=150,
            max_depth=4,
            min_samples_leaf=20,
            loss="squared_error",
            max_bins=128,
        )
    elif model_name == "lgb":
        kw.update(n_estimators=150, max_depth=4, min_child_samples=20, num_leaves=15, verbose=-1)
    return kw


def _build_target(
    df_cut: pd.DataFrame,
    target_type: str,
    target_col: str = "target_next_ret",
) -> pd.Series:
    """Build the target vector from the dataframe according to `target_type`.

    - ``raw``: the raw forward-return column (default).
    - ``vol_adjusted``: forward-return / trailing 20-day vol.
    - ``excess``: forward-return - rolling 60-day median forward-return.
    """
    y_raw = df_cut[target_col].copy()
    if target_type == "raw":
        return y_raw
    if target_type == "vol_adjusted":
        vol = df_cut["ret"].rolling(20, min_periods=10).std()
        return y_raw / vol.replace(0, np.nan)
    if target_type == "excess":
        median_excess = y_raw.rolling(60, min_periods=20).median()
        return y_raw - median_excess

    raise ValueError(f"Unknown target_type '{target_type}'")


# ---------- model evaluation ----------


def _feasible_date_range(df: pd.DataFrame, train_size: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (first_test_date, last_test_date) so that at least one full walk-forward
    window exists.  Returns NaT-NaT when no feasible range exists."""
    if len(df) < train_size + 1:
        return pd.NaT, pd.NaT
    return df["date"].iloc[train_size], df["date"].iloc[-1]


# ---------- public API ----------


def evaluate_forecast(
    df: pd.DataFrame,
    train_size: int = 250,
    test_size: int = 20,
    horizon: int | None = 1,
    model: str = "gb",
    feature_set: str = "default",
    macro_df: pd.DataFrame | None = None,
    asset_dfs: dict[str, pd.DataFrame] | None = None,
    target_type: str = "raw",
    target_horizon: int = 1,
    ensemble_method: str = "mean",
    use_all_train: bool = False,
    loss: str = "squared_error",
    single_split: bool = False,
    **model_kwargs,
) -> dict:
    """Walk-forward or single-split out-of-sample forecast evaluation.

    Parameters
    ----------
    df:
        Must contain ``date``, ``daily_return`` (and OHLC when feature_set='vol*').
    train_size:
        Number of rows in the initial training window.
    test_size:
        Number of rows per test window (or size of test split when ``single_split=True``).
    horizon:
        Number of days ahead to forecast (h-step return).
    model:
        ``'gb'`` or ``'lgb'``.
    feature_set:
        Feature group name (default/vol/vol_macro/vol_rich/vol_medium).
    macro_df:
        Macro dataframe for vol_macro feature set.
    asset_dfs:
        Asset dict for cross-asset features in vol_macro/vol_rich.
    target_type:
        Target transformation: ``'raw'`` (default), ``'vol_adjusted'``, or ``'excess'``.
    ensemble_method:
        How to combine predictions from overlapping windows: ``'mean'`` or ``'median'``.
    use_all_train:
        If True, train on all data before the test window (expanding window).
    loss:
        Loss function passed to GB: 'squared_error' (default) or 'huber'.
    single_split:
        If True, use a single train/test split (train on last ``train_size`` rows
        before the test period) instead of walk-forward.  Much faster for research sweeps.
    **model_kwargs:
        Additional kwargs forwarded to the regressor constructor.

    Returns
    -------
    dict with keys: horizon, ic, ic_pvalue, direction_accuracy, counts, predictions, dates,
    sharpe, r2, train_size, test_size, n_models, median_model_count, mean_train_rows,
    model, feature_set, target_type, ensemble_method, loss, feature_cols, available_feature_cols.
    """
    df = df.copy()

    if feature_set == "vol_medium" and macro_df is not None:
        df = feat.make_features(df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs)
    elif feature_set == "vol_medium":
        df = feat.make_features(df, feature_set="vol_medium")
    else:
        df = feat.make_features(df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs)

    # Override target if target_horizon > 1: use cumulative N-day return
    if target_horizon > 1:
        df["target_next_ret"] = df["ret"].rolling(target_horizon).sum().shift(-target_horizon)

    df = df.dropna(subset=["target_next_ret"]).reset_index(drop=True)

    if len(df) < train_size + 1:
        return _empty_result(df, model, feature_set, target_type, ensemble_method, loss, horizon)

    all_feature_cols = feat.feature_columns(feature_set)
    available_cols = [c for c in all_feature_cols if c in df.columns]
    df = df.dropna(subset=available_cols, how="all").reset_index(drop=True)
    available_cols = [c for c in available_cols if df[c].notna().any()]

    if not available_cols:
        return _empty_result(df, model, feature_set, target_type, ensemble_method, loss, horizon)

    n = len(df)
    all_preds: pd.Series = pd.Series(index=df.index, dtype=float)
    model_count: pd.Series = pd.Series(0, index=df.index, dtype=int)
    train_rows_list: list[int] = []
    median_preds: dict[int | str, list[float]] = {}  # index -> list of predictions for median

    if single_split:
        train_end = n - test_size
        train_idx = list(range(0, train_end))
        test_idx = list(range(train_end, n))
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        y_train = _build_target(df_train, target_type, "target_next_ret").to_numpy().ravel()
        X_train = df_train[available_cols].to_numpy()
        X_test = df_test[available_cols].to_numpy()
        train_std = np.nanstd(X_train, axis=0)
        non_const = train_std > 0
        if non_const.sum() >= 2:
            X_train = X_train[:, non_const]
            X_test = X_test[:, non_const]
        # Drop NaN in y_train (vol_adjusted/excess produce NaN for early rows)
        valid_idx = ~np.isnan(y_train)
        if valid_idx.sum() < 50:
            return _empty_result(
                df, model, feature_set, target_type, ensemble_method, loss, horizon
            )
        X_train = X_train[valid_idx]
        y_train = y_train[valid_idx]
        try:
            model_cls = _MODELS[model]
        except KeyError:
            raise ValueError(f"Unknown model '{model}'; choose from {list(_MODELS)}") from None
        kw = {**_model_kwargs(model), **model_kwargs}
        if loss != "squared_error":
            kw["loss"] = loss
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*early_stopping.*")
            clf = model_cls(**kw)
            clf.fit(X_train, y_train)
            all_preds.iloc[test_idx] = clf.predict(X_test)
            model_count.iloc[test_idx] = 1
            train_rows_list.append(len(y_train))
    else:
        for start in range(0, n - train_size, test_size):
            if use_all_train:
                # Expanding window: always train from row 0 to current position
                train_start = 0
                train_end = start + train_size
            else:
                # Rolling window: fixed-size training window that slides forward
                train_start = start
                train_end = start + train_size

            train_end = min(train_end, n - 1)
            test_end = min(train_end + test_size, n)

            train_idx = list(range(train_start, train_end))
            test_idx = list(range(train_end, test_end))
            if not test_idx:
                break

            df_train = df.iloc[train_idx]
            df_test = df.iloc[test_idx]

            y_train = _build_target(df_train, target_type, "target_next_ret")
            y_train = y_train.to_numpy().ravel()
            X_train = df_train[available_cols].to_numpy()
            X_test = df_test[available_cols].to_numpy()

            # Drop constant features (std==0) — HistGB crashes on them
            train_std = np.nanstd(X_train, axis=0)
            non_const = train_std > 0
            if non_const.sum() < 2:
                continue
            X_train = X_train[:, non_const]
            X_test = X_test[:, non_const]

            # Drop NaN in y (vol_adjusted/excess targets produce NaN for early rows)
            valid = ~np.isnan(y_train)
            X_train = X_train[valid]
            y_train = y_train[valid]

            if len(y_train) < 50:
                break

            try:
                model_cls = _MODELS[model]
            except KeyError:
                raise ValueError(f"Unknown model '{model}'; choose from {list(_MODELS)}") from None

            kw = {**_model_kwargs(model), **model_kwargs}
            if loss != "squared_error":
                kw["loss"] = loss

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message=".*early_stopping.*"
                )
                clf = model_cls(**kw)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

            if ensemble_method not in ("mean", "median"):
                raise ValueError(f"Unknown ensemble_method: {ensemble_method}")
            if ensemble_method == "median":
                for i, pred in zip(test_idx, preds, strict=False):
                    median_preds.setdefault(i, []).append(float(pred))
            else:
                all_preds.iloc[test_idx] = all_preds.iloc[test_idx].add(preds, fill_value=0)
            model_count.iloc[test_idx] += 1
            train_rows_list.append(len(y_train))

    return _compute_metrics(
        df,
        all_preds,
        model_count,
        available_cols,
        model,
        feature_set,
        target_type,
        ensemble_method,
        loss,
        horizon,
        n,
        train_size,
        test_size,
        train_rows_list,
        target_horizon,
        median_preds,
    )


def train_latest_forecast(
    df: pd.DataFrame,
    train_size: int = 250,
    model: str = "gb",
    feature_set: str = "default",
    macro_df: pd.DataFrame | None = None,
    asset_dfs: dict[str, pd.DataFrame] | None = None,
    target_type: str = "raw",
    target_horizon: int = 1,
    loss: str = "squared_error",
    **model_kwargs,
) -> dict:
    """Train on all rows with known forward targets and predict the latest feature-valid row.

    This is deliberately separate from ``evaluate_forecast``.  The walk-forward evaluator must
    drop the final ``target_horizon`` rows because their realized forward return is unknown, but
    the live dashboard forecast should still be as-of the latest row whose features are known.
    """
    df = df.copy()
    if feature_set == "vol_medium" and macro_df is not None:
        feat_df = feat.make_features(
            df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs
        )
    elif feature_set == "vol_medium":
        feat_df = feat.make_features(df, feature_set="vol_medium")
    else:
        feat_df = feat.make_features(
            df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs
        )

    if target_horizon > 1:
        feat_df["target_next_ret"] = (
            feat_df["ret"].rolling(target_horizon).sum().shift(-target_horizon)
        )

    all_feature_cols = feat.feature_columns(feature_set)
    available_cols = [c for c in all_feature_cols if c in feat_df.columns]
    if not available_cols:
        return {"prediction": np.nan, "as_of": pd.NaT, "train_rows": 0}

    feature_valid = feat_df[available_cols].notna().any(axis=1)
    available_cols = [c for c in available_cols if feat_df.loc[feature_valid, c].notna().any()]
    if not available_cols:
        return {"prediction": np.nan, "as_of": pd.NaT, "train_rows": 0}

    train_df = feat_df[feature_valid & feat_df["target_next_ret"].notna()].copy()
    latest_df = feat_df[feature_valid].tail(1)
    if latest_df.empty or len(train_df) < train_size:
        return {"prediction": np.nan, "as_of": pd.NaT, "train_rows": len(train_df)}

    y_train = _build_target(train_df, target_type, "target_next_ret").to_numpy().ravel()
    X_train = train_df[available_cols].to_numpy()
    X_latest = latest_df[available_cols].to_numpy()

    train_std = np.nanstd(X_train, axis=0)
    non_const = train_std > 0
    if non_const.sum() < 2:
        return {"prediction": np.nan, "as_of": pd.NaT, "train_rows": len(train_df)}
    X_train = X_train[:, non_const]
    X_latest = X_latest[:, non_const]

    valid_y = ~np.isnan(y_train)
    X_train = X_train[valid_y]
    y_train = y_train[valid_y]
    if len(y_train) < train_size:
        return {"prediction": np.nan, "as_of": pd.NaT, "train_rows": len(y_train)}

    try:
        model_cls = _MODELS[model]
    except KeyError:
        raise ValueError(f"Unknown model '{model}'; choose from {list(_MODELS)}") from None
    kw = {**_model_kwargs(model), **model_kwargs}
    if loss != "squared_error":
        kw["loss"] = loss
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*early_stopping.*")
        clf = model_cls(**kw)
        clf.fit(X_train, y_train)
        pred = float(clf.predict(X_latest)[0])

    return {
        "prediction": pred,
        "as_of": pd.to_datetime(latest_df["date"].iloc[0]),
        "train_rows": len(y_train),
        "available_feature_cols": available_cols,
    }


# ---------- internal: helpers ----------


def _empty_result(
    df: pd.DataFrame,
    model: str,
    feature_set: str,
    target_type: str,
    ensemble_method: str,
    loss: str,
    horizon: int | None,
) -> dict:
    return {
        "horizon": horizon,
        "ic": np.nan,
        "ic_pvalue": np.nan,
        "direction_accuracy": np.nan,
        "prediction_count": 0,
        "predictions": pd.Series(dtype=float),
        "dates": pd.Series(dtype=float),
        "sharpe": np.nan,
        "r2": np.nan,
        "train_size": None,
        "test_size": None,
        "n_models": 0,
        "median_model_count": 0,
        "mean_train_rows": 0,
        "model": model,
        "feature_set": feature_set,
        "target_type": target_type,
        "ensemble_method": ensemble_method,
        "loss": loss,
        "feature_cols": [],
        "available_feature_cols": [],
    }


def _compute_metrics(
    df: pd.DataFrame,
    all_preds: pd.Series,
    model_count: pd.Series,
    available_cols: list[str],
    model: str,
    feature_set: str,
    target_type: str,
    ensemble_method: str,
    loss: str,
    horizon: int | None,
    n: int,
    train_size: int,
    test_size: int,
    train_rows_list: list[int],
    target_horizon: int,
    median_preds: dict[int | str, list[float]] | None = None,
) -> dict:
    if ensemble_method == "median":
        preds = pd.Series(dtype=float, index=df.index)
        for idx, vals in (median_preds or {}).items():
            preds[idx] = float(np.median(vals))
    else:
        preds = all_preds / model_count.replace(0, np.nan)

    valid = (model_count > 0) & df["target_next_ret"].notna()
    y_true = df.loc[valid, "target_next_ret"]
    y_pred = preds.loc[valid]

    if len(y_true) < 5:
        return _empty_result(df, model, feature_set, target_type, ensemble_method, loss, horizon)

    from scipy.stats import pearsonr

    ic_val, ic_pval = pearsonr(y_true, y_pred)
    ic_val = float(ic_val) if not pd.isna(ic_val) else 0.0
    ic_pval = float(ic_pval) if not pd.isna(ic_pval) else 1.0

    direction_tp = ((y_true > 0) & (y_pred > 0)).sum()
    direction_tn = ((y_true < 0) & (y_pred < 0)).sum()
    direction_accuracy = (direction_tp + direction_tn) / len(y_true) if len(y_true) > 0 else np.nan
    positive_target_rate = float((y_true > 0).mean()) if len(y_true) > 0 else np.nan
    positive_prediction_rate = float((y_pred > 0).mean()) if len(y_pred) > 0 else np.nan
    negative_target_rate = float((y_true < 0).mean()) if len(y_true) > 0 else np.nan
    baseline_direction_accuracy = max(positive_target_rate, negative_target_rate)
    direction_edge = direction_accuracy - baseline_direction_accuracy

    strategy_target_ret = np.sign(y_pred) * y_true
    if target_horizon == 1 and strategy_target_ret.std() > 0:
        sharpe = strategy_target_ret.mean() / strategy_target_ret.std() * np.sqrt(252)
    else:
        # Multi-day forward targets are overlapping observations; reporting them as a daily
        # tradable Sharpe is misleading. Keep the field for API compatibility but fail closed.
        sharpe = np.nan

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n_models = len(train_rows_list)
    median_model_count_val = (
        int(model_count[model_count > 0].median()) if (model_count > 0).any() else 0
    )
    mean_train_rows = int(np.mean(train_rows_list)) if train_rows_list else 0

    return {
        "horizon": horizon,
        "ic": ic_val,
        "ic_pvalue": ic_pval,
        "direction_accuracy": direction_accuracy,
        "baseline_direction_accuracy": baseline_direction_accuracy,
        "direction_edge": direction_edge,
        "positive_target_rate": positive_target_rate,
        "positive_prediction_rate": positive_prediction_rate,
        "prediction_count": len(y_true),
        "predictions": y_pred,
        "y_true": y_true,
        "dates": df.loc[valid, "date"],
        "sharpe": sharpe,
        "r2": r2,
        "train_size": train_size,
        "test_size": test_size,
        "n_models": n_models,
        "median_model_count": median_model_count_val,
        "mean_train_rows": mean_train_rows,
        "model": model,
        "feature_set": feature_set,
        "target_type": target_type,
        "ensemble_method": ensemble_method,
        "loss": loss,
        "feature_cols": feat.feature_columns(feature_set),
        "available_feature_cols": available_cols,
    }
