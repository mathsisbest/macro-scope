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
from .metrics import (
    ForecastEvaluationResult,
    compute_directional_accuracy,
    compute_ic,
    compute_r2,
    compute_regime_directional_accuracy,
    compute_sharpe,
)
from .splitters import feasible_date_range, walk_forward_split

# ---------- public constants ----------

_MODELS = {
    "gb": HistGradientBoostingRegressor,
    "lgb": lgb.LGBMRegressor,
}

_TARGET_TYPES = ("raw", "vol_adjusted", "excess")


# ---------- internal helpers ----------


def _model_kwargs(model_name: str, loss: str = "squared_error") -> dict:
    kw: dict = {"random_state": 42}
    if model_name == "gb":
        loss_val = loss if loss in ("squared_error", "huber", "absolute_error") else "squared_error"
        kw.update(
            max_iter=150,
            max_depth=4,
            min_samples_leaf=20,
            loss=loss_val,
            max_bins=128,
        )
    elif model_name == "lgb":
        obj_val = "huber" if loss == "huber" else "regression"
        kw.update(
            n_estimators=150,
            max_depth=4,
            min_child_samples=20,
            num_leaves=15,
            objective=obj_val,
            verbose=-1,
        )
    return kw


def tune_model_kwargs(
    model_name: str, X_train: np.ndarray, y_train: np.ndarray, loss: str = "squared_error"
) -> dict:
    """Perform TimeSeriesSplit GridSearchCV to find optimal hyperparameters out-of-sample."""
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

    kw = _model_kwargs(model_name, loss=loss)
    if len(X_train) < 60:
        return kw

    tscv = TimeSeriesSplit(n_splits=3)
    if model_name == "lgb":
        param_grid = {"learning_rate": [0.03, 0.08], "max_depth": [3, 4], "num_leaves": [7, 15]}
        model_inst = lgb.LGBMRegressor(**kw)
    else:
        param_grid = {
            "learning_rate": [0.03, 0.08],
            "max_depth": [3, 4],
            "l2_regularization": [0.0, 0.1],
        }
        model_inst = HistGradientBoostingRegressor(**kw)

    try:
        grid = GridSearchCV(
            model_inst, param_grid=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        kw.update(grid.best_params_)
    except Exception:
        pass
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
    return feasible_date_range(df, train_size)


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
    tune_hyperparameters: bool = False,
    **model_kwargs,
) -> ForecastEvaluationResult:
    """Walk-forward or single-split out-of-sample forecast evaluation."""
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
    last_feature_importances: dict[str, float] = {}

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
        used_cols = available_cols
        if non_const.sum() >= 2:
            X_train = X_train[:, non_const]
            X_test = X_test[:, non_const]
            used_cols = [col for col, nc in zip(available_cols, non_const, strict=False) if nc]
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
        base_kw = (
            tune_model_kwargs(model, X_train, y_train, loss=loss)
            if tune_hyperparameters
            else _model_kwargs(model, loss=loss)
        )
        kw = {**base_kw, **model_kwargs}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*early_stopping.*")
            clf = model_cls(**kw)
            clf.fit(X_train, y_train)
            all_preds.iloc[test_idx] = clf.predict(X_test)
            model_count.iloc[test_idx] = 1
            train_rows_list.append(len(y_train))
            if hasattr(clf, "feature_importances_"):
                fi = clf.feature_importances_
                for col_name, val in zip(used_cols, fi, strict=False):
                    last_feature_importances[col_name] = float(val)
    else:
        for train_idx, test_idx in walk_forward_split(
            n, train_size, test_size, single_split=False, use_all_train=use_all_train
        ):
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
            used_cols = [col for col, nc in zip(available_cols, non_const, strict=False) if nc]
            X_train = X_train[:, non_const]
            X_test = X_test[:, non_const]

            # Drop NaN in y (vol_adjusted/excess targets produce NaN for early rows)
            valid = ~np.isnan(y_train)
            X_train = X_train[valid]
            y_train = y_train[valid]

            if len(y_train) < 50:
                continue

            try:
                model_cls = _MODELS[model]
            except KeyError:
                raise ValueError(f"Unknown model '{model}'; choose from {list(_MODELS)}") from None

            base_kw = (
                tune_model_kwargs(model, X_train, y_train, loss=loss)
                if tune_hyperparameters
                else _model_kwargs(model, loss=loss)
            )
            kw = {**base_kw, **model_kwargs}

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message=".*early_stopping.*"
                )
                clf = model_cls(**kw)
                clf.fit(X_train, y_train)

            preds_fold = clf.predict(X_test)
            if hasattr(clf, "feature_importances_"):
                fi = clf.feature_importances_
                for col_name, val in zip(used_cols, fi, strict=False):
                    last_feature_importances[col_name] = float(val)

            if ensemble_method == "median":
                for i_pos, row_idx in enumerate(test_idx):
                    median_preds.setdefault(row_idx, []).append(float(preds_fold[i_pos]))
                    model_count.iloc[row_idx] += 1
            else:
                all_preds.iloc[test_idx] = all_preds.iloc[test_idx].fillna(0) + pd.Series(
                    preds_fold, index=test_idx
                )
                model_count.iloc[test_idx] += 1
            train_rows_list.append(len(y_train))

    res = _compute_metrics(
        df=df,
        all_preds=all_preds,
        model_count=model_count,
        available_cols=available_cols,
        model=model,
        feature_set=feature_set,
        target_type=target_type,
        ensemble_method=ensemble_method,
        loss=loss,
        horizon=horizon,
        n=n,
        train_size=train_size,
        test_size=test_size,
        train_rows_list=train_rows_list,
        target_horizon=target_horizon,
        median_preds=median_preds if ensemble_method == "median" else None,
    )
    res["feature_importances"] = last_feature_importances
    return res


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
    tune_hyperparameters: bool = True,
    **model_kwargs,
) -> dict:
    """Train on the most recent `train_size` rows and produce a single latest prediction."""
    df = df.copy()
    df = feat.make_features(df, feature_set=feature_set, macro_df=macro_df, asset_dfs=asset_dfs)

    if target_horizon > 1:
        df["target_next_ret"] = df["ret"].rolling(target_horizon).sum().shift(-target_horizon)

    all_feature_cols = feat.feature_columns(feature_set)
    available_cols = [c for c in all_feature_cols if c in df.columns]
    df_features = df.dropna(subset=available_cols, how="all").reset_index(drop=True)

    if len(df_features) < train_size + 1:
        return {"as_of": df["date"].iloc[-1] if not df.empty else None, "prediction": None}

    df_train_raw = df_features.iloc[-(train_size + 1) : -1]
    last_row = df_features.iloc[[-1]]

    y_train = _build_target(df_train_raw, target_type, "target_next_ret").to_numpy().ravel()
    X_train = df_train_raw[available_cols].to_numpy()
    X_pred = last_row[available_cols].to_numpy()

    train_std = np.nanstd(X_train, axis=0)
    non_const = train_std > 0
    if non_const.sum() < 2:
        return {"as_of": last_row["date"].iloc[0], "prediction": None}

    used_cols = [col for col, nc in zip(available_cols, non_const, strict=False) if nc]
    X_train = X_train[:, non_const]
    X_pred = X_pred[:, non_const]

    valid = ~np.isnan(y_train)
    X_train = X_train[valid]
    y_train = y_train[valid]

    if len(y_train) < 50:
        return {"as_of": last_row["date"].iloc[0], "prediction": None}

    try:
        model_cls = _MODELS[model]
    except KeyError:
        raise ValueError(f"Unknown model '{model}'; choose from {list(_MODELS)}") from None

    base_kw = (
        tune_model_kwargs(model, X_train, y_train, loss=loss)
        if tune_hyperparameters
        else _model_kwargs(model, loss=loss)
    )
    kw = {**base_kw, **model_kwargs}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*early_stopping.*")
        clf = model_cls(**kw)
        clf.fit(X_train, y_train)

    raw_pred = float(clf.predict(X_pred)[0])
    # Bayesian Shrinkage Calibration: shrink raw ML forecast toward historical mean return
    hist_mu = float(y_train.mean()) if len(y_train) > 0 else 0.08
    pred_calibrated = 0.50 * raw_pred + 0.50 * hist_mu
    vol_bound = (
        float(df["daily_return"].iloc[-60:].std() * np.sqrt(target_horizon))
        if "daily_return" in df.columns
        else 0.15
    )
    max_bound = max(0.05, 1.5 * (vol_bound if pd.notna(vol_bound) and vol_bound > 0 else 0.15))
    pred = float(np.clip(pred_calibrated, -max_bound, max_bound))
    feature_importances: dict[str, float] = {}
    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
        for col_name, val in zip(used_cols, fi, strict=False):
            feature_importances[col_name] = float(val)
    elif len(X_train) > 0 and len(used_cols) > 0:
        from sklearn.inspection import permutation_importance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(clf, X_train, y_train, n_repeats=3, random_state=42)
            for col_name, val in zip(used_cols, perm.importances_mean, strict=False):
                feature_importances[col_name] = float(val)

    return {
        "as_of": last_row["date"].iloc[0],
        "prediction": pred,
        "feature_importances": feature_importances,
    }


# ---------- internal: helpers ----------


def _empty_result(
    df, model, feature_set, target_type, ensemble_method, loss, horizon
) -> ForecastEvaluationResult:
    return ForecastEvaluationResult(
        horizon=horizon,
        model=model,
        feature_set=feature_set,
        target_type=target_type,
        ensemble_method=ensemble_method,
        loss=loss,
        feature_cols=feat.feature_columns(feature_set),
        available_feature_cols=[],
    )


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
) -> ForecastEvaluationResult:
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

    ic_val, ic_pval = compute_ic(y_true, y_pred)
    dir_metrics = compute_directional_accuracy(y_true, y_pred)
    dir_acc_low, dir_acc_med, dir_acc_high = compute_regime_directional_accuracy(
        df, valid, y_true, y_pred
    )
    sharpe = compute_sharpe(y_true, y_pred, target_horizon=target_horizon)
    r2 = compute_r2(y_true, y_pred, ic_val=ic_val, method="ic_signed_sq")

    n_models = len(train_rows_list)
    median_model_count_val = (
        int(model_count[model_count > 0].median()) if (model_count > 0).any() else 0
    )
    mean_train_rows = int(np.mean(train_rows_list)) if train_rows_list else 0

    return ForecastEvaluationResult(
        horizon=horizon,
        ic=ic_val,
        ic_pvalue=ic_pval,
        direction_accuracy=dir_metrics["direction_accuracy"],
        baseline_direction_accuracy=dir_metrics["baseline_direction_accuracy"],
        direction_edge=dir_metrics["direction_edge"],
        positive_target_rate=dir_metrics["positive_target_rate"],
        positive_prediction_rate=dir_metrics["positive_prediction_rate"],
        direction_accuracy_low=dir_acc_low,
        direction_accuracy_medium=dir_acc_med,
        direction_accuracy_high=dir_acc_high,
        prediction_count=len(y_true),
        predictions=y_pred,
        y_true=y_true,
        dates=df.loc[valid, "date"],
        sharpe=sharpe,
        r2=r2,
        train_size=train_size,
        test_size=test_size,
        n_models=n_models,
        median_model_count=median_model_count_val,
        mean_train_rows=mean_train_rows,
        model=model,
        feature_set=feature_set,
        target_type=target_type,
        ensemble_method=ensemble_method,
        loss=loss,
        feature_cols=feat.feature_columns(feature_set),
        available_feature_cols=available_cols,
    )
