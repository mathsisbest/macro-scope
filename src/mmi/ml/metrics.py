"""Canonical ML evaluation metrics and result schemas for Macro Scope."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


@dataclass
class ForecastEvaluationResult:
    """Type-safe evaluation result schema for ML forecasts."""

    horizon: int | None = None
    ic: float = 0.0
    ic_pvalue: float = 1.0
    direction_accuracy: float = 0.0
    baseline_direction_accuracy: float = 0.0
    direction_edge: float = 0.0
    positive_target_rate: float = 0.0
    positive_prediction_rate: float = 0.0
    direction_accuracy_low: float = np.nan
    direction_accuracy_medium: float = np.nan
    direction_accuracy_high: float = np.nan
    prediction_count: int = 0
    predictions: pd.Series = field(default_factory=pd.Series)
    y_true: pd.Series = field(default_factory=pd.Series)
    dates: pd.Series = field(default_factory=pd.Series)
    sharpe: float = np.nan
    r2: float = 0.0
    train_size: int = 250
    test_size: int = 20
    n_models: int = 0
    median_model_count: int = 0
    mean_train_rows: int = 0
    model: str = "gb"
    feature_set: str = "default"
    target_type: str = "raw"
    ensemble_method: str = "mean"
    loss: str = "squared_error"
    feature_cols: list[str] = field(default_factory=list)
    available_feature_cols: list[str] = field(default_factory=list)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def compute_ic(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Compute Information Coefficient (Pearson Correlation) and p-value."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0, 1.0

    try:
        ic_val, ic_pval = pearsonr(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
        )
        ic_val = float(ic_val) if not pd.isna(ic_val) else 0.0
        ic_pval = float(ic_pval) if not pd.isna(ic_pval) else 1.0
        return ic_val, ic_pval
    except Exception:
        return 0.0, 1.0


def compute_directional_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute directional accuracy, baseline, edge, and hit rates."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)

    if n == 0:
        return {
            "direction_accuracy": np.nan,
            "baseline_direction_accuracy": np.nan,
            "direction_edge": np.nan,
            "positive_target_rate": np.nan,
            "positive_prediction_rate": np.nan,
        }

    direction_tp = int(((yt > 0) & (yp > 0)).sum())
    direction_tn = int(((yt < 0) & (yp < 0)).sum())
    direction_accuracy = float((direction_tp + direction_tn) / n)

    positive_target_rate = float((yt > 0).mean())
    positive_prediction_rate = float((yp > 0).mean())
    negative_target_rate = float((yt < 0).mean())

    baseline_direction_accuracy = float(max(positive_target_rate, negative_target_rate))
    direction_edge = float(direction_accuracy - baseline_direction_accuracy)

    return {
        "direction_accuracy": direction_accuracy,
        "baseline_direction_accuracy": baseline_direction_accuracy,
        "direction_edge": direction_edge,
        "positive_target_rate": positive_target_rate,
        "positive_prediction_rate": positive_prediction_rate,
    }


def compute_regime_directional_accuracy(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[float, float, float]:
    """Compute regime-specific directional accuracy based on 20-day volatility tertiles."""
    dir_acc_low, dir_acc_med, dir_acc_high = np.nan, np.nan, np.nan
    if "ret" not in df.columns:
        return dir_acc_low, dir_acc_med, dir_acc_high

    vol_20d = df["ret"].rolling(20, min_periods=5).std()
    valid_vol = vol_20d.loc[valid_mask]
    if len(valid_vol.dropna()) < 6:
        return dir_acc_low, dir_acc_med, dir_acc_high

    try:
        regimes = pd.qcut(
            valid_vol,
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
        accs = {}
        for reg_key in ["low", "medium", "high"]:
            mask = regimes == reg_key
            if mask.sum() > 0:
                yt = y_true[mask]
                yp = y_pred[mask]
                tp = ((yt > 0) & (yp > 0)).sum()
                tn = ((yt < 0) & (yp < 0)).sum()
                accs[reg_key] = float((tp + tn) / len(yt)) if len(yt) > 0 else np.nan

        return (
            accs.get("low", np.nan),
            accs.get("medium", np.nan),
            accs.get("high", np.nan),
        )
    except Exception:
        return dir_acc_low, dir_acc_med, dir_acc_high


def compute_sharpe(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    target_horizon: int = 1,
) -> float:
    """Compute annualized Sharpe ratio for directional return strategy."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    strategy_ret = np.sign(yp) * yt

    if target_horizon == 1 and strategy_ret.std() > 0:
        return float(strategy_ret.mean() / strategy_ret.std() * np.sqrt(252))
    return float(np.nan)


def compute_r2(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    ic_val: float | None = None,
    method: str = "ic_signed_sq",
) -> float:
    """Compute Out-of-Sample R2 metric.

    Methods:
    - 'ic_signed_sq': ic**2 if ic > 0 else -(ic**2) (Standard Macro Scope directional R2)
    - 'regression': 1.0 - sum((y_true - y_pred)**2) / sum((y_true - mean(y_true))**2)
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) < 3 or np.std(yt) == 0 or np.std(yp) == 0:
        return 0.0

    if method == "ic_signed_sq":
        if ic_val is None:
            ic_val, _ = compute_ic(yt, yp)
        return float(ic_val**2 if ic_val > 0 else -(ic_val**2))
    elif method == "regression":
        ss_res = float(((yt - yp) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    raise ValueError(f"Unknown R2 method '{method}'")
