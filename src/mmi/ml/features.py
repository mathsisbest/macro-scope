"""Feature engineering for return forecasting.

Features are intentionally simple and explainable (lagged returns + rolling stats):
the portfolio point is sound methodology — leakage-free features and honest evaluation —
not a giant black-box model.
"""

from __future__ import annotations

import pandas as pd

_LAGS = [1, 2, 3, 5]
_WINDOWS = [5, 10, 20]


def feature_columns() -> list[str]:
    cols = [f"ret_lag{lag}" for lag in _LAGS]
    for w in _WINDOWS:
        cols += [f"roll_mean{w}", f"roll_std{w}"]
    return cols


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given columns [date, close, daily_return], add lag/rolling features + the target.

    Target is the *next* day's return (shift(-1)) so we never leak the future.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["ret"] = out["daily_return"]
    for lag in _LAGS:
        out[f"ret_lag{lag}"] = out["ret"].shift(lag)
    for w in _WINDOWS:
        out[f"roll_mean{w}"] = out["ret"].rolling(w).mean()
        out[f"roll_std{w}"] = out["ret"].rolling(w).std()
    out["target_next_ret"] = out["ret"].shift(-1)
    return out
