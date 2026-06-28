"""Feature engineering for return forecasting.

Features are intentionally simple and explainable (lagged returns + rolling stats):
the portfolio point is sound methodology — leakage-free features and honest evaluation —
not a giant black-box model.

Optional volatility feature set (feature_set='vol'):
  - Garman-Klass daily vol estimate from OHLC
  - HAR cascade: trailing 1d / 5d / 22d averages of that vol
  - Longer trailing realized-vol windows (10d, 63d)
  All features at row t use ONLY data <= t (strict trailing, no shift(-k) except target).
"""

from __future__ import annotations

import math

import pandas as pd

_LAGS = [1, 2, 3, 5]
_WINDOWS = [5, 10, 20]

# Volatility feature set
_VOL_HAR_WINDOWS = [1, 5, 22]
_VOL_EXTRA_WINDOWS = [10, 63]

# Garman-Klass constant: 0.5·ln2·(ln(H/L))² - (2·ln2-1)·(ln(C/O))²
# expressed as lambda below; computed per-row via apply for clarity.
_GK_C1 = 0.5
_GK_C2 = 2.0 * math.log(2) - 1.0  # ≈ 0.3863


def feature_columns(feature_set: str = "default") -> list[str]:
    """Return the ordered list of feature column names.

    Parameters
    ----------
    feature_set:
        ``'default'`` (unchanged legacy set) or ``'vol'`` (adds Garman-Klass +
        HAR cascade + extra realized-vol windows on top of the default set).
    """
    cols = [f"ret_lag{lag}" for lag in _LAGS]
    for w in _WINDOWS:
        cols += [f"roll_mean{w}", f"roll_std{w}"]
    if feature_set == "vol":
        cols += _vol_feature_names()
    return cols


def har_feature_names() -> list[str]:
    """Ordered HAR-cascade feature column names (``har_vol_<w>d`` for each cascade window).

    Single source of truth for the HAR column names, derived from ``_VOL_HAR_WINDOWS``.
    Consumers (e.g. the ``rv_har`` model's ``_HAR_COLS``) MUST derive their column list from
    this helper rather than hardcoding a parallel literal, so the two can never silently drift
    if the cascade windows change.
    """
    return [f"har_vol_{w}d" for w in _VOL_HAR_WINDOWS]


def _vol_feature_names() -> list[str]:
    names = ["gk_vol"]
    names += har_feature_names()
    for w in _VOL_EXTRA_WINDOWS:
        names.append(f"rv_trail_{w}d")
    return names


def _garman_klass_vol(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Garman-Klass daily volatility estimate (annualised square-root form not applied; raw daily).

    GK = sqrt(0.5·(ln H/L)² - (2·ln2-1)·(ln C/O)²)

    All inputs must be positive prices.  Returns NaN where any input is NaN or <= 0.
    """
    import numpy as np

    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        gk2 = _GK_C1 * log_hl**2 - _GK_C2 * log_co**2
        # Clamp negatives (numerical noise) to 0 before sqrt
        gk2 = gk2.clip(lower=0)
        return np.sqrt(gk2)


def make_features(df: pd.DataFrame, feature_set: str = "default") -> pd.DataFrame:
    """Given columns [date, close, daily_return], add lag/rolling features + the target.

    Target is the *next* day's return (shift(-1)) so we never leak the future.

    When ``feature_set='vol'`` the dataframe must additionally contain columns
    ``open``, ``high``, ``low``, ``close``.  All new vol features are strictly
    trailing (computed from data at rows <= t).

    Parameters
    ----------
    df:
        Must be sorted by date or will be sorted internally.
    feature_set:
        ``'default'`` or ``'vol'``.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["ret"] = out["daily_return"]
    for lag in _LAGS:
        out[f"ret_lag{lag}"] = out["ret"].shift(lag)
    for w in _WINDOWS:
        out[f"roll_mean{w}"] = out["ret"].rolling(w).mean()
        out[f"roll_std{w}"] = out["ret"].rolling(w).std()
    out["target_next_ret"] = out["ret"].shift(-1)

    if feature_set == "vol":
        out = _add_vol_features(out)

    return out


def _add_vol_features(out: pd.DataFrame) -> pd.DataFrame:
    """Add Garman-Klass vol + HAR cascade + extra trailing RV windows (all leakage-free)."""
    # Garman-Klass: uses same-day OHLC — row t features describe day-t's own price range.
    # This is the conventional GK usage: today's intraday range proxy is known at close.
    # The *target* for Wave-2 (forward RV) is always label = RV_{t+1..t+5} so no leakage.
    gk = _garman_klass_vol(out["open"], out["high"], out["low"], out["close"])
    out["gk_vol"] = gk

    # HAR cascade — trailing rolling means of gk_vol; min_periods=1 so early rows aren't all NaN
    for w in _VOL_HAR_WINDOWS:
        # shift(1) so that at row t we use only gk_vol from rows < t (strict past).
        # For the daily lag (1d) this is just yesterday's GK vol.
        out[f"har_vol_{w}d"] = gk.shift(1).rolling(w, min_periods=1).mean()

    # Longer trailing realized-vol windows (std of lagged GK vol)
    for w in _VOL_EXTRA_WINDOWS:
        out[f"rv_trail_{w}d"] = gk.shift(1).rolling(w, min_periods=1).std()

    return out
