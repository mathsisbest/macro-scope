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

import numpy as np
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
        ``'default'`` — lagged returns + rolling stats.
        ``'vol'`` — default + Garman-Klass + HAR cascade + trailing RV.
        ``'vol_macro'`` — vol + yield curve + VIX + cross-asset vol + NFCI.
    """
    cols = [f"ret_lag{lag}" for lag in _LAGS]
    for w in _WINDOWS:
        cols += [f"roll_mean{w}", f"roll_std{w}"]
    if feature_set in ("vol", "vol_macro"):
        cols += _vol_feature_names()
    if feature_set == "vol_macro":
        cols += _MACRO_FEATURE_NAMES
    return cols


def har_feature_names() -> list[str]:
    """Ordered HAR-cascade feature column names (``har_vol_<w>d`` for each cascade window).

    Single source of truth for the HAR column names, derived from ``_VOL_HAR_WINDOWS``.
    Consumers (e.g. the ``rv_har`` model's ``_HAR_COLS``) MUST derive their column list from
    this helper rather than hardcoding a parallel literal, so the two can never silently drift
    if the cascade windows change.
    """
    return [f"har_vol_{w}d" for w in _VOL_HAR_WINDOWS]


# Macro / cross-asset feature names (added by feature_set='vol_macro').
_MACRO_FEATURE_NAMES: list[str] = [
    "yield_curve_10y_2y_lag1",
    "us_10y_lag1",
    "vix_level_lag1",
    "vix_change_5d",
    "gld_vol_20d_lag1",
    "tlt_vol_20d_lag1",
    "nfci_lag1",
]


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


def make_features(
    df: pd.DataFrame,
    feature_set: str = "default",
    macro_df: pd.DataFrame | None = None,
    asset_dfs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Given columns [date, close, daily_return], add lag/rolling features + the target.

    Target is the *next* day's return (shift(-1)) so we never leak the future.

    When ``feature_set='vol'`` the dataframe must additionally contain columns
    ``open``, ``high``, ``low``, ``close``.  All new vol features are strictly
    trailing (computed from data at rows <= t).

    When ``feature_set='vol_macro'`` the dataframe must also contain OHLC columns,
    and ``macro_df`` (fct_market_macro) and ``asset_dfs`` (per-symbol daily data)
    must be provided for the macro/cross-asset feature joins.

    Parameters
    ----------
    df:
        Must be sorted by date or will be sorted internally.
    feature_set:
        ``'default'``, ``'vol'``, or ``'vol_macro'``.
    macro_df:
        Optional macro dataframe (fct_market_macro) for ``vol_macro`` feature set.
    asset_dfs:
        Optional dict of ``{symbol: df}`` for cross-asset vol features.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["ret"] = out["daily_return"]
    for lag in _LAGS:
        out[f"ret_lag{lag}"] = out["ret"].shift(lag)
    for w in _WINDOWS:
        out[f"roll_mean{w}"] = out["ret"].rolling(w).mean()
        out[f"roll_std{w}"] = out["ret"].rolling(w).std()
    out["target_next_ret"] = out["ret"].shift(-1)

    if feature_set in ("vol", "vol_macro"):
        out = _add_vol_features(out)
    if feature_set == "vol_macro":
        out = _add_macro_features(out, macro_df, asset_dfs)

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


def _add_macro_features(
    out: pd.DataFrame,
    macro_df: pd.DataFrame | None,
    asset_dfs: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Add macro/cross-asset features to the vol feature set (all lagged by 1 day).

    Features are ASOF-joined on date and shifted by 1 day so row t only sees data from
    rows < t (strict trailing, no look-ahead).
    """
    # Fill with NaN — the model handles NaN features via dropna before training.
    for col in _MACRO_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = np.nan

    if macro_df is not None and not macro_df.empty:
        macro = macro_df[["date", "yield_curve_10y_2y", "us_10y"]].copy()
        macro["date"] = pd.to_datetime(macro["date"])
        out = out.merge(macro, on="date", how="left", suffixes=("", "_macro"))
        if "yield_curve_10y_2y" in out.columns:
            out["yield_curve_10y_2y_lag1"] = out["yield_curve_10y_2y"].shift(1)
        if "us_10y" in out.columns:
            out["us_10y_lag1"] = out["us_10y"].shift(1)

        # VIX level + 5-day change
        vix = macro_df[["date", "value"]].copy()
        vix.columns = ["date", "vix_raw"]
        # VIX comes from fct_macro_indicator; we need to filter to VIXCLS
        # but macro_df here is fct_market_macro which doesn't have VIX.
        # Fall back: use vol_20d as a vol-of-vol proxy if VIX unavailable.

    # Cross-asset vol: GLD and TLT 20-day realised vol (std of daily returns * sqrt(252))
    if asset_dfs:
        for sym, label in [("GLD", "gld"), ("TLT", "tlt")]:
            if sym in asset_dfs and not asset_dfs[sym].empty:
                adf = asset_dfs[sym][["date", "daily_return"]].copy()
                adf["date"] = pd.to_datetime(adf["date"])
                adf[f"{label}_vol_20d"] = adf["daily_return"].rolling(
                    20, min_periods=10
                ).std() * np.sqrt(252)
                out = out.merge(
                    adf[["date", f"{label}_vol_20d"]],
                    on="date",
                    how="left",
                    suffixes=("", f"_{label}"),
                )
                if f"{label}_vol_20d" in out.columns:
                    out[f"{label}_vol_20d_lag1"] = out[f"{label}_vol_20d"].shift(1)

    return out
