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
        ``'vol_macro'`` — vol + yield curve + VIX + cross-asset vol.
        ``'vol_rich'`` — vol + macro + kurtosis/skewness + vol-of-vol + correlations
                         + calendar effects.
    """
    cols = [f"ret_lag{lag}" for lag in _LAGS]
    for w in _WINDOWS:
        cols += [f"roll_mean{w}", f"roll_std{w}"]
    if feature_set in ("vol", "vol_macro", "vol_rich"):
        cols += _vol_feature_names()
    if feature_set in ("vol_macro", "vol_rich"):
        cols += _MACRO_FEATURE_NAMES
    if feature_set == "vol_rich":
        cols += _RICH_FEATURE_NAMES
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
    # Yield curve
    "yc_10y2y_lag1",
    "yc_10y2y_change_20d",
    "yc_slope_zscore_60d",
    # Treasury yields
    "us_10y_lag1",
    "us_2y_lag1",
    "us_3m_lag1",
    "us_10y_change_20d",
    # Policy
    "fedfunds_lag1",
    "fedfunds_change_60d",
    # VIX / risk
    "vix_level_lag1",
    "vix_change_5d",
    "vix_zscore_60d",
    # Oil / inflation
    "wti_change_20d",
    # Dollar
    "dollar_change_20d",
    # Employment (leading)
    "claims_change_4w",
    # Financial conditions
    "nfci_lag1",
    "nfci_change_20d",
    # Cross-asset vol
    "gld_vol_20d_lag1",
    "tlt_vol_20d_lag1",
]

# Rich feature names (added by feature_set='vol_rich').
_RICH_FEATURE_NAMES: list[str] = [
    # Higher moments
    "ret_kurt_20d",
    "ret_skew_20d",
    "ret_max_20d",
    # Vol-of-vol term structure
    "vol_of_vol_5d",
    "vol_of_vol_22d",
    "vol_dispersion",  # short-term vol minus long-term vol
    # Cross-asset correlations
    "corr_spy_tlt_20d",
    "corr_spy_gld_20d",
    # Calendar
    "day_of_week",
    "month_of_year",
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

    if feature_set in ("vol", "vol_macro", "vol_rich"):
        out = _add_vol_features(out)
    if feature_set in ("vol_macro", "vol_rich"):
        out = _add_macro_features(out, macro_df, asset_dfs)
    if feature_set == "vol_rich":
        out = _add_rich_features(out, asset_dfs)

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
    """Add comprehensive macro/cross-asset features (all lagged by 1 day).

    The macro_df is a wide-format DataFrame from fct_macro_indicator with columns for each
    FRED series_id. We ASOF-merge onto SPY dates, then compute derived features.
    """
    out["date"] = pd.to_datetime(out["date"])

    # ASOF-merge all FRED series onto SPY trading dates
    if macro_df is not None and not macro_df.empty and "date" in macro_df.columns:
        macro = macro_df.copy()
        macro["date"] = pd.to_datetime(macro["date"])
        out = pd.merge_asof(
            out.sort_values("date"),
            macro.sort_values("date"),
            on="date",
            direction="backward",
        )

    # --- Yield curve features ---
    if "T10Y2Y" in out.columns:
        yc = out["T10Y2Y"]
        out["yc_10y2y_lag1"] = yc.shift(1)
        out["yc_10y2y_change_20d"] = yc.diff(20).shift(1)
        # Z-score of yield curve slope (60-day rolling)
        yc_mean = yc.rolling(60, min_periods=20).mean()
        yc_std = yc.rolling(60, min_periods=20).std()
        out["yc_slope_zscore_60d"] = ((yc - yc_mean) / yc_std.replace(0, np.nan)).shift(1)

    # --- Treasury yield features ---
    for sid, name in [("DGS10", "us_10y"), ("DGS2", "us_2y"), ("DGS3MO", "us_3m")]:
        if sid in out.columns:
            out[f"{name}_lag1"] = out[sid].shift(1)
    if "DGS10" in out.columns:
        out["us_10y_change_20d"] = out["DGS10"].diff(20).shift(1)

    # --- Policy ---
    if "FEDFUNDS" in out.columns:
        out["fedfunds_lag1"] = out["FEDFUNDS"].shift(1)
        out["fedfunds_change_60d"] = out["FEDFUNDS"].diff(60).shift(1)

    # --- VIX / risk ---
    if "VIXCLS" in out.columns:
        vix = out["VIXCLS"]
        out["vix_level_lag1"] = vix.shift(1)
        out["vix_change_5d"] = vix.diff(5).shift(1)
        vix_mean = vix.rolling(60, min_periods=20).mean()
        vix_std = vix.rolling(60, min_periods=20).std()
        out["vix_zscore_60d"] = ((vix - vix_mean) / vix_std.replace(0, np.nan)).shift(1)

    # --- Oil ---
    if "DCOILWTICO" in out.columns:
        out["wti_change_20d"] = out["DCOILWTICO"].pct_change(20).shift(1)

    # --- Dollar ---
    if "DTWEXBGS" in out.columns:
        out["dollar_change_20d"] = out["DTWEXBGS"].pct_change(20).shift(1)

    # --- Employment (leading indicator) ---
    if "ICSA" in out.columns:
        # 4-week average change in initial claims
        claims_4w = out["ICSA"].rolling(4, min_periods=1).mean()
        out["claims_change_4w"] = claims_4w.diff(4).shift(1)

    # --- Financial conditions ---
    if "NFCI" in out.columns:
        out["nfci_lag1"] = out["NFCI"].shift(1)
        out["nfci_change_20d"] = out["NFCI"].diff(20).shift(1)

    # --- Forward-fill all macro features ---
    for col in _MACRO_FEATURE_NAMES:
        if col in out.columns:
            out[col] = out[col].ffill()

    # --- Cross-asset vol: GLD and TLT ---
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

    # Ensure all expected columns exist (filled with NaN if not computed)
    for col in _MACRO_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = np.nan

    return out


def _add_rich_features(
    out: pd.DataFrame,
    asset_dfs: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Add higher moments, vol-of-vol, cross-asset correlations, and calendar effects.

    All features are strictly trailing (no look-ahead).
    """
    ret = out["ret"]

    # --- Higher moments (20d rolling) ---
    out["ret_kurt_20d"] = ret.rolling(20, min_periods=10).kurt()
    out["ret_skew_20d"] = ret.rolling(20, min_periods=10).skew()
    out["ret_max_20d"] = ret.rolling(20, min_periods=10).max()

    # --- Vol-of-vol term structure ---
    if "gk_vol" in out.columns:
        gk = out["gk_vol"]
        out["vol_of_vol_5d"] = gk.shift(1).rolling(5, min_periods=3).std()
        out["vol_of_vol_22d"] = gk.shift(1).rolling(22, min_periods=10).std()
        short_vol = gk.shift(1).rolling(5, min_periods=3).mean()
        long_vol = gk.shift(1).rolling(22, min_periods=10).mean()
        out["vol_dispersion"] = short_vol - long_vol

    # --- Cross-asset correlations (20d rolling) ---
    for label in ["gld", "tlt"]:
        col_name = f"corr_spy_{label}_20d"
        if col_name not in out.columns:
            out[col_name] = np.nan
    if asset_dfs:
        for sym, label in [("GLD", "gld"), ("TLT", "tlt")]:
            if sym in asset_dfs and not asset_dfs[sym].empty:
                adf = asset_dfs[sym][["date", "daily_return"]].copy()
                adf["date"] = pd.to_datetime(adf["date"])
                adf = adf.rename(columns={"daily_return": f"{label}_ret"})
                out = out.merge(adf, on="date", how="left", suffixes=("", f"_{label}"))
                if f"{label}_ret" in out.columns:
                    combined = pd.concat([ret, out[f"{label}_ret"]], axis=1)
                    combined.columns = ["spy_ret", f"{label}_ret"]
                    out[f"corr_spy_{label}_20d"] = (
                        combined["spy_ret"]
                        .rolling(20, min_periods=10)
                        .corr(combined[f"{label}_ret"])
                    )

    # --- Macro regime: yield curve slope change (now in _add_macro_features) ---

    # --- Calendar effects ---
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"])
        out["day_of_week"] = dates.dt.dayofweek / 4.0
        out["month_of_year"] = dates.dt.month / 12.0

    # Forward-fill correlation features (available from GLD/TLT inception onward)
    for col in ["corr_spy_tlt_20d", "corr_spy_gld_20d", "yc_slope_change_20d"]:
        if col in out.columns:
            out[col] = out[col].ffill()

    return out
