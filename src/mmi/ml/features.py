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
        ``'vol_medium'`` — vol + macro subset + key rich features (~27 total,
                           fast enough for portfolio backtest).
        ``'vol_rich'`` — vol + macro + kurtosis/skewness + vol-of-vol + correlations
                         + calendar effects (50 features).
        ``'vol_rich_plus'`` — vol_rich + unused FRED macro + breakeven inflation +
                         recession prob + cross-asset spreads + mom_rev (75 features).
    """
    cols = [f"ret_lag{lag}" for lag in _LAGS]
    for w in _WINDOWS:
        cols += [f"roll_mean{w}", f"roll_std{w}"]
    if feature_set in ("vol", "vol_macro", "vol_rich", "vol_medium", "vol_rich_plus"):
        cols += _vol_feature_names()
    if feature_set in ("vol_macro", "vol_rich", "vol_medium", "vol_rich_plus"):
        cols += _MACRO_FEATURE_NAMES
    if feature_set in ("vol_rich", "vol_rich_plus"):
        cols += _RICH_FEATURE_NAMES
    if feature_set == "vol_medium":
        cols += _MEDIUM_FEATURE_NAMES
    if feature_set == "vol_rich_plus":
        cols += _EXTENDED_FEATURE_NAMES
    if feature_set == "mom_rev":
        cols += _MOM_REV_FEATURE_NAMES
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
# Daily/weekly features only — monthly/quarterly series (CPI, GDP, UNRATE etc.) are too
# low-frequency and create noise when forward-filled to daily dates. The features below
# have enough granularity to provide genuine predictive signal.
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
    # Oil
    "wti_change_20d",
    # Dollar
    "dollar_change_20d",
    # Employment (weekly)
    "claims_change_4w",
    # Financial conditions
    "nfci_lag1",
    "nfci_change_20d",
    # Cross-asset vol
    "gld_vol_20d_lag1",
    "tlt_vol_20d_lag1",
    # Valuation (Shiller CAPE + yields)
    "cape",
    "excess_cape_yield",
    "div_yield",
    "earn_yield",
]

# Medium feature names (added by feature_set='vol_medium' — fast, key predictors only).
_MEDIUM_FEATURE_NAMES: list[str] = [
    # Vol-of-vol term structure (key regime signal)
    "vol_of_vol_5d",
    "vol_of_vol_22d",
    "vol_dispersion",
    # Momentum + mean-reversion (return-prediction signals)
    "ret_momentum_63d",
    "ret_reversal_5d",
    "ret_reversal_20d",
    "ret_vol_ratio_5d_20d",
    "ret_trend_strength",
    # Interaction: VIX × yield curve
    "vix_x_yc_slope",
    "nfci_x_dollar_zscore",
]

# Rich feature names (added by feature_set='vol_rich').
_EXTENDED_FEATURE_NAMES: list[str] = [
    # Unused FRED macro series (exist in macro_df, not wired into any feature set)
    "indpro_growth_1y",
    "cpi_inflation_1y",
    "core_pce_inflation_1y",
    "unrate_level",
    "unrate_change_3m",
    "payems_growth_1m",
    "payems_growth_1y",
    "m2_growth_1y",
    "walcl_growth_4w",
    "umcsent_level",
    "umcsent_change_1m",
    "sahm_rule_level",
    # Breakeven inflation (TLT - TIP return spread)
    "breakeven_inflation_20d",
    "breakeven_inflation_60d",
    "breakeven_inflation_change_20d",
    # Recession probability
    "recession_prob_lag1",
    "recession_prob_change_20d",
    # Cross-asset return spreads
    "spy_gld_spread_20d",
    "spy_tlt_spread_20d",
    "gld_tlt_spread_20d",
    "vea_spy_spread_20d",
    "spy_tip_spread_20d",
    "btc_spy_spread_20d",
    # Momentum features not in vol_rich (unique ones from mom_rev set)
    "mom_21d",
    "mom_accel",
    "rev_10d",
    "ret_zscore_20d",
    "ret_zscore_60d",
    "dist_from_mean_20d",
    "dist_from_mean_60d",
    "trend_60d",
]


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
    # Cross-asset regime signals
    "corr_spy_tlt_zscore_60d",
    "corr_spy_gld_zscore_60d",
    "dollar_zscore_60d",
    "cross_asset_dispersion_20d",
    "equity_bond_spread_20d",
    # Calendar
    "day_of_week",
    "month_of_year",
    # Interaction features (cross-terms of strongest predictors)
    "vix_x_yc_slope",
    "vol_disp_x_vol_of_vol",
    "nfci_x_dollar_zscore",
    "skew_x_vol_dispersion",
    # Return-prediction features: momentum + mean-reversion + trend
    "ret_momentum_63d",  # 3-month cumulative return (momentum signal)
    "ret_momentum_126d",  # 6-month cumulative return (intermediate momentum)
    "ret_momentum_252d",  # 12-month cumulative return (long-term momentum)
    "ret_reversal_5d",  # 5-day cumulative return (short-term reversal)
    "ret_reversal_20d",  # 20-day cumulative return (mean-reversion signal)
    "ret_vol_ratio_5d_20d",  # short-term vol / long-term vol (regime shift proxy)
    "ret_trend_strength",  # |mean_20d| / std_20d (signal-to-noise ratio)
    "ret_autocorr_20d",  # 20-day return autocorrelation (persistence signal)
    "yc_slope_x_vix",  # yield curve slope × VIX level (macro regime)
    "momentum_x_vol",  # 6m momentum × 20d vol (risk-adjusted momentum)
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
        ``'default'``, ``'vol'``, ``'vol_macro'``, ``'vol_medium'``, ``'vol_rich'``,
        or ``'vol_rich_plus'``.
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

    if feature_set == "mom_rev":
        out = _add_mom_rev_features(out)
    if feature_set in ("vol", "vol_macro", "vol_rich", "vol_medium", "vol_rich_plus"):
        out = _add_vol_features(out)
    if feature_set in ("vol_macro", "vol_rich", "vol_medium", "vol_rich_plus"):
        out = _add_macro_features(out, macro_df, asset_dfs)
    if feature_set in ("vol_rich", "vol_rich_plus"):
        out = _add_rich_features(out, asset_dfs)
    if feature_set == "vol_medium":
        out = _add_medium_features(out)
    if feature_set == "vol_rich_plus":
        out = _add_extended_features(out, asset_dfs)

    return out


_MOM_REV_FEATURE_NAMES: list[str] = [
    "mom_21d",
    "mom_63d",
    "mom_126d",
    "mom_252d",
    "mom_accel",
    "rev_5d",
    "rev_10d",
    "ret_zscore_20d",
    "ret_zscore_60d",
    "dist_from_mean_20d",
    "dist_from_mean_60d",
    "trend_20d",
    "trend_60d",
]


def _add_mom_rev_features(out: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and mean-reversion features (all leakage-free)."""
    ret = out["ret"]

    # Momentum: cumulative returns
    out["mom_21d"] = ret.rolling(21, min_periods=10).sum().shift(1)
    out["mom_63d"] = ret.rolling(63, min_periods=30).sum().shift(1)
    out["mom_126d"] = ret.rolling(126, min_periods=60).sum().shift(1)
    out["mom_252d"] = ret.rolling(252, min_periods=120).sum().shift(1)
    out["mom_accel"] = out["mom_63d"] - out["mom_63d"].shift(21)

    # Mean-reversion: short-term reversals
    out["rev_5d"] = -ret.rolling(5, min_periods=3).sum().shift(1)
    out["rev_10d"] = -ret.rolling(10, min_periods=5).sum().shift(1)

    # Z-scores
    for w, name in [(20, "ret_zscore_20d"), (60, "ret_zscore_60d")]:
        mean = ret.rolling(w, min_periods=w // 2).mean()
        std = ret.rolling(w, min_periods=w // 2).std()
        out[name] = ((ret - mean) / std.replace(0, np.nan)).shift(1)

    # Distance from rolling mean
    for w, name in [(20, "dist_from_mean_20d"), (60, "dist_from_mean_60d")]:
        mean = ret.rolling(w, min_periods=w // 2).mean()
        out[name] = (ret - mean).shift(1)

    # Trend strength
    for w, name in [(20, "trend_20d"), (60, "trend_60d")]:
        mean = ret.rolling(w, min_periods=w // 2).mean()
        std = ret.rolling(w, min_periods=w // 2).std()
        out[name] = (mean.abs() / std.replace(0, np.nan)).shift(1)

    return out


def _add_vol_features(out: pd.DataFrame) -> pd.DataFrame:
    """Add Garman-Klass vol + HAR cascade + extra trailing RV windows (all leakage-free)."""
    gk = _garman_klass_vol(out["open"], out["high"], out["low"], out["close"])
    out["gk_vol"] = gk

    for w in _VOL_HAR_WINDOWS:
        out[f"har_vol_{w}d"] = gk.shift(1).rolling(w, min_periods=1).mean()

    for w in _VOL_EXTRA_WINDOWS:
        out[f"rv_trail_{w}d"] = gk.shift(1).rolling(w, min_periods=1).std()

    return out


def _add_macro_features(
    out: pd.DataFrame,
    macro_df: pd.DataFrame | None,
    asset_dfs: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Add comprehensive macro/cross-asset features (all lagged by 1 day)."""
    out["date"] = pd.to_datetime(out["date"])

    if macro_df is not None and not macro_df.empty and "date" in macro_df.columns:
        macro = macro_df.copy()
        macro["date"] = pd.to_datetime(macro["date"]).astype("datetime64[ns]")
        out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
        out = pd.merge_asof(
            out.sort_values("date"),
            macro.sort_values("date"),
            on="date",
            direction="backward",
        )

    if "T10Y2Y" in out.columns:
        yc = out["T10Y2Y"]
        out["yc_10y2y_lag1"] = yc.shift(1)
        out["yc_10y2y_change_20d"] = yc.diff(20).shift(1)
        yc_mean = yc.rolling(60, min_periods=20).mean()
        yc_std = yc.rolling(60, min_periods=20).std()
        out["yc_slope_zscore_60d"] = ((yc - yc_mean) / yc_std.replace(0, np.nan)).shift(1)

    for sid, name in [("DGS10", "us_10y"), ("DGS2", "us_2y"), ("DGS3MO", "us_3m")]:
        if sid in out.columns:
            out[f"{name}_lag1"] = out[sid].shift(1)
    if "DGS10" in out.columns:
        out["us_10y_change_20d"] = out["DGS10"].diff(20).shift(1)

    if "FEDFUNDS" in out.columns:
        out["fedfunds_lag1"] = out["FEDFUNDS"].shift(1)
        out["fedfunds_change_60d"] = out["FEDFUNDS"].diff(60).shift(1)

    if "VIXCLS" in out.columns:
        vix = out["VIXCLS"]
        out["vix_level_lag1"] = vix.shift(1)
        out["vix_change_5d"] = vix.diff(5).shift(1)
        vix_mean = vix.rolling(60, min_periods=20).mean()
        vix_std = vix.rolling(60, min_periods=20).std()
        out["vix_zscore_60d"] = ((vix - vix_mean) / vix_std.replace(0, np.nan)).shift(1)

    if "DCOILWTICO" in out.columns:
        out["wti_change_20d"] = out["DCOILWTICO"].pct_change(20).shift(1)

    if "DTWEXBGS" in out.columns:
        out["dollar_change_20d"] = out["DTWEXBGS"].pct_change(20).shift(1)

    if "ICSA" in out.columns:
        claims_4w = out["ICSA"].rolling(4, min_periods=1).mean()
        out["claims_change_4w"] = claims_4w.diff(4).shift(1)

    if "NFCI" in out.columns:
        out["nfci_lag1"] = out["NFCI"].shift(1)
        out["nfci_change_20d"] = out["NFCI"].diff(20).shift(1)

    for col in _MACRO_FEATURE_NAMES:
        if col in out.columns:
            out[col] = out[col].ffill()

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

    for col in _MACRO_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = np.nan

    return out


def _add_medium_features(out: pd.DataFrame) -> pd.DataFrame:
    """Add a fast subset of rich features for the vol_medium feature set.

    Includes vol-of-vol term structure, momentum/reversal signals, trend strength,
    and key interaction features. Cheap to compute (no per-symbol cross-asset corrs).
    """
    ret = out["ret"]

    if "gk_vol" in out.columns:
        gk = out["gk_vol"]
        out["vol_of_vol_5d"] = gk.shift(1).rolling(5, min_periods=3).std()
        out["vol_of_vol_22d"] = gk.shift(1).rolling(22, min_periods=10).std()
        short_vol = gk.shift(1).rolling(5, min_periods=3).mean()
        long_vol = gk.shift(1).rolling(22, min_periods=10).mean()
        out["vol_dispersion"] = short_vol - long_vol

    out["ret_momentum_63d"] = ret.rolling(63, min_periods=31).sum().shift(1)

    for w, name in [(5, "ret_reversal_5d"), (20, "ret_reversal_20d")]:
        out[name] = ret.rolling(w, min_periods=w).sum().shift(1)

    vol_5d = ret.rolling(5, min_periods=3).std()
    vol_20d = ret.rolling(20, min_periods=10).std()
    out["ret_vol_ratio_5d_20d"] = (vol_5d / vol_20d.replace(0, np.nan)).shift(1)

    ret_mean_20d = ret.rolling(20, min_periods=10).mean()
    ret_std_20d = ret.rolling(20, min_periods=10).std()
    out["ret_trend_strength"] = (ret_mean_20d.abs() / ret_std_20d.replace(0, np.nan)).shift(1)

    if "DTWEXBGS" in out.columns and "dollar_zscore_60d" not in out.columns:
        dollar = out["DTWEXBGS"]
        d_mean = dollar.rolling(60, min_periods=20).mean()
        d_std = dollar.rolling(60, min_periods=20).std()
        out["dollar_zscore_60d"] = ((dollar - d_mean) / d_std.replace(0, np.nan)).shift(1)

    if "vix_zscore_60d" in out.columns and "yc_slope_zscore_60d" in out.columns:
        out["vix_x_yc_slope"] = out["vix_zscore_60d"] * out["yc_slope_zscore_60d"]
    if "nfci_lag1" in out.columns and "dollar_zscore_60d" in out.columns:
        out["nfci_x_dollar_zscore"] = out["nfci_lag1"] * out["dollar_zscore_60d"]

    for col in _MEDIUM_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = np.nan

    return out


def _add_rich_features(
    out: pd.DataFrame,
    asset_dfs: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Add higher moments, vol-of-vol, cross-asset correlations, and calendar effects."""
    ret = out["ret"]

    out["ret_kurt_20d"] = ret.rolling(20, min_periods=10).kurt()
    out["ret_skew_20d"] = ret.rolling(20, min_periods=10).skew()
    out["ret_max_20d"] = ret.rolling(20, min_periods=10).max()

    if "gk_vol" in out.columns:
        gk = out["gk_vol"]
        out["vol_of_vol_5d"] = gk.shift(1).rolling(5, min_periods=3).std()
        out["vol_of_vol_22d"] = gk.shift(1).rolling(22, min_periods=10).std()
        short_vol = gk.shift(1).rolling(5, min_periods=3).mean()
        long_vol = gk.shift(1).rolling(22, min_periods=10).mean()
        out["vol_dispersion"] = short_vol - long_vol

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

    for label in ["gld", "tlt"]:
        corr_col = f"corr_spy_{label}_20d"
        zscore_col = f"corr_spy_{label}_zscore_60d"
        if corr_col in out.columns:
            c = out[corr_col]
            c_mean = c.rolling(60, min_periods=20).mean()
            c_std = c.rolling(60, min_periods=20).std()
            out[zscore_col] = ((c - c_mean) / c_std.replace(0, np.nan)).shift(1)

    if "DTWEXBGS" in out.columns:
        dollar = out["DTWEXBGS"]
        d_mean = dollar.rolling(60, min_periods=20).mean()
        d_std = dollar.rolling(60, min_periods=20).std()
        out["dollar_zscore_60d"] = ((dollar - d_mean) / d_std.replace(0, np.nan)).shift(1)

    if asset_dfs:
        ret_cols = []
        for _sym, label in [("GLD", "gld"), ("TLT", "tlt")]:
            if f"{label}_ret" in out.columns:
                ret_cols.append(f"{label}_ret")
        if ret_cols:
            disp_raw = out[ret_cols].std(axis=1).rolling(20, min_periods=10).mean()
            out["cross_asset_dispersion_20d"] = disp_raw.shift(1)

    if "tlt_ret" in out.columns:
        spread_raw = (ret - out["tlt_ret"]).rolling(20, min_periods=10).mean()
        out["equity_bond_spread_20d"] = spread_raw.shift(1)

    if "date" in out.columns:
        dates = pd.to_datetime(out["date"])
        out["day_of_week"] = dates.dt.dayofweek / 4.0
        out["month_of_year"] = dates.dt.month / 12.0

    if "vix_zscore_60d" in out.columns and "yc_slope_zscore_60d" in out.columns:
        out["vix_x_yc_slope"] = out["vix_zscore_60d"] * out["yc_slope_zscore_60d"]
    if "vol_dispersion" in out.columns and "vol_of_vol_22d" in out.columns:
        out["vol_disp_x_vol_of_vol"] = out["vol_dispersion"] * out["vol_of_vol_22d"]
    if "nfci_lag1" in out.columns and "dollar_zscore_60d" in out.columns:
        out["nfci_x_dollar_zscore"] = out["nfci_lag1"] * out["dollar_zscore_60d"]
    if "ret_skew_20d" in out.columns and "vol_dispersion" in out.columns:
        out["skew_x_vol_dispersion"] = out["ret_skew_20d"] * out["vol_dispersion"]

    for w, name in [
        (63, "ret_momentum_63d"),
        (126, "ret_momentum_126d"),
        (252, "ret_momentum_252d"),
    ]:
        out[name] = ret.rolling(w, min_periods=w // 2).sum().shift(1)

    for w, name in [(5, "ret_reversal_5d"), (20, "ret_reversal_20d")]:
        out[name] = ret.rolling(w, min_periods=w).sum().shift(1)

    vol_5d = ret.rolling(5, min_periods=3).std()
    vol_20d = ret.rolling(20, min_periods=10).std()
    out["ret_vol_ratio_5d_20d"] = (vol_5d / vol_20d.replace(0, np.nan)).shift(1)

    ret_mean_20d = ret.rolling(20, min_periods=10).mean()
    ret_std_20d = ret.rolling(20, min_periods=10).std()
    out["ret_trend_strength"] = (ret_mean_20d.abs() / ret_std_20d.replace(0, np.nan)).shift(1)

    out["ret_autocorr_20d"] = (
        ret.rolling(20, min_periods=15)
        .apply(lambda x: x.autocorr(lag=1) if len(x) > 5 else np.nan, raw=False)
        .shift(1)
    )

    if "yc_slope_zscore_60d" in out.columns and "vix_level_lag1" in out.columns:
        out["yc_slope_x_vix"] = out["yc_slope_zscore_60d"] * out["vix_level_lag1"]
    if "ret_momentum_63d" in out.columns and "gk_vol" in out.columns:
        out["momentum_x_vol"] = out["ret_momentum_63d"] * out["gk_vol"]

    for col in ["corr_spy_tlt_20d", "corr_spy_gld_20d"]:
        if col in out.columns:
            out[col] = out[col].ffill()

    return out


def _add_extended_features(
    out: pd.DataFrame,
    asset_dfs: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Add extended features for vol_rich_plus: unused FRED, breakeven inflation,
    recession probability, cross-asset spreads, and unique mom_rev features.

    All features are leakage-free (shifted by 1). The function gracefully handles
    missing data sources — if a required FRED column or asset isn't available, the
    corresponding features are filled as NaN.
    """
    ret = out["ret"]

    # --- Unused FRED macro series ---
    fred_unused = [
        # (FRED column, feature name, transform)
        ("INDPRO", "indpro_growth_1y", lambda s: s.pct_change(252)),
        ("CPIAUCSL", "cpi_inflation_1y", lambda s: s.pct_change(252)),
        ("PCEPILFE", "core_pce_inflation_1y", lambda s: s.pct_change(252)),
        ("UNRATE", "unrate_level", lambda s: s),
        ("UNRATE", "unrate_change_3m", lambda s: s.diff(63)),
        ("PAYEMS", "payems_growth_1m", lambda s: s.pct_change(21)),
        ("PAYEMS", "payems_growth_1y", lambda s: s.pct_change(252)),
        ("M2SL", "m2_growth_1y", lambda s: s.pct_change(252)),
        ("WALCL", "walcl_growth_4w", lambda s: s.pct_change(20)),
        ("UMCSENT", "umcsent_level", lambda s: s),
        ("UMCSENT", "umcsent_change_1m", lambda s: s.diff(21)),
        ("SAHMREALTIME", "sahm_rule_level", lambda s: s),
    ]
    for fred_col, feat_name, transform in fred_unused:
        if fred_col in out.columns:
            out[feat_name] = transform(out[fred_col]).shift(1)

    # --- Breakeven inflation (TLT - TIP return spread) ---
    if asset_dfs:
        for sym, label in [("TLT", "tlt"), ("TIP", "tip")]:
            if sym not in asset_dfs or asset_dfs[sym].empty:
                continue
            adf = asset_dfs[sym][["date", "daily_return"]].copy()
            adf["date"] = pd.to_datetime(adf["date"])
            adf = adf.rename(columns={"daily_return": f"{label}_ret"})
            out = out.merge(adf, on="date", how="left", suffixes=("", f"_{label}"))

        if "tlt_ret" in out.columns and "tip_ret" in out.columns:
            be_raw = out["tlt_ret"] - out["tip_ret"]
            out["breakeven_inflation_20d"] = be_raw.rolling(20, min_periods=10).mean().shift(1)
            out["breakeven_inflation_60d"] = be_raw.rolling(60, min_periods=20).mean().shift(1)
            out["breakeven_inflation_change_20d"] = out["breakeven_inflation_20d"].diff(20)

    # --- Recession probability ---
    if "recession_prob" in out.columns:
        out["recession_prob_lag1"] = out["recession_prob"].shift(1)
        out["recession_prob_change_20d"] = out["recession_prob"].diff(20).shift(1)

    # --- Cross-asset return spreads ---
    if asset_dfs:
        extra_symbols = [
            ("GLD", "gld"),
            ("TLT", "tlt"),
            ("VEA", "vea"),
            ("TIP", "tip"),
            ("BTC", "btc"),
        ]
        for sym, label in extra_symbols:
            if sym not in asset_dfs or asset_dfs[sym].empty:
                continue
            if f"{label}_ret" in out.columns:
                continue
            adf = asset_dfs[sym][["date", "daily_return"]].copy()
            adf["date"] = pd.to_datetime(adf["date"])
            adf = adf.rename(columns={"daily_return": f"{label}_ret"})
            out = out.merge(adf, on="date", how="left", suffixes=("", f"_{label}"))

        for _sym, label, feat_name in [
            ("GLD", "gld", "spy_gld_spread_20d"),
            ("TLT", "tlt", "spy_tlt_spread_20d"),
            ("VEA", "vea", "vea_spy_spread_20d"),
            ("TIP", "tip", "spy_tip_spread_20d"),
            ("BTC", "btc", "btc_spy_spread_20d"),
        ]:
            if f"{label}_ret" in out.columns:
                spread = (ret - out[f"{label}_ret"]).rolling(20, min_periods=10).mean()
                out[feat_name] = spread.shift(1)

        if "gld_ret" in out.columns and "tlt_ret" in out.columns:
            spread = (out["gld_ret"] - out["tlt_ret"]).rolling(20, min_periods=10).mean()
            out["gld_tlt_spread_20d"] = spread.shift(1)

    # --- Momentum features (unique ones not in vol_rich) ---
    out["mom_21d"] = ret.rolling(21, min_periods=10).sum().shift(1)
    out["mom_accel"] = out["mom_21d"] - out["mom_21d"].shift(21)
    out["rev_10d"] = -ret.rolling(10, min_periods=5).sum().shift(1)
    for w, name in [(20, "ret_zscore_20d"), (60, "ret_zscore_60d")]:
        mean = ret.rolling(w, min_periods=w // 2).mean()
        std = ret.rolling(w, min_periods=w // 2).std()
        out[name] = ((ret - mean) / std.replace(0, np.nan)).shift(1)
    for w, name in [(20, "dist_from_mean_20d"), (60, "dist_from_mean_60d")]:
        mean = ret.rolling(w, min_periods=w // 2).mean()
        out[name] = (ret - mean).shift(1)
    for w, name in [(60, "trend_60d")]:
        mean = ret.rolling(w, min_periods=w // 2).mean()
        std = ret.rolling(w, min_periods=w // 2).std()
        out[name] = (mean.abs() / std.replace(0, np.nan)).shift(1)

    for col in _EXTENDED_FEATURE_NAMES:
        if col not in out.columns:
            out[col] = np.nan

    return out
