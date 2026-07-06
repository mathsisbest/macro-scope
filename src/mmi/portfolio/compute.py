"""Portfolio-level return forecasts and signal generation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..ml import features as feat
from ..ml.forecast import evaluate_forecast
from ..utils.logging import get_logger

log = get_logger("portfolio.compute")


def compute_all_predictions(
    db,
    universe: Sequence[str] | None = None,
    train_size: int = 160,
    test_size: int = 300,
    model: str = "gb",
    feature_set: str = "default",
    target_horizon: int = 252,
    target_horizon: int = 63,
    target_type: str = "raw",
    ensemble_method: str = "mean",
    loss: str = "squared_error",
    **model_kwargs,
) -> pd.DataFrame:
    """Compute out-of-sample predictions for each ticker using rolling window.

    Uses the best-performing config: GB/default/63d target/rolling window/train=250.

    Returns pd.DataFrame with columns ``date, symbol, pred_ret, ue, pos_signal, pred_vol``.
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
            target_horizon=target_horizon,
            ensemble_method=ensemble_method,
            use_all_train=False,  # Rolling window
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


def compute_ml_mu_panel(
    asset_daily: pd.DataFrame,
    *,
    window: str = "",
    asset_daily_full: pd.DataFrame | None = None,
    train_size: int = 160,
    test_size: int = 300,
    target_horizon: int = 252,
    model: str = "gb",
    feature_set: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ML forecast panel for the portfolio backtest.

    For each asset and rebalance date, produces a point-in-time forecast of the
    forward return. The portfolio gate (skill → λ → blend with historical mean)
    is applied downstream.

    Returns ``(mu_panel [date, symbol, mu], gate [date, forecast_skill, forecast_weight])``.
    """
    symbols = list(asset_daily["symbol"].unique()) if "symbol" in asset_daily.columns else ["SPY"]
    mu_rows: list[dict] = []
    gate_rows: list[dict] = []

    for sym in symbols:
        sym_data = asset_daily[asset_daily["symbol"] == sym].copy()
        if sym_data.empty or len(sym_data) < train_size + test_size:
            continue

        # Get OHLC from full data if available
        ohlc_data = sym_data
        if asset_daily_full is not None and sym in asset_daily_full.get("symbol", pd.Series()).values:
            ohlc_full = asset_daily_full[asset_daily_full["symbol"] == sym]
            if not ohlc_full.empty:
                ohlc_data = ohlc_full

        # Check if we have the required columns
        required = {"date", "daily_return"}
        if feature_set in ("vol", "vol_macro", "vol_rich", "vol_medium"):
            required |= {"open", "high", "low", "close"}
        if not required.issubset(ohlc_data.columns):
            log.warning("skip %s: missing columns %s", sym, required - set(ohlc_data.columns))
            continue

        try:
            res = evaluate_forecast(
                df=ohlc_data,
                train_size=train_size,
                test_size=test_size,
                horizon=None,
                model=model,
                feature_set=feature_set,
                target_type="raw",
                target_horizon=target_horizon,
                use_all_train=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s forecast: %s", sym, exc)
            continue

        if res.get("prediction_count", 0) == 0:
            continue

        dates = res.get("dates", pd.Series(dtype="object"))
        preds = res.get("predictions", pd.Series(dtype=float))

        if isinstance(dates, pd.Series) and isinstance(preds, pd.Series):
            for d, p in zip(dates.values, preds.values):
                if pd.notna(p):
                    mu_rows.append({
                        "date": pd.Timestamp(d),
                        "symbol": sym,
                        "mu": float(p) / target_horizon,  # Daily-equivalent
                    })

    mu_panel = pd.DataFrame(mu_rows) if mu_rows else pd.DataFrame(columns=["date", "symbol", "mu"])
    gate = pd.DataFrame(columns=["date", "forecast_skill", "forecast_weight"])

    log.info(
        "ml_mu_panel: %d mu rows, %d assets, target_horizon=%d",
        len(mu_panel), mu_panel["symbol"].nunique() if not mu_panel.empty else 0, target_horizon,
    )
    return mu_panel, gate


def btc_aligned_returns(asset_daily: pd.DataFrame, *, btc_symbol: str = "BTC") -> pd.DataFrame:
    """BTC daily returns recomputed on the equity trading calendar."""
    btc = asset_daily[asset_daily["symbol"] == btc_symbol]
    if btc.empty:
        return pd.DataFrame(columns=["date", "daily_return"])
    equity_dates = pd.DatetimeIndex(
        sorted(asset_daily.loc[asset_daily["asset_class"] != "crypto", "date"].unique())
    )
    btc_returns = btc.set_index("date")["daily_return"].sort_index()
    wealth = (1.0 + btc_returns.fillna(0.0)).cumprod()
    on_equity = wealth.reindex(equity_dates).ffill()
    aligned = on_equity.pct_change()
    return pd.DataFrame({"date": equity_dates, "daily_return": aligned.to_numpy()})


def window_asset_daily(
    asset_daily: pd.DataFrame,
    window_id: str,
    *,
    btc_floor: pd.Timestamp | None = None,
    btc_aligned: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Filter asset_daily to the specified window's universe."""
    from mmi.portfolio import windows

    non_crypto = asset_daily[
        (asset_daily["asset_class"] != "crypto")
        & (asset_daily["symbol"].isin(windows.PORTFOLIO_UNIVERSE))
    ]
    if window_id == windows.EX_BTC_2002:
        if non_crypto.empty:
            return non_crypto.copy()
        common_start = non_crypto.groupby("symbol")["date"].min().max()
        return non_crypto[non_crypto["date"] >= common_start].copy()
    if btc_floor is None:
        return non_crypto.copy()
    non_crypto = non_crypto[non_crypto["date"] >= btc_floor]
    if window_id == windows.EX_BTC_2015:
        return non_crypto.copy()
    if window_id == windows.INC_BTC_2015:
        if btc_aligned is None:
            return non_crypto.copy()
        btc = btc_aligned[btc_aligned["date"] >= btc_floor].dropna(subset=["daily_return"]).copy()
        btc["symbol"] = "BTC"
        btc["asset_class"] = "crypto"
        return pd.concat([non_crypto, btc], ignore_index=True)
    return non_crypto.copy()


def compute_portfolio_returns(
    asset_daily: pd.DataFrame,
    *,
    ml_mu_panel: pd.DataFrame | None = None,
    window: str = "",
    asset_daily_full: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute portfolio returns: equal-weight + ML-tilted + regime-aware ML."""
    panel = asset_daily.pivot_table(index="date", columns="symbol", values="daily_return")
    panel = panel.sort_index().dropna(how="all")

    frames = []

    # 1. Equal-weight baseline
    ew_ret = panel.mean(axis=1)
    result = pd.DataFrame({
        "window_id": window,
        "strategy": "equal_weight",
        "date": panel.index,
        "daily_return": ew_ret.values,
    })
    result["cumulative_return"] = (1 + result["daily_return"]).cumprod() - 1
    frames.append(result)

    if ml_mu_panel is None or ml_mu_panel.empty:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    ml_pivot = ml_mu_panel.pivot_table(index="date", columns="symbol", values="mu")
    common_dates = panel.index.intersection(ml_pivot.index)

    if len(common_dates) == 0:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # 2. ML-tilted strategy (uniform weight on positive signals)
    ml_tilt = panel.loc[common_dates].copy()
    for date in common_dates:
        if date in ml_pivot.index:
            signals = ml_pivot.loc[date]
            pos_signals = signals[signals > 0]
            if len(pos_signals) > 0:
                weights = pos_signals / pos_signals.sum()
                ml_tilt.loc[date] = panel.loc[date] * weights.reindex(panel.columns, fill_value=0)
            else:
                ml_tilt.loc[date] = panel.loc[date] / len(panel.columns)

    ml_ret = ml_tilt.sum(axis=1)
    result_ml = pd.DataFrame({
        "window_id": window,
        "strategy": "ml_tilt",
        "date": common_dates,
        "daily_return": ml_ret.values,
    })
    result_ml["cumulative_return"] = (1 + result_ml["daily_return"]).cumprod() - 1
    frames.append(result_ml)

    # 3. Regime-aware ML: size up during negative momentum, size down during positive
    # Momentum regime: 63d rolling return of the equal-weight portfolio
    ew_series = panel.loc[common_dates].mean(axis=1)
    mom_63d = ew_series.rolling(63, min_periods=20).mean()

    ml_regime = panel.loc[common_dates].copy()
    for date in common_dates:
        if date in ml_pivot.index and date in mom_63d.index:
            signals = ml_pivot.loc[date]
            mom = mom_63d.loc[date]

            if pd.isna(mom):
                # No regime data yet — use equal weight
                ml_regime.loc[date] = panel.loc[date] / len(panel.columns)
                continue

            # Regime multiplier: 2x during negative momentum, 0.5x during positive
            if mom < 0:
                regime_mult = 2.0  # Size up when model is more accurate
            else:
                regime_mult = 0.5  # Size down when model is less accurate

            pos_signals = signals[signals > 0]
            if len(pos_signals) > 0:
                # Apply regime multiplier to position sizing
                raw_weights = pos_signals / pos_signals.sum()
                # Scale up: increase concentration during negative momentum
                adjusted_weights = raw_weights * regime_mult
                # Normalize to sum to 1 (cap at 3x any single position)
                adjusted_weights = adjusted_weights.clip(upper=1.0 / len(pos_signals) * 3)
                adjusted_weights = adjusted_weights / adjusted_weights.sum()
                ml_regime.loc[date] = panel.loc[date] * adjusted_weights.reindex(panel.columns, fill_value=0)
            else:
                ml_regime.loc[date] = panel.loc[date] / len(panel.columns)

    regime_ret = ml_regime.sum(axis=1)
    result_regime = pd.DataFrame({
        "window_id": window,
        "strategy": "ml_regime",
        "date": common_dates,
        "daily_return": regime_ret.values,
    })
    result_regime["cumulative_return"] = (1 + result_regime["daily_return"]).cumprod() - 1
    frames.append(result_regime)

    return pd.concat(frames, ignore_index=True)


def compute_attribution(
    asset_daily: pd.DataFrame,
    *,
    ml_mu_panel: pd.DataFrame | None = None,
    window: str = "",
    asset_daily_full: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-asset contribution to portfolio return."""
    panel = asset_daily.pivot_table(index="date", columns="symbol", values="daily_return")
    panel = panel.sort_index().dropna(how="all")

    # Equal-weight attribution
    n = len(panel.columns)
    weight = 1.0 / n
    rows = []
    for sym in panel.columns:
        contribution = (panel[sym] * weight).sum()
        rows.append({
            "window_id": window,
            "strategy": "equal_weight",
            "symbol": sym,
            "contribution_to_return": float(contribution),
            "contribution_to_risk": 1.0 / n,
        })
    return pd.DataFrame(rows)
