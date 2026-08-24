"""Portfolio-level return forecasts and signal generation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

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
            asset_dfs=asset_dfs,
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
            for d, p in zip(dates.values, preds.values, strict=False):
                output_rows.append(
                    {
                        "date": pd.Timestamp(d),
                        "symbol": sym,
                        "pred_ret": float(p) if pd.notna(p) else np.nan,
                    }
                )

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
    train_size: int = 252,
    test_size: int = 252,
    target_horizon: int = 20,
    model: str = "gb",
    feature_set: str = "vol_macro",
    macro_df: pd.DataFrame | None = None,
    asset_dfs: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the ML forecast panel for the portfolio backtest.

    For each asset and rebalance date, produces a point-in-time forecast of the
    forward return. The portfolio gate (skill → λ → blend with historical mean)
    is applied downstream.

    Returns ``(mu_panel [date, symbol, mu], gate [date, forecast_skill, forecast_weight])``.
    """
    symbols = list(asset_daily["symbol"].unique()) if "symbol" in asset_daily.columns else ["SPY"]
    mu_rows: list[dict] = []

    for sym in symbols:
        sym_data = asset_daily[asset_daily["symbol"] == sym].copy()
        if sym_data.empty or len(sym_data) < train_size + test_size:
            continue

        # Get OHLC from full data if available
        ohlc_data = sym_data
        if (
            asset_daily_full is not None
            and sym in asset_daily_full.get("symbol", pd.Series()).to_numpy()
        ):
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
                macro_df=macro_df,
                asset_dfs=asset_dfs,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s forecast: %s", sym, exc)
            continue

        if res.get("prediction_count", 0) == 0:
            continue

        dates = res.get("dates", pd.Series(dtype="object"))
        preds = res.get("predictions", pd.Series(dtype=float))

        if isinstance(dates, pd.Series) and isinstance(preds, pd.Series):
            for d, p in zip(dates.values, preds.values, strict=False):
                if pd.notna(p):
                    mu_rows.append(
                        {
                            "date": pd.Timestamp(d),
                            "symbol": sym,
                            "mu": float(p) / target_horizon,  # Daily-equivalent
                        }
                    )

    mu_panel = pd.DataFrame(mu_rows) if mu_rows else pd.DataFrame(columns=["date", "symbol", "mu"])
    gate = pd.DataFrame(columns=["date", "forecast_skill", "forecast_weight"])

    log.info(
        "ml_mu_panel: %d mu rows, %d assets, target_horizon=%d",
        len(mu_panel),
        mu_panel["symbol"].nunique() if not mu_panel.empty else 0,
        target_horizon,
    )
    return mu_panel, gate


def btc_aligned_returns(asset_daily: pd.DataFrame, *, btc_symbol: str = "BTC") -> pd.DataFrame:
    """BTC daily returns recomputed on the equity trading calendar."""
    from mmi.portfolio import windows

    btc = asset_daily[asset_daily["symbol"] == btc_symbol]
    if btc.empty:
        return pd.DataFrame(columns=["date", "daily_return"])
    equity_dates = pd.DatetimeIndex(
        sorted(
            pd.to_datetime(
                asset_daily.loc[asset_daily["symbol"].isin(windows.PORTFOLIO_UNIVERSE), "date"]
            ).unique()
        )
    )
    btc_returns = btc.set_index("date")["daily_return"].sort_index()
    btc_returns.index = pd.to_datetime(btc_returns.index)
    btc_returns = btc_returns[~btc_returns.index.duplicated(keep="first")]  # deduplicate
    btc_valid = btc_returns.notna()
    if btc_valid.any():
        first = btc_returns.first_valid_index()
        last = btc_returns.last_valid_index()
        interior: int = int(btc_returns.loc[first:last].isna().sum())
        leading: int = int(btc_valid.index.get_loc(first))
        trailing: int = int(len(btc_valid) - 1 - btc_valid.index.get_loc(last))
        if interior > 0:
            log.warning(
                "%s aligned returns: %d interior NaN observation(s) treated as 0%% "
                "(leading/trailing gaps skipped: %d/%d)",
                btc_symbol,
                interior,
                leading,
                trailing,
            )
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
    regime_mult_negative: float = 1.2,
    regime_mult_positive: float = 0.8,
    max_leverage: float = 1.0,
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute portfolio returns: equal-weight + ML-tilted + regime-aware ML.

    Parameters
    ----------
    regime_mult_negative:
        Position size multiplier during negative momentum regime (default 1.2x).
    regime_mult_positive:
        Position size multiplier during positive momentum regime (default 0.8x).
    max_leverage:
        Maximum total portfolio leverage limit (default 1.0 for long-only).
    """
    panel = asset_daily.pivot_table(index="date", columns="symbol", values="daily_return")
    panel = panel.sort_index().dropna(how="all")

    frames = []

    # 1. Equal-weight baseline
    ew_ret = panel.mean(axis=1)
    result = pd.DataFrame(
        {
            "window_id": window,
            "strategy": "equal_weight",
            "date": panel.index,
            "daily_return": ew_ret.to_numpy(),
        }
    )
    result["cumulative_return"] = (1 + result["daily_return"]).cumprod() - 1
    frames.append(result)

    if ml_mu_panel is None or ml_mu_panel.empty:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    ml_pivot = ml_mu_panel.pivot_table(index="date", columns="symbol", values="mu")
    common_dates = panel.index.intersection(ml_pivot.index)

    if len(common_dates) == 0:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # 2. ML-tilted strategy (signal-proportional / predicted-Sharpe-proportional weight)
    # Trailing 21-day realized volatility per asset for predicted Sharpe scaling
    trailing_vol = panel.rolling(21, min_periods=5).std().fillna(0.01)
    trailing_vol = trailing_vol.clip(lower=0.001)

    ml_tilt = panel.loc[common_dates].copy()
    conviction_threshold_mult: float = 0.0  # signals must be positive (> 0)
    max_weight: float = 0.40  # 40% concentration cap per asset

    for date in common_dates:
        if date in ml_pivot.index:
            signals = ml_pivot.loc[date].dropna()
            pos_signals = signals[signals > conviction_threshold_mult]
            if len(pos_signals) > 0:
                vols = trailing_vol.loc[date].reindex(pos_signals.index).fillna(0.01)
                # Predicted Sharpe ratio = mu / sigma
                pred_sharpe = (pos_signals / vols).clip(lower=0.0)
                # Iteratively cap weights at max_weight (40%) and re-normalize un-capped weights
                effective_cap = (
                    min(max_weight, 1.0 / len(pos_signals))
                    if max_weight * len(pos_signals) < 1.0
                    else max_weight
                )
                weights = pred_sharpe / pred_sharpe.sum()
                for _ in range(10):
                    if weights.max() <= effective_cap + 1e-6:
                        break
                    capped_mask = weights >= effective_cap
                    uncapped_mask = ~capped_mask
                    if not uncapped_mask.any():
                        weights = pd.Series(1.0 / len(pos_signals), index=pos_signals.index)
                        break
                    excess_mass = (weights[capped_mask] - effective_cap).sum()
                    weights[capped_mask] = effective_cap
                    uncapped_sum = weights[uncapped_mask].sum()
                    if uncapped_sum > 0:
                        weights[uncapped_mask] += excess_mass * (
                            weights[uncapped_mask] / uncapped_sum
                        )
                    else:
                        weights[uncapped_mask] = excess_mass / uncapped_mask.sum()

                ml_tilt.loc[date] = panel.loc[date] * weights.reindex(panel.columns, fill_value=0)
            else:
                ml_tilt.loc[date] = panel.loc[date] / len(panel.columns)

    ml_ret = ml_tilt.sum(axis=1)
    result_ml = pd.DataFrame(
        {
            "window_id": window,
            "strategy": "ml_tilt",
            "date": common_dates,
            "daily_return": ml_ret.to_numpy(),
        }
    )
    result_ml["cumulative_return"] = (1 + result_ml["daily_return"]).cumprod() - 1
    frames.append(result_ml)

    # 3. Regime-aware ML strategy (configurable regime multipliers + max_leverage constraint)
    # Momentum regime: 63d rolling return of the equal-weight portfolio
    ew_series = panel.loc[common_dates].mean(axis=1)
    mom_63d = ew_series.rolling(63, min_periods=20).sum()

    ml_regime = panel.loc[common_dates].copy()
    if regime_df is not None and "date" in regime_df.columns:
        regime_indexed = regime_df.set_index("date")
    else:
        regime_indexed = None

    for date in common_dates:
        if date in ml_pivot.index:
            signals = ml_pivot.loc[date].dropna()

            if regime_indexed is not None and date in regime_indexed.index:
                r_val = regime_indexed.loc[date, "regime"]
                if r_val == "Low":
                    regime_mult = regime_mult_positive
                elif r_val == "High":
                    regime_mult = regime_mult_negative
                else:
                    regime_mult = 1.0
            else:
                if date not in mom_63d.index:
                    ml_regime.loc[date] = panel.loc[date] / len(panel.columns)
                    continue
                mom = mom_63d.loc[date]
                if pd.isna(mom):
                    # No regime data yet — use equal weight
                    ml_regime.loc[date] = panel.loc[date] / len(panel.columns)
                    continue

                # Configurable regime multiplier
                regime_mult = regime_mult_negative if mom < 0 else regime_mult_positive

            pos_signals = signals[signals > 0]
            if len(pos_signals) > 0:
                # Apply signal-proportional base weighting
                vols = trailing_vol.loc[date].reindex(pos_signals.index).fillna(0.01)
                pred_sharpe = (pos_signals / vols).clip(lower=0.0)
                raw_weights = (
                    pred_sharpe / pred_sharpe.sum()
                    if pred_sharpe.sum() > 0
                    else pd.Series(1.0 / len(pos_signals), index=pos_signals.index)
                )

                # Scale by regime multiplier and enforce max_leverage limit (default 1.0)
                scaled_weights = raw_weights * regime_mult
                total_alloc = scaled_weights.sum()
                if total_alloc > max_leverage:
                    scaled_weights = scaled_weights * (max_leverage / total_alloc)

                ml_regime.loc[date] = panel.loc[date] * scaled_weights.reindex(
                    panel.columns, fill_value=0
                )
            else:
                ml_regime.loc[date] = panel.loc[date] / len(panel.columns)

    regime_ret = ml_regime.sum(axis=1)
    result_regime = pd.DataFrame(
        {
            "window_id": window,
            "strategy": "ml_regime",
            "date": common_dates,
            "daily_return": regime_ret.to_numpy(),
        }
    )
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
        rows.append(
            {
                "window_id": window,
                "strategy": "equal_weight",
                "symbol": sym,
                "contribution_to_return": float(contribution),
                "contribution_to_risk": 1.0 / n,
            }
        )
    return pd.DataFrame(rows)
