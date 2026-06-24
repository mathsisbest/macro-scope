"""Compute portfolio backtests from marts.fct_asset_daily and land them for dbt to model.

Pure helpers (operate on DataFrames); the ``mmi portfolio`` CLI wires them to DuckDB
(raw.portfolio_returns), which dbt then declares as a source and builds tested marts on top of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.ml.forecast_panel import walk_forward_mu
from mmi.portfolio import windows
from mmi.portfolio.backtest import (
    FIXED_WEIGHT,
    MVO_ML,
    STRATEGIES,
    rebalance_dates,
    run_backtest_full,
)
from mmi.utils.logging import get_logger

log = get_logger("portfolio.compute")

# A classic 60/40 benchmark: 60% broad equities, 40% bonds. Run through the SAME backtest engine
# as the solver strategies (same panel, dates, rebalance cadence and costs) so the comparison is
# like-for-like. Skipped if neither leg is in the tracked universe.
BENCHMARK = "sixty_forty"
_BENCHMARK_EQUITY = "SPY"
_BENCHMARK_BONDS = ("TLT", "TIP")  # prefer long Treasuries; fall back to TIPS


def build_returns_panel(asset_daily: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long ``[symbol, date, daily_return]`` frame into a wide date x symbol panel."""
    panel = asset_daily.pivot_table(index="date", columns="symbol", values="daily_return")
    return panel.sort_index().dropna(how="all")


def btc_aligned_returns(asset_daily: pd.DataFrame, *, btc_symbol: str = "BTC") -> pd.DataFrame:
    """BTC daily returns recomputed on the equity (non-crypto) trading calendar.

    BTC trades 7 days a week; equities don't. Pivoting raw BTC returns next to equities would
    inject NaN equity rows on weekends. Instead we compound BTC close-to-close to the previous
    equity trading day (Fri->Mon spans the weekend), so each 5-day-calendar bar carries the true
    close-to-close move. This understates BTC's *standalone daily* vol (a multi-day move shows as
    one bar) — documented, and it does not bias the close-to-close return level. Returns
    ``[date, daily_return]`` on the equity calendar; empty if BTC is absent.
    """
    btc = asset_daily[asset_daily["symbol"] == btc_symbol]
    if btc.empty:
        return pd.DataFrame(columns=["date", "daily_return"])
    equity_dates = pd.DatetimeIndex(
        sorted(asset_daily.loc[asset_daily["asset_class"] != "crypto", "date"].unique())
    )
    # Wealth index over BTC's own (7-day) calendar, then sampled at equity dates: the ratio between
    # consecutive equity dates is the compounded return over any intervening non-trading days.
    btc_returns = btc.set_index("date")["daily_return"].sort_index()
    # Warn if there are interior (non-leading, non-trailing) NaN values before filling.
    # Leading/trailing NaN is expected warm-up; interior NaN suggests a data gap that silently
    # becomes a 0% return — the caller should know.
    _first_valid = btc_returns.first_valid_index()
    _last_valid = btc_returns.last_valid_index()
    if _first_valid is not None and _last_valid is not None:
        _interior = btc_returns.loc[_first_valid:_last_valid]
        _n_interior_nan = int(_interior.isna().sum())
        if _n_interior_nan > 0:
            log.warning(
                "btc_aligned_returns: filling %d interior NaN value(s) with 0.0 "
                "(each becomes a 0%% return — check for data gaps)",
                _n_interior_nan,
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
    """The long ``[symbol, date, daily_return, asset_class]`` frame for one backtest window.

    Built as a SEPARATE filtered frame per window — never one merged panel — because a merged
    panel's ``dropna(how='any')`` (in the ML path) would collapse the ex-BTC 2002+ history to BTC's
    ~2015 start. ``ex_btc_2002`` is the full non-crypto history; the 2015 windows start at
    ``btc_floor``; ``inc_btc_2015`` adds BTC on the equity calendar (``btc_aligned``) so it shares
    an identical date set with ``ex_btc_2015`` (the same-period control).
    """
    non_crypto = asset_daily[asset_daily["asset_class"] != "crypto"]
    if window_id == windows.EX_BTC_2002:
        return non_crypto.copy()
    if btc_floor is None:
        raise ValueError(f"{window_id} requires a BTC inception floor (BTC absent from the data)")
    non_crypto = non_crypto[non_crypto["date"] >= btc_floor]
    if window_id == windows.EX_BTC_2015:
        return non_crypto.copy()
    if window_id == windows.INC_BTC_2015:
        if btc_aligned is None:
            raise ValueError("inc_btc_2015 requires the equity-calendar BTC series")
        btc = btc_aligned[btc_aligned["date"] >= btc_floor].dropna(subset=["daily_return"]).copy()
        btc["symbol"] = "BTC"
        btc["asset_class"] = "crypto"
        return pd.concat([non_crypto, btc], ignore_index=True)
    raise ValueError(f"unknown window {window_id}")


def _sixty_forty_weights(symbols: list) -> np.ndarray | None:
    """0.6 on the equity anchor, 0.4 on the first available bond, 0 elsewhere (None if absent)."""
    bond = next((b for b in _BENCHMARK_BONDS if b in symbols), None)
    if _BENCHMARK_EQUITY not in symbols or bond is None:
        return None
    weights = pd.Series(0.0, index=symbols)
    weights[_BENCHMARK_EQUITY] = 0.6
    weights[bond] = 0.4
    return weights.to_numpy()


_GATE_MIN_OBS = 6  # min scored rebalances before an asset's forecast skill is trusted


def build_ml_mu_panel(
    panel: pd.DataFrame,
    rebals: list,
    *,
    lookback: int,
    horizon: int = 21,
    lambda_max: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend the point-in-time ML forecast toward the historical-mean prior, gated by skill.

    For each rebalance ``t`` and asset, the expected-return input to mvo_ml is
    ``mu = lambda(t)*mu_forecast + (1-lambda(t))*mu_hist`` where ``mu_forecast`` is the C2
    point-in-time forecast (daily-equivalent), ``mu_hist`` the trailing-window daily mean (the
    prior), and ``lambda(t) = lambda_max * s(t)``. ``s(t)`` in [0, 1] is the mean per-asset
    fractional improvement of the forecast over the prior at predicting realised forward returns,
    measured ONLY over rebalances whose outcome was realised strictly before ``t`` (point-in-time —
    no leak). No out-of-sample edge over the prior -> ``s≈0`` -> ``mu≈mu_hist`` ->
    ``mvo_ml≈mvo_histmean``. Missing forecasts fall back to the prior.

    Returns ``(mu_panel [date, symbol, mu], gate [date, skill, lambda])``.
    """
    symbols = list(panel.columns)
    long = (
        panel.rename_axis("date")
        .reset_index()
        .melt(id_vars=["date"], var_name="symbol", value_name="daily_return")
    )
    mu_fc_df, _skill = walk_forward_mu(long, rebals, horizon=horizon)
    mu_fc = (
        mu_fc_df.pivot_table(index="date", columns="symbol", values="mu")
        if not mu_fc_df.empty
        else pd.DataFrame()
    )

    pos = {d: i for i, d in enumerate(panel.index)}
    mu_hist: dict = {}
    realised: dict = {}
    for t in rebals:
        mu_hist[t] = panel.loc[:t].iloc[:-1].tail(lookback).mean()
        i = pos[t]
        fwd = panel.iloc[i + 1 : i + 1 + horizon]  # the horizon days AFTER t
        realised[t] = fwd.mean() if len(fwd) == horizon else pd.Series(np.nan, index=symbols)

    def _forecast(t, sym: str) -> float:
        if not mu_fc.empty and t in mu_fc.index and sym in mu_fc.columns:
            value = mu_fc.loc[t, sym]
            return float(value) if np.isfinite(value) else float("nan")
        return float("nan")

    mu_rows: list[dict] = []
    gate_rows: list[dict] = []
    for t in rebals:
        # only rebalances whose forward window has fully realised before t (point-in-time gate)
        scored = [r for r in rebals if r < t and pos[r] + 1 + horizon <= pos[t]]
        skills: list[float] = []
        for sym in symbols:
            err_fc, err_prior = [], []
            for r in scored:
                actual = realised[r][sym]
                fc = _forecast(r, sym)
                if not (np.isfinite(actual) and np.isfinite(fc)):
                    continue
                err_fc.append(abs(fc - actual))
                err_prior.append(abs(float(mu_hist[r][sym]) - actual))
            if len(err_fc) >= _GATE_MIN_OBS and np.mean(err_prior) > 0:
                skills.append(max(0.0, 1.0 - float(np.mean(err_fc)) / float(np.mean(err_prior))))
        skill = float(np.mean(skills)) if skills else 0.0
        lam = lambda_max * skill
        gate_rows.append({"date": t, "forecast_skill": skill, "forecast_weight": lam})
        for sym in symbols:
            hist = float(mu_hist[t][sym])
            fc = _forecast(t, sym)
            mu = lam * fc + (1.0 - lam) * hist if np.isfinite(fc) else hist
            mu_rows.append({"date": t, "symbol": sym, "mu": mu})
    gate = pd.DataFrame(gate_rows)
    if not gate.empty:
        # Surface the gate so a "mvo_ml ≈ mvo_histmean" result is visibly because lambda≈0 (no
        # forecast edge), not a silent bug. (C4 lands it as a mart for the dashboard/brief.)
        log.info(
            "ml gate: mean forecast_weight=%.4f (max %.4f) over %d rebalances",
            float(gate["forecast_weight"].mean()),
            float(gate["forecast_weight"].max()),
            len(gate),
        )
    return pd.DataFrame(mu_rows), gate


def compute_ml_mu_panel(
    asset_daily: pd.DataFrame,
    *,
    lookback: int = 252,
    freq: str = "M",
    horizon: int = 21,
    lambda_max: float = 0.5,
    window: str = windows.DEFAULT_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the mvo_ml blended-mu panel + gate ONCE, so callers reuse it (and can land the gate).

    Returns ``(mu_panel [date, symbol, mu], gate)`` where gate is
    ``[window_id, date, forecast_skill, forecast_weight]``. The ``window`` is stamped on the landed
    gate frame (the mu_panel is internal and is not landed).
    """
    panel = build_returns_panel(asset_daily).dropna(how="any")
    rebals = rebalance_dates(panel.index, freq, lookback)
    if not rebals:
        empty_mu = pd.DataFrame(columns=["date", "symbol", "mu"])
        empty_gate = pd.DataFrame(
            columns=["window_id", "date", "forecast_skill", "forecast_weight"]
        )
        return empty_mu, empty_gate
    mu_panel, gate = build_ml_mu_panel(
        panel, rebals, lookback=lookback, horizon=horizon, lambda_max=lambda_max
    )
    if not gate.empty:
        gate.insert(0, "window_id", window)
    return mu_panel, gate


def _strategy_runs(
    panel: pd.DataFrame,
    *,
    strategies: tuple,
    lookback: int,
    freq: str,
    cost: float,
    horizon: int,
    lambda_max: float,
    include_ml: bool,
    ml_mu_panel: pd.DataFrame | None = None,
):
    """Yield ``(label, returns, contributions)`` for each strategy, the 60/40 benchmark, and (when
    ``include_ml``) the gated ``mvo_ml``. A precomputed ``ml_mu_panel`` is reused if given (so the
    forecast runs once across returns + attribution).

    Both ``compute_portfolio_returns`` and ``compute_attribution`` iterate this, so the returns and
    their attribution always come from the SAME backtest runs (same panel, dates, costs).
    """
    for strategy in strategies:
        out, contrib = run_backtest_full(
            panel, strategy=strategy, lookback=lookback, freq=freq, cost=cost
        )
        yield strategy, out, contrib

    bench_weights = _sixty_forty_weights(list(panel.columns))
    if bench_weights is not None:
        out, contrib = run_backtest_full(
            panel,
            strategy=FIXED_WEIGHT,
            lookback=lookback,
            freq=freq,
            cost=cost,
            fixed_weights=bench_weights,
        )
        yield BENCHMARK, out, contrib

    if include_ml:
        clean = panel.dropna(how="any")  # the forecast + gate need a complete (no-NaN) panel
        rebals = rebalance_dates(clean.index, freq, lookback)
        if rebals:
            mu_panel = ml_mu_panel
            if mu_panel is None:  # build it here unless a precomputed panel was supplied
                mu_panel, _gate = build_ml_mu_panel(
                    clean, rebals, lookback=lookback, horizon=horizon, lambda_max=lambda_max
                )
            out, contrib = run_backtest_full(
                clean, strategy=MVO_ML, lookback=lookback, freq=freq, cost=cost, mu_panel=mu_panel
            )
            yield MVO_ML, out, contrib


def compute_portfolio_returns(
    asset_daily: pd.DataFrame,
    *,
    strategies: tuple = STRATEGIES,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
    horizon: int = 21,
    lambda_max: float = 0.5,
    include_ml: bool = True,
    ml_mu_panel: pd.DataFrame | None = None,
    window: str = windows.DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Backtest each strategy, the 60/40 benchmark, and (when ``include_ml``) the gated mvo_ml.

    Columns: ``[window_id, strategy, date, daily_return, cumulative_return]``. ``sixty_forty`` is
    appended when its legs are in the universe; ``mvo_ml`` when ``include_ml`` (the ML forecast).
    """
    panel = build_returns_panel(asset_daily)
    frames = []
    for label, out, _ in _strategy_runs(
        panel,
        strategies=strategies,
        lookback=lookback,
        freq=freq,
        cost=cost,
        horizon=horizon,
        lambda_max=lambda_max,
        include_ml=include_ml,
        ml_mu_panel=ml_mu_panel,
    ):
        result = out.reset_index()
        result.insert(0, "strategy", label)
        result.insert(0, "window_id", window)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def compute_attribution(
    asset_daily: pd.DataFrame,
    *,
    strategies: tuple = STRATEGIES,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
    horizon: int = 21,
    lambda_max: float = 0.5,
    include_ml: bool = True,
    ml_mu_panel: pd.DataFrame | None = None,
    window: str = windows.DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Per-(strategy, symbol) return + risk attribution, from the same backtest runs.

    Columns: ``[window_id, strategy, symbol, contribution_to_return, contribution_to_risk,
    strategy_gross_return]``.
    - ``contribution_to_return`` = ``sum_t w_{t-1}*r_t`` for the asset; across assets it sums to the
      strategy's gross period return (``strategy_gross_return``). A ``(costs)`` row carries the
      negative cost drag, so the asset rows + cost row reconcile to the net return.
    - ``contribution_to_risk`` = the asset's share of realised portfolio variance,
      ``cov(asset contribution, gross daily return) / var(gross daily return)``; across assets it
      sums to 1. Assets never held are omitted (they contribute exactly zero to both).
    """
    panel = build_returns_panel(asset_daily)
    rows: list[dict] = []
    for label, _, contrib in _strategy_runs(
        panel,
        strategies=strategies,
        lookback=lookback,
        freq=freq,
        cost=cost,
        horizon=horizon,
        lambda_max=lambda_max,
        include_ml=include_ml,
        ml_mu_panel=ml_mu_panel,
    ):
        if contrib.empty:
            continue
        asset_cols = [c for c in contrib.columns if c != "__cost__"]
        gross_daily = contrib[asset_cols].sum(axis=1)
        gross_return = float(gross_daily.sum())
        variance = float(gross_daily.var(ddof=1))
        for symbol in asset_cols:
            contribution = contrib[symbol]
            to_return = float(contribution.sum())
            # 1e-12 floor: a (degenerate) near-constant gross series would otherwise divide
            # covariance noise into garbage shares. Real daily-return variance is ~1e-4.
            to_risk = float(contribution.cov(gross_daily) / variance) if variance > 1e-12 else 0.0
            if to_return == 0.0 and to_risk == 0.0:
                continue  # never held -> exact zero; omit for a clean attribution
            rows.append(
                {
                    "strategy": label,
                    "symbol": symbol,
                    "contribution_to_return": to_return,
                    "contribution_to_risk": to_risk,
                    "strategy_gross_return": gross_return,
                }
            )
        rows.append(
            {
                "strategy": label,
                "symbol": "(costs)",
                "contribution_to_return": float(contrib["__cost__"].sum()),
                "contribution_to_risk": 0.0,
                "strategy_gross_return": gross_return,
            }
        )
    attribution = pd.DataFrame(rows)
    if not attribution.empty:
        attribution.insert(0, "window_id", window)
    return attribution
