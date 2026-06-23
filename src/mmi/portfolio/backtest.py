"""Walk-forward portfolio backtest — strictly point-in-time (no look-ahead).

Operates on a wide daily-returns panel (index = date, columns = symbols) so it is pure and
unit-testable independent of the data layer. At each rebalance date, target weights are computed
from a trailing covariance window using ONLY returns strictly *before* that date; between
rebalances the weights drift with realised returns; turnover incurs a transaction cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.portfolio import engine

STRATEGIES = ("equal_weight", "inverse_vol", "risk_parity", "mvo_histmean")
# A fixed-weight benchmark (e.g. 60/40) is NOT a solver strategy: it is run through this same
# engine — same rebalance cadence, drift, turnover cost, return clipping and point-in-time warmup —
# so its track record is a like-for-like comparison rather than a flattering, zero-cost SQL series.
FIXED_WEIGHT = "fixed_weight"
# The ML max-Sharpe strategy: max-Sharpe on a caller-supplied, point-in-time blended mu (built by
# compute.build_ml_mu_panel) + the same Ledoit-Wolf cov as mvo_histmean. Not in STRATEGIES — it is
# orchestrated by compute (it needs the forecast), like the benchmark.
MVO_ML = "mvo_ml"


def _solve(
    strategy: str,
    window: pd.DataFrame,
    fixed_weights: np.ndarray | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    n = window.shape[1]
    if strategy == FIXED_WEIGHT:
        assert fixed_weights is not None  # caller-supplied target, validated in run_backtest
        return np.asarray(fixed_weights, dtype=float)
    if strategy == MVO_ML:
        assert mu is not None  # caller-supplied blended mu, validated in run_backtest_full
        return engine.max_sharpe(engine.ledoit_wolf_cov(window.to_numpy()), np.asarray(mu, float))
    if strategy == "equal_weight" or n == 1:
        return engine.equal_weight(n)  # a single asset is trivially 100% weight
    cov = np.atleast_2d(np.cov(window.to_numpy(), rowvar=False))
    if strategy == "inverse_vol":
        return engine.inverse_volatility(cov)
    if strategy == "risk_parity":
        return engine.risk_parity(cov)
    if strategy == "mvo_histmean":
        # Max-Sharpe with the trailing-window mean as expected returns + Ledoit-Wolf shrunk cov.
        # The upcoming mvo_ml uses the SAME shrunk cov, so its comparison isolates the ML mu.
        arr = window.to_numpy()
        return engine.max_sharpe(engine.ledoit_wolf_cov(arr), arr.mean(axis=0))
    raise ValueError(f"unknown strategy: {strategy} (expected one of {STRATEGIES})")


def rebalance_dates(index: pd.DatetimeIndex, freq: str, warmup: int) -> list:
    """Last trading day of each month (``M``) or quarter (``Q``), after ``warmup`` observations."""
    if len(index) <= warmup:
        return []
    eligible = pd.Series(index[warmup:], index=index[warmup:])
    if freq == "M":
        keys = [eligible.index.year, eligible.index.month]
    elif freq == "Q":
        keys = [eligible.index.year, eligible.index.quarter]
    else:
        raise ValueError(f"unknown rebalance freq: {freq} (expected 'M' or 'Q')")
    return list(eligible.groupby(keys).last())


def run_backtest_full(
    returns: pd.DataFrame,
    *,
    strategy: str,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
    fixed_weights: np.ndarray | None = None,
    mu_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest ``strategy`` and also return per-asset daily return contributions.

    Returns ``(returns, contributions)``:
    - ``returns`` — indexed by date with ``daily_return`` (net of costs) and ``cumulative_return``.
    - ``contributions`` — indexed by the *invested* dates, one column per symbol holding that day's
      gross contribution ``w_{t-1} * r_t``, plus a ``__cost__`` column (the negative cost drag).
      Per-asset gross contributions sum to the gross daily return; gross plus ``__cost__`` is the
      net ``daily_return``. Used for performance attribution.

    ``cost`` is a **round-trip** transaction cost; a rebalance pays ``cost * 0.5 * turnover`` where
    ``turnover = sum |w_target - w_drifted|`` (so one-way trades, including the initial buy from
    cash, cost ``cost / 2`` per unit). Per-asset daily returns are clipped at -100% (a long
    position cannot lose more than its capital).

    Known simplifications (fine for this showcase): the cost is a return drag and is not removed
    from the drifting wealth base; and the most recent partial month/quarter rebalances on its last
    available day, so the final reported point is provisional until that period completes.
    """
    if strategy not in STRATEGIES and strategy not in (FIXED_WEIGHT, MVO_ML):
        raise ValueError(f"unknown strategy: {strategy} (expected one of {STRATEGIES})")
    panel = returns.dropna(how="any").sort_index()
    symbols = list(panel.columns)
    if strategy == FIXED_WEIGHT:
        if fixed_weights is None or len(fixed_weights) != len(symbols):
            raise ValueError("fixed_weight requires fixed_weights aligned to the panel columns")
        if not np.isfinite(fixed_weights).all() or not np.isclose(np.sum(fixed_weights), 1.0):
            raise ValueError("fixed_weights must be finite and sum to 1")
    mu_wide: pd.DataFrame | None = None
    if strategy == MVO_ML:
        if mu_panel is None or mu_panel.empty:
            raise ValueError("mvo_ml requires a mu_panel of (date, symbol, mu)")
        mu_wide = mu_panel.pivot_table(index="date", columns="symbol", values="mu").reindex(
            columns=symbols
        )
    rebals = set(rebalance_dates(panel.index, freq, lookback))

    weights: pd.Series | None = None
    records: list[tuple] = []
    contributions: list[dict] = []
    for date, row in panel.iterrows():
        ret = row.clip(lower=-1.0)  # a long position can lose at most 100% (guards bad ticks)
        cost_today = 0.0
        if date in rebals:
            window = panel.loc[:date].iloc[:-1].tail(lookback)  # strictly BEFORE `date`
            if len(window) >= lookback:
                if strategy == MVO_ML:
                    assert mu_wide is not None  # validated above when strategy == MVO_ML
                    mu_vec = (
                        mu_wide.loc[date].to_numpy()
                        if date in mu_wide.index
                        else window.to_numpy().mean(axis=0)  # fallback to the prior
                    )
                    target = pd.Series(_solve(strategy, window, mu=mu_vec), index=symbols)
                else:
                    target = pd.Series(_solve(strategy, window, fixed_weights), index=symbols)
                if not np.isfinite(target.to_numpy()).all():
                    raise ValueError(f"non-finite weights from {strategy} at {date}")
                prior = weights if weights is not None else pd.Series(0.0, index=symbols)
                turnover = float((target - prior).abs().sum())
                cost_today = cost * 0.5 * turnover
                weights = target
        if weights is None:
            records.append((date, 0.0))  # pre-warmup: uninvested
            continue
        gross = weights * ret  # per-asset gross contribution to the day's return
        records.append((date, float(gross.sum()) - cost_today))
        contributions.append({"date": date, "__cost__": -cost_today, **gross.to_dict()})
        drifted = weights * (1.0 + ret)  # let weights float into the next day
        total = float(drifted.sum())
        weights = (
            drifted / total
            if total > 0
            else pd.Series(engine.equal_weight(len(symbols)), index=symbols)
        )

    out = pd.DataFrame(records, columns=["date", "daily_return"]).set_index("date")
    out["cumulative_return"] = (1.0 + out["daily_return"]).cumprod() - 1.0
    contrib = (
        pd.DataFrame(contributions).set_index("date")
        if contributions
        else pd.DataFrame(columns=["__cost__", *symbols])
    )
    return out, contrib


def run_backtest(
    returns: pd.DataFrame,
    *,
    strategy: str,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
    fixed_weights: np.ndarray | None = None,
    mu_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Backtest ``strategy``; return the per-day returns frame (see :func:`run_backtest_full`)."""
    return run_backtest_full(
        returns,
        strategy=strategy,
        lookback=lookback,
        freq=freq,
        cost=cost,
        fixed_weights=fixed_weights,
        mu_panel=mu_panel,
    )[0]
