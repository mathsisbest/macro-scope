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

STRATEGIES = ("equal_weight", "inverse_vol", "risk_parity")


def _solve(strategy: str, window: pd.DataFrame) -> np.ndarray:
    n = window.shape[1]
    if strategy == "equal_weight" or n == 1:
        return engine.equal_weight(n)  # a single asset is trivially 100% weight
    cov = np.atleast_2d(np.cov(window.to_numpy(), rowvar=False))
    if strategy == "inverse_vol":
        return engine.inverse_volatility(cov)
    if strategy == "risk_parity":
        return engine.risk_parity(cov)
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


def run_backtest(
    returns: pd.DataFrame,
    *,
    strategy: str,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
) -> pd.DataFrame:
    """Backtest ``strategy`` over a daily-returns panel.

    Returns a frame indexed by date with ``daily_return`` (net of costs) and ``cumulative_return``.
    ``cost`` is a **round-trip** transaction cost; a rebalance pays ``cost * 0.5 * turnover`` where
    ``turnover = sum |w_target - w_drifted|`` (so one-way trades, including the initial buy from
    cash, cost ``cost / 2`` per unit). Per-asset daily returns are clipped at -100% (a long
    position cannot lose more than its capital).

    Known simplifications (fine for this showcase): the cost is a return drag and is not removed
    from the drifting wealth base; and the most recent partial month/quarter rebalances on its last
    available day, so the final reported point is provisional until that period completes.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy} (expected one of {STRATEGIES})")
    panel = returns.dropna(how="any").sort_index()
    symbols = list(panel.columns)
    rebals = set(rebalance_dates(panel.index, freq, lookback))

    weights: pd.Series | None = None
    records: list[tuple] = []
    for date, row in panel.iterrows():
        ret = row.clip(lower=-1.0)  # a long position can lose at most 100% (guards bad ticks)
        cost_today = 0.0
        if date in rebals:
            window = panel.loc[:date].iloc[:-1].tail(lookback)  # strictly BEFORE `date`
            if len(window) >= lookback:
                target = pd.Series(_solve(strategy, window), index=symbols)
                if not np.isfinite(target.to_numpy()).all():
                    raise ValueError(f"non-finite weights from {strategy} at {date}")
                prior = weights if weights is not None else pd.Series(0.0, index=symbols)
                turnover = float((target - prior).abs().sum())
                cost_today = cost * 0.5 * turnover
                weights = target
        if weights is None:
            records.append((date, 0.0))  # pre-warmup: uninvested
            continue
        records.append((date, float((weights * ret).sum()) - cost_today))
        drifted = weights * (1.0 + ret)  # let weights float into the next day
        total = float(drifted.sum())
        weights = (
            drifted / total
            if total > 0
            else pd.Series(engine.equal_weight(len(symbols)), index=symbols)
        )

    out = pd.DataFrame(records, columns=["date", "daily_return"]).set_index("date")
    out["cumulative_return"] = (1.0 + out["daily_return"]).cumprod() - 1.0
    return out
