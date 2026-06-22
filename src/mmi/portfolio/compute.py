"""Compute portfolio backtests from marts.fct_asset_daily and land them for dbt to model.

Pure helpers (operate on DataFrames); the ``mmi portfolio`` CLI wires them to DuckDB
(raw.portfolio_returns), which dbt then declares as a source and builds tested marts on top of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.portfolio.backtest import FIXED_WEIGHT, STRATEGIES, run_backtest

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


def _sixty_forty_weights(symbols: list) -> np.ndarray | None:
    """0.6 on the equity anchor, 0.4 on the first available bond, 0 elsewhere (None if absent)."""
    bond = next((b for b in _BENCHMARK_BONDS if b in symbols), None)
    if _BENCHMARK_EQUITY not in symbols or bond is None:
        return None
    weights = pd.Series(0.0, index=symbols)
    weights[_BENCHMARK_EQUITY] = 0.6
    weights[bond] = 0.4
    return weights.to_numpy()


def compute_portfolio_returns(
    asset_daily: pd.DataFrame,
    *,
    strategies: tuple = STRATEGIES,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
) -> pd.DataFrame:
    """Backtest each strategy plus the 60/40 benchmark; return a long frame.

    Columns: ``[strategy, date, daily_return, cumulative_return]``. The ``sixty_forty`` benchmark is
    appended whenever the equity + a bond leg are present in the universe (same engine, same panel).
    """
    panel = build_returns_panel(asset_daily)
    frames = []
    for strategy in strategies:
        result = run_backtest(panel, strategy=strategy, lookback=lookback, freq=freq, cost=cost)
        result = result.reset_index()
        result.insert(0, "strategy", strategy)
        frames.append(result)

    bench_weights = _sixty_forty_weights(list(panel.columns))
    if bench_weights is not None:
        result = run_backtest(
            panel,
            strategy=FIXED_WEIGHT,
            lookback=lookback,
            freq=freq,
            cost=cost,
            fixed_weights=bench_weights,
        ).reset_index()
        result.insert(0, "strategy", BENCHMARK)
        frames.append(result)

    return pd.concat(frames, ignore_index=True)
