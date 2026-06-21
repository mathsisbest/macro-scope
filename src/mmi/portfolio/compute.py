"""Compute portfolio backtests from marts.fct_asset_daily and land them for dbt to model.

Pure helpers (operate on DataFrames); the ``mmi portfolio`` CLI wires them to DuckDB
(raw.portfolio_returns), which dbt then declares as a source and builds tested marts on top of.
"""

from __future__ import annotations

import pandas as pd

from mmi.portfolio.backtest import STRATEGIES, run_backtest


def build_returns_panel(asset_daily: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long ``[symbol, date, daily_return]`` frame into a wide date x symbol panel."""
    panel = asset_daily.pivot_table(index="date", columns="symbol", values="daily_return")
    return panel.sort_index().dropna(how="all")


def compute_portfolio_returns(
    asset_daily: pd.DataFrame,
    *,
    strategies: tuple = STRATEGIES,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
) -> pd.DataFrame:
    """Backtest each strategy; return long ``[strategy, date, daily_return, cumulative_return]``."""
    panel = build_returns_panel(asset_daily)
    frames = []
    for strategy in strategies:
        result = run_backtest(panel, strategy=strategy, lookback=lookback, freq=freq, cost=cost)
        result = result.reset_index()
        result.insert(0, "strategy", strategy)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)
