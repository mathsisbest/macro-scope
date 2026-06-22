"""Compute portfolio backtests from marts.fct_asset_daily and land them for dbt to model.

Pure helpers (operate on DataFrames); the ``mmi portfolio`` CLI wires them to DuckDB
(raw.portfolio_returns), which dbt then declares as a source and builds tested marts on top of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.portfolio.backtest import FIXED_WEIGHT, STRATEGIES, run_backtest_full

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


def _strategy_runs(
    panel: pd.DataFrame, *, strategies: tuple, lookback: int, freq: str, cost: float
):
    """Yield ``(label, returns, contributions)`` for each strategy plus the 60/40 benchmark.

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
    for label, out, _ in _strategy_runs(
        panel, strategies=strategies, lookback=lookback, freq=freq, cost=cost
    ):
        result = out.reset_index()
        result.insert(0, "strategy", label)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def compute_attribution(
    asset_daily: pd.DataFrame,
    *,
    strategies: tuple = STRATEGIES,
    lookback: int = 252,
    freq: str = "M",
    cost: float = 0.001,
) -> pd.DataFrame:
    """Per-(strategy, symbol) return + risk attribution, from the same backtest runs.

    Columns: ``[strategy, symbol, contribution_to_return, contribution_to_risk,
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
        panel, strategies=strategies, lookback=lookback, freq=freq, cost=cost
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
    return pd.DataFrame(rows)
