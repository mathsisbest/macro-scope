"""Block-bootstrap confidence intervals on portfolio performance — honest uncertainty.

A stationary (Politis-Romano) bootstrap resamples *blocks* of the daily-return series so serial
dependence is preserved (an iid bootstrap would understate the variance of a Sharpe ratio). We
resample the SAME dates across every strategy (paired), so a Sharpe *difference* between two
strategies accounts for their cross-correlation. With only a few years of data and a handful of
strategies, Sharpe gaps are usually NOT statistically distinguishable — this quantifies that
instead of letting a narrative imply skill that isn't there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe (rf=0); 0.0 if the series has no variance."""
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(returns) / sd * np.sqrt(TRADING_DAYS))


def stationary_bootstrap_indices(
    n: int, n_boot: int, avg_block: float, rng: np.random.Generator
) -> np.ndarray:
    """``n_boot x n`` resample index matrix; geometric block lengths (mean ``avg_block``), wrapping.

    Vectorised over bootstrap replicates: with prob ``1/avg_block`` a new block starts at a fresh
    random position, otherwise the previous index advances by one (mod ``n``).
    """
    p = 1.0 / avg_block
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    cur = starts[:, 0].copy()
    idx[:, 0] = cur
    for t in range(1, n):
        cur = np.where(new_block[:, t], starts[:, t], (cur + 1) % n)
        idx[:, t] = cur
    return idx


def _bootstrap_sharpe(returns: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Annualised Sharpe for each bootstrap resample (``idx`` is ``n_boot x n``)."""
    samples = returns[idx]
    sd = samples.std(axis=1, ddof=1)
    mean = samples.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd == 0.0, 0.0, mean / sd * np.sqrt(TRADING_DAYS))
    return out


def _invested(wide: pd.DataFrame) -> pd.DataFrame:
    """Drop the leading warm-up rows where every strategy is still in cash (all-zero returns)."""
    invested = (wide != 0).any(axis=1).cummax()
    return wide.loc[invested]


def bootstrap_strategy_stats(
    returns_long: pd.DataFrame,
    *,
    n_boot: int = 2000,
    ci: float = 0.90,
    avg_block: int = 21,
    seed: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap Sharpe CIs per strategy + pairwise Sharpe-difference CIs.

    ``returns_long``: ``[strategy, date, daily_return, ...]``. Returns ``(per_strategy, pairs)``:
    - per_strategy: ``strategy, sharpe, sharpe_lo, sharpe_hi, n_obs, n_boot, ci_pct, block_days``
    - pairs: ``strategy_a, strategy_b, sharpe_a, sharpe_b, sharpe_diff, diff_lo, diff_hi,
      distinguishable`` (``distinguishable`` = the difference CI excludes zero).
    """
    wide = returns_long.pivot_table(
        index="date", columns="strategy", values="daily_return"
    ).sort_index()
    wide = _invested(wide.dropna(how="any"))  # common, invested dates -> paired resampling
    strategies = list(wide.columns)
    n = len(wide)
    if n < 2:
        raise ValueError(f"need >= 2 invested observations for a bootstrap, got {n}")

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, n_boot, avg_block, rng)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    arrs = {s: wide[s].to_numpy(dtype=float) for s in strategies}
    point = {s: sharpe(arrs[s]) for s in strategies}
    boot = {s: _bootstrap_sharpe(arrs[s], idx) for s in strategies}  # same idx => paired

    per_strategy = pd.DataFrame(
        [
            {
                "strategy": s,
                "sharpe": point[s],
                "sharpe_lo": float(np.percentile(boot[s], lo_q)),
                "sharpe_hi": float(np.percentile(boot[s], hi_q)),
                "n_obs": n,
                "n_boot": n_boot,
                "ci_pct": ci,
                "block_days": avg_block,
            }
            for s in strategies
        ]
    )

    pairs = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a, b = strategies[i], strategies[j]
            diff = boot[a] - boot[b]  # paired difference (same resampled dates)
            lo, hi = float(np.percentile(diff, lo_q)), float(np.percentile(diff, hi_q))
            pairs.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "sharpe_a": point[a],
                    "sharpe_b": point[b],
                    "sharpe_diff": point[a] - point[b],
                    "diff_lo": lo,
                    "diff_hi": hi,
                    "distinguishable": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return per_strategy, pd.DataFrame(pairs)
