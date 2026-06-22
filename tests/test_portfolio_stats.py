"""Block-bootstrap stats: correct Sharpe, sound resampling, honest distinguishability."""

import numpy as np
import pandas as pd

from mmi.portfolio.stats import (
    bootstrap_strategy_stats,
    sharpe,
    stationary_bootstrap_indices,
)

TRADING_DAYS = 252


def _long(specs: dict, n: int = 400, seed: int = 0, warmup: int = 0) -> pd.DataFrame:
    """Long [strategy, date, daily_return]; `specs` maps strategy -> (mean, std)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n + warmup)
    rows = []
    for strat, (mu, sd) in specs.items():
        rets = np.concatenate([np.zeros(warmup), rng.normal(mu, sd, n)])  # leading cash warmup
        for d, x in zip(idx, rets, strict=True):
            rows.append({"strategy": strat, "date": d, "daily_return": float(x)})
    return pd.DataFrame(rows)


def test_sharpe_matches_definition_and_handles_zero_variance():
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.003])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    assert np.isclose(sharpe(r), expected)
    assert sharpe(np.zeros(10)) == 0.0  # no variance -> 0, not NaN


def test_bootstrap_indices_in_range_and_reproducible():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = stationary_bootstrap_indices(100, 500, 21, rng1)
    b = stationary_bootstrap_indices(100, 500, 21, rng2)
    assert a.shape == (500, 100)
    assert a.min() >= 0 and a.max() < 100
    assert np.array_equal(a, b)  # same seed -> identical resamples


def test_stats_structure_ci_ordering_and_reproducible():
    df = _long({"a": (0.0004, 0.01), "b": (0.0002, 0.012), "c": (0.0006, 0.02)}, n=400)
    per1, pairs1 = bootstrap_strategy_stats(df, n_boot=1000, seed=1)
    per2, pairs2 = bootstrap_strategy_stats(df, n_boot=1000, seed=1)

    assert list(per1["strategy"]) == ["a", "b", "c"]
    assert (per1["sharpe_lo"] <= per1["sharpe_hi"]).all()
    assert len(pairs1) == 3  # C(3, 2)
    assert (pairs1["diff_lo"] <= pairs1["diff_hi"]).all()
    # reproducible given the seed
    pd.testing.assert_frame_equal(per1, per2)
    pd.testing.assert_frame_equal(pairs1, pairs2)


def test_identical_strategies_are_not_distinguishable():
    # Same realised returns for both -> every paired bootstrap difference is exactly 0.
    base = _long({"a": (0.0005, 0.01)}, n=400, seed=3)
    twin = base.copy()
    twin["strategy"] = "b"
    _, pairs = bootstrap_strategy_stats(pd.concat([base, twin]), n_boot=1000, seed=2)
    row = pairs.iloc[0]
    assert np.isclose(row["sharpe_diff"], 0.0)
    assert not row["distinguishable"]  # a strategy cannot be distinguished from itself


def test_strongly_separated_strategies_are_distinguishable():
    df = _long({"winner": (0.005, 0.004), "loser": (-0.001, 0.01)}, n=400, seed=4)
    _, pairs = bootstrap_strategy_stats(df, n_boot=1000, seed=5)
    row = pairs.iloc[0]
    assert row["distinguishable"]  # a huge Sharpe gap -> difference CI excludes 0
    assert (row["diff_lo"] > 0) == (row["sharpe_diff"] > 0)


def test_warmup_rows_are_trimmed_from_the_sample():
    df = _long({"a": (0.0004, 0.01), "b": (0.0003, 0.012)}, n=300, warmup=50)
    per, _ = bootstrap_strategy_stats(df, n_boot=500, seed=6)
    assert (per["n_obs"] == 300).all()  # the 50 leading all-cash rows are excluded
