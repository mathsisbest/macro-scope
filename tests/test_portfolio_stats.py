"""Block-bootstrap stats: correct Sharpe, sound resampling, honest distinguishability."""

import numpy as np
import pandas as pd

from mmi.portfolio import windows
from mmi.portfolio.stats import (
    annualised_return,
    bootstrap_p_value,
    bootstrap_strategy_return_stats,
    bootstrap_strategy_stats,
    paired_btc_effect,
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
    assert (per1["window_id"] == windows.DEFAULT_WINDOW).all()  # Phase D window dimension stamped
    assert (pairs1["window_id"] == windows.DEFAULT_WINDOW).all()
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


def _window_long(daily_returns: np.ndarray, strategy: str = "equal_weight") -> pd.DataFrame:
    """A single-strategy [strategy, date, daily_return] frame for a given return series."""
    idx = pd.bdate_range("2015-01-01", periods=len(daily_returns))
    return pd.DataFrame(
        {"strategy": strategy, "date": idx, "daily_return": daily_returns.astype(float)}
    )


def test_paired_btc_effect_identical_windows_give_exact_zero_diff():
    # If ex and inc are the SAME series, the PAIRED bootstrap (same resampled dates in both) yields
    # diff == 0 on EVERY replicate -> a degenerate [0, 0] CI. An unpaired pair of independent
    # bootstraps would instead spread around 0. So this exactly pins the pairing.
    rets = np.random.default_rng(0).normal(0.0005, 0.01, 300)
    long = _window_long(rets)
    eff = paired_btc_effect(long, long.copy(), n_boot=500)
    row = eff.iloc[0]
    assert np.isclose(row["sharpe_diff"], 0.0)
    assert row["diff_lo"] == 0.0 and row["diff_hi"] == 0.0  # paired -> identical resamples cancel
    assert not row["distinguishable"]


def test_paired_btc_effect_detects_a_strong_improvement():
    # inc = ex shifted up by a constant -> strictly higher mean, same vol -> higher Sharpe in every
    # resample -> a positive, distinguishable difference.
    base = np.random.default_rng(1).normal(0.0, 0.01, 400)
    ex = _window_long(base)
    inc = _window_long(base + 0.002)
    row = paired_btc_effect(ex, inc, n_boot=1000).iloc[0]
    assert row["sharpe_inc"] > row["sharpe_ex"]
    assert row["sharpe_diff"] > 0 and row["diff_lo"] > 0 and row["distinguishable"]


def test_paired_btc_effect_empty_without_common_dates():
    a = _window_long(np.full(200, 0.01))
    b = a.copy()
    b["date"] = pd.bdate_range("2020-01-01", periods=200)  # disjoint dates
    assert paired_btc_effect(a, b).empty


# --- annualised return: the metric the return-significance reporting bootstraps -------------


def test_annualised_return_matches_definition_and_handles_degenerate_inputs():
    r = np.array([0.01, -0.005, 0.002, 0.0, 0.003])
    expected = np.prod(1.0 + r) ** (TRADING_DAYS / len(r)) - 1.0
    assert np.isclose(annualised_return(r), expected)
    assert annualised_return(np.zeros(10)) == 0.0  # no growth -> 0, not NaN
    assert annualised_return(np.array([])) == 0.0  # nothing to compound -> 0
    assert annualised_return(np.array([-1.0, 0.0])) == -1.0  # total loss -> -1, not NaN


def test_bootstrap_p_value_boundaries():
    assert bootstrap_p_value(np.array([1.0, 2.0, 3.0])) == 0.0  # nothing <= 0 -> p = 0 (strongest)
    assert bootstrap_p_value(np.array([-1.0, -2.0, -3.0])) == 0.0  # nothing >= 0 -> p = 0
    assert bootstrap_p_value(np.zeros(10)) == 1.0  # all-zero diffs -> no evidence, not NaN
    assert bootstrap_p_value(np.array([])) == 1.0
    # half positive / half negative -> p = 1.0; one-sided tail -> p = 2 * share of that tail
    assert bootstrap_p_value(np.array([-1.0, -1.0, 1.0, 1.0])) == 1.0
    assert np.isclose(bootstrap_p_value(np.array([-1.0, 1.0, 1.0, 1.0])), 0.5)
    # consistency with the CI rule: diff CI excludes 0 <=> p < (1 - ci) on the same replicates
    rng = np.random.default_rng(11)
    diff = rng.normal(0.5, 1.0, 2000)
    lo, hi = np.percentile(diff, 5.0), np.percentile(diff, 95.0)
    p = bootstrap_p_value(diff)
    assert (lo > 0.0 or hi < 0.0) == (p < 0.10)


def test_return_stats_structure_reproducible_and_ci_ordered():
    df = _long({"a": (0.0004, 0.01), "b": (0.0002, 0.012), "c": (0.0006, 0.02)}, n=400)
    per1, pairs1 = bootstrap_strategy_return_stats(df, n_boot=1000, seed=1)
    per2, pairs2 = bootstrap_strategy_return_stats(df, n_boot=1000, seed=1)

    assert list(per1["strategy"]) == ["a", "b", "c"]
    assert (per1["window_id"] == windows.DEFAULT_WINDOW).all()
    assert (per1["ann_return_lo"] <= per1["ann_return_hi"]).all()
    assert len(pairs1) == 3  # C(3, 2)
    assert (pairs1["diff_lo"] <= pairs1["diff_hi"]).all()
    # per-strategy point estimate is exactly the annualised return of the raw series
    raw_a = _long({"a": (0.0004, 0.01), "b": (0.0002, 0.012), "c": (0.0006, 0.02)}, n=400)
    raw_a = raw_a.loc[raw_a["strategy"] == "a", "daily_return"]
    assert np.isclose(
        per1.loc[per1["strategy"] == "a", "ann_return"].iloc[0],
        annualised_return(raw_a.to_numpy()),
    )
    # reproducible given the seed
    pd.testing.assert_frame_equal(per1, per2)
    pd.testing.assert_frame_equal(pairs1, pairs2)
    # p-values are floored at 1/n_boot, never an overprecise 0.0
    assert (pairs1["p_value"] >= 1 / 1000).all()


def test_identical_strategies_have_no_return_significance():
    base = _long({"a": (0.0005, 0.01)}, n=400, seed=3)
    twin = base.copy()
    twin["strategy"] = "b"
    _, pairs = bootstrap_strategy_return_stats(pd.concat([base, twin]), n_boot=1000, seed=2)
    row = pairs.iloc[0]
    assert np.isclose(row["ann_return_diff"], 0.0)
    assert row["p_value"] == 1.0
    assert not row["distinguishable"]  # a strategy cannot be distinguished from itself


def test_strongly_separated_returns_are_distinguishable():
    df = _long({"winner": (0.005, 0.004), "loser": (-0.001, 0.01)}, n=400, seed=4)
    _, pairs = bootstrap_strategy_return_stats(df, n_boot=1000, seed=5)
    row = pairs.iloc[0]
    assert row["distinguishable"]  # a huge return gap -> difference CI excludes 0
    assert row["p_value"] < 0.10
    assert (row["diff_lo"] > 0) == (row["ann_return_diff"] > 0)


def test_return_stats_trim_warmup_rows():
    df = _long({"a": (0.0004, 0.01), "b": (0.0003, 0.012)}, n=300, warmup=50)
    per, _ = bootstrap_strategy_return_stats(df, n_boot=500, seed=6)
    assert (per["n_obs"] == 300).all()  # the 50 leading all-cash rows are excluded
