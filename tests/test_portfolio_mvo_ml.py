"""Gated mvo_ml: point-in-time mu blend (no look-ahead), lambda-bounded, runs through the engine."""

import numpy as np
import pandas as pd
import pytest

from mmi.portfolio.backtest import MVO_ML, rebalance_dates, run_backtest
from mmi.portfolio.compute import (
    build_ml_mu_panel,
    compute_ml_mu_panel,
    compute_portfolio_returns,
)


def _panel(n: int = 320, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    data = rng.normal(0.0004, 0.01, size=(n, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _long(n: int = 200, assets: tuple = ("A", "B", "C"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rows = []
    for asset in assets:
        rets = rng.normal(0.0004, 0.01, n)
        for day, ret in zip(idx, rets, strict=True):
            rows.append({"symbol": asset, "date": day, "daily_return": float(ret)})
    return pd.DataFrame(rows)


def test_gate_lambda_bounded_and_blend_is_prior_when_no_skill():
    panel = _panel(320)
    rebals = rebalance_dates(panel.index, "M", 60)
    mu, gate = build_ml_mu_panel(panel, rebals, lookback=60, horizon=5, lambda_max=0.5)
    assert set(gate.columns) == {"date", "forecast_skill", "forecast_weight"}
    assert ((gate["forecast_weight"] >= 0.0) & (gate["forecast_weight"] <= 0.5 + 1e-12)).all()
    # noise returns: the forecast has no edge over the prior, so the first rebalance gates to 0
    assert gate.sort_values("date")["forecast_weight"].iloc[0] == 0.0


def test_build_ml_mu_panel_truncation_invariant():
    """Gate look-ahead guard: early-rebalance mu is identical whether or not later data exists."""
    panel = _panel(320)
    rebals = rebalance_dates(panel.index, "M", 60)
    early = [r for r in rebals if r <= panel.index[230]]
    full, _ = build_ml_mu_panel(panel, rebals, lookback=60, horizon=5, lambda_max=0.5)
    trunc_panel = panel.iloc[:260]
    trunc_rebals = [r for r in rebals if r in trunc_panel.index]
    trunc, _ = build_ml_mu_panel(trunc_panel, trunc_rebals, lookback=60, horizon=5, lambda_max=0.5)
    assert early
    for rebal in early:
        a = full[full["date"] == rebal].set_index("symbol")["mu"]
        b = trunc[trunc["date"] == rebal].set_index("symbol")["mu"]
        common = a.index.intersection(b.index)
        assert len(common) > 0
        assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy())


def test_mvo_ml_requires_a_mu_panel_and_runs_with_one():
    panel = _panel(320)
    with pytest.raises(ValueError):
        run_backtest(panel, strategy=MVO_ML, lookback=60, freq="M")  # no mu_panel -> error
    rebals = rebalance_dates(panel.index, "M", 60)
    mu, _ = build_ml_mu_panel(panel, rebals, lookback=60, horizon=5, lambda_max=0.5)
    out = run_backtest(panel, strategy=MVO_ML, lookback=60, freq="M", mu_panel=mu)
    assert list(out.columns) == ["daily_return", "cumulative_return"]
    assert out["daily_return"].notna().all()


def test_compute_includes_mvo_ml_when_enabled():
    out = compute_portfolio_returns(_long(200), lookback=30, freq="M", horizon=5, include_ml=True)
    assert "mvo_ml" in set(out["strategy"].unique())


def test_mvo_ml_equals_mvo_histmean_when_lambda_max_zero():
    # lambda_max=0 forces lambda(t)=0 everywhere -> mu_blend == mu_hist -> identical to mvo_histmean
    # (same Ledoit-Wolf cov, same mu). This locks the "comparison isolates the ML mu" guarantee.
    panel = _panel(320)
    rebals = rebalance_dates(panel.index, "M", 60)
    mu, _ = build_ml_mu_panel(panel, rebals, lookback=60, horizon=5, lambda_max=0.0)
    ml = run_backtest(panel, strategy=MVO_ML, lookback=60, freq="M", mu_panel=mu)
    hist = run_backtest(panel, strategy="mvo_histmean", lookback=60, freq="M")
    assert np.allclose(ml["daily_return"].to_numpy(), hist["daily_return"].to_numpy())


def test_mu_panel_has_complete_coverage_and_no_nan():
    panel = _panel(320)
    rebals = rebalance_dates(panel.index, "M", 60)
    mu, _ = build_ml_mu_panel(panel, rebals, lookback=60, horizon=5, lambda_max=0.5)
    assert not mu["mu"].isna().any()  # every blend is finite (missing forecasts fall back to prior)
    pairs = set(map(tuple, mu[["date", "symbol"]].to_numpy()))
    assert pairs == {(d, s) for d in rebals for s in panel.columns}  # one row per (rebal, symbol)


def test_compute_ml_mu_panel_returns_mu_and_gate():
    mu, gate = compute_ml_mu_panel(_long(200), lookback=30, horizon=5)
    assert set(mu.columns) == {"date", "symbol", "mu"}  # mu_panel is internal; no window stamp
    assert set(gate.columns) == {"window_id", "date", "forecast_skill", "forecast_weight"}


def test_precomputed_ml_mu_panel_matches_building_internally():
    # The cmd_portfolio dedup: passing a precomputed panel gives the SAME mvo_ml as building inline.
    df = _long(200)
    mu, _ = compute_ml_mu_panel(df, lookback=30, horizon=5)
    passed = compute_portfolio_returns(df, lookback=30, freq="M", horizon=5, ml_mu_panel=mu)
    internal = compute_portfolio_returns(df, lookback=30, freq="M", horizon=5, include_ml=True)
    a = passed[passed["strategy"] == "mvo_ml"].reset_index(drop=True)
    b = internal[internal["strategy"] == "mvo_ml"].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
