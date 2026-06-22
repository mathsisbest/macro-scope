"""Walk-forward mu panel: strictly point-in-time (truncation-invariant), correct shapes + skill."""

import numpy as np
import pandas as pd

from mmi.ml.forecast_panel import walk_forward_mu


def _long(n: int = 400, assets: tuple = ("AAA", "BBB"), seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rows = []
    for asset in assets:
        rets = rng.normal(0.0004, 0.01, n)
        for day, ret in zip(idx, rets, strict=True):
            rows.append({"symbol": asset, "date": day, "daily_return": float(ret)})
    return pd.DataFrame(rows), idx


def test_mu_panel_shapes_and_skill_columns():
    df, idx = _long(400)
    rebals = list(idx[200::21])  # ~monthly after warm-up
    mu, skill = walk_forward_mu(df, rebals, horizon=5, min_train=60, n_estimators=40, seed=0)
    assert set(mu.columns) == {"date", "symbol", "mu"}
    assert set(mu["symbol"].unique()) <= {"AAA", "BBB"}
    assert set(mu["date"].unique()) <= set(pd.to_datetime(rebals))  # only at rebalance dates
    assert mu["mu"].notna().all()
    assert mu["mu"].abs().max() < 0.05  # daily-equivalent scale (not cumulative horizon returns)
    assert {"symbol", "horizon", "n_preds", "mae", "baseline_mae", "r2_oos", "dir_acc"} <= set(
        skill.columns
    )


def test_no_lookahead_truncating_future_keeps_early_forecasts_identical():
    """The real guard: an early-rebalance forecast must not change when later data is removed.

    Catches an embargo/feature leak — if the model peeked at returns on/after the rebalance, adding
    or removing distant-future data would shift the forecast.
    """
    df, idx = _long(400)
    rebals = list(idx[200::21])
    early = [r for r in rebals if r <= idx[300]]
    full, _ = walk_forward_mu(df, rebals, horizon=5, min_train=60, n_estimators=40, seed=0)
    trunc_df = df[df["date"] <= idx[330]]
    trunc, _ = walk_forward_mu(trunc_df, rebals, horizon=5, min_train=60, n_estimators=40, seed=0)
    assert early  # there are early rebalances to compare
    for rebal in early:
        a = full[full["date"] == rebal].set_index("symbol")["mu"]
        b = trunc[trunc["date"] == rebal].set_index("symbol")["mu"]
        common = a.index.intersection(b.index)
        assert len(common) > 0
        assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy())


def test_reproducible_given_seed():
    df, idx = _long(400)
    rebals = list(idx[200::21])
    a, _ = walk_forward_mu(df, rebals, horizon=5, min_train=60, n_estimators=40, seed=0)
    b, _ = walk_forward_mu(df, rebals, horizon=5, min_train=60, n_estimators=40, seed=0)
    pd.testing.assert_frame_equal(a, b)
