"""Weight solvers: correct shapes, and ERC genuinely equalises risk contributions."""

import numpy as np

from mmi.portfolio.engine import (
    equal_weight,
    inverse_volatility,
    ledoit_wolf_cov,
    max_sharpe,
    risk_contributions,
    risk_parity,
)


def test_equal_weight():
    assert np.allclose(equal_weight(4), 0.25)


def test_inverse_volatility_weights():
    cov = np.diag([0.04, 0.01])  # vols 0.2 and 0.1 -> weights 1/3, 2/3
    w = inverse_volatility(cov)
    assert np.allclose(w, [1 / 3, 2 / 3])
    assert np.isclose(w.sum(), 1.0)


def test_inverse_volatility_floors_zero_variance():
    cov = np.array([[0.04, 0.0], [0.0, 0.0]])  # second asset has zero variance
    w = inverse_volatility(cov)
    assert np.isfinite(w).all()  # finite, not NaN/inf
    assert np.isclose(w.sum(), 1.0)


def test_risk_parity_matches_inverse_vol_when_uncorrelated():
    # A known property: with zero correlations, ERC == inverse-vol.
    cov = np.diag([0.04, 0.01, 0.0025])
    assert np.allclose(risk_parity(cov), inverse_volatility(cov), atol=1e-4)


def test_risk_parity_equalises_risk_contributions():
    cov = np.array(
        [
            [0.0400, 0.0060, 0.0000],
            [0.0060, 0.0100, -0.0020],
            [0.0000, -0.0020, 0.0225],
        ]
    )
    w = risk_parity(cov)
    rc = risk_contributions(w, cov)
    assert np.allclose(rc, rc.mean(), rtol=1e-3)  # equal risk contributions (the ERC definition)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()  # long-only


def test_risk_parity_solves_erc_at_daily_covariance_scale():
    # Regression guard for the scale bug: at *daily* magnitudes (entries ~1e-4) with a high-vol
    # (BTC-like) asset, the raw objective fell below SLSQP's ftol at the 1/N start, so risk_parity
    # silently returned equal_weight. It must now solve true ERC (== inverse-vol when uncorrelated).
    cov = np.diag([1.0, 1.0, 1.0, 36.0]) * 1e-4  # last asset ~6x the vol of the others
    w = risk_parity(cov)
    assert np.allclose(w, inverse_volatility(cov), atol=1e-3)  # true ERC, not 1/N
    assert not np.allclose(w, equal_weight(4), atol=0.05)  # the buggy 1/N answer is rejected
    rc = risk_contributions(w, cov)
    assert np.allclose(rc, rc.mean(), rtol=1e-3)  # contributions genuinely equalised


def test_risk_parity_is_scale_invariant():
    # ERC weights are invariant to scaling the covariance by a positive scalar; the fix must hold
    # this at daily magnitudes, where the unscaled solver previously diverged to 1/N.
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.25]])
    assert np.allclose(risk_parity(cov), risk_parity(cov * 1e-4), atol=1e-4)


def test_max_sharpe_favours_the_higher_reward_per_risk_asset():
    cov = np.diag([0.04, 0.01])  # A vol 0.20, B vol 0.10
    mu = np.array([0.10, 0.08])
    w = max_sharpe(cov, mu, max_weight=1.0)
    assert w[1] > w[0]  # B's reward-per-risk (0.08/0.10) beats A's (0.10/0.20) -> larger weight
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all()  # long-only


def test_max_sharpe_respects_the_weight_cap():
    cov = np.diag([0.01, 0.04, 0.04])  # asset 0 dominates (high return, low vol)
    mu = np.array([0.10, 0.02, 0.02])
    w = max_sharpe(cov, mu, max_weight=0.40)
    assert w[0] <= 0.40 + 1e-6  # capped — not all-in on asset 0
    assert np.isclose(w.sum(), 1.0)


def test_ledoit_wolf_cov_is_psd_symmetric_and_shrinks_off_diagonals():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, (200, 5))
    cov = ledoit_wolf_cov(x)
    assert cov.shape == (5, 5)
    assert np.allclose(cov, cov.T)  # symmetric
    assert (np.linalg.eigvalsh(cov) > 0).all()  # positive-definite
    sample = np.cov(x, rowvar=False)
    off = ~np.eye(5, dtype=bool)
    assert np.abs(cov[off]).sum() < np.abs(sample[off]).sum()  # off-diagonals shrunk toward 0


def test_ledoit_wolf_conditions_a_near_singular_covariance():
    # two near-collinear assets -> near-singular sample cov; shrinkage improves the conditioning
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, 200)
    x = np.column_stack([base, base + rng.normal(0, 1e-5, 200), rng.normal(0, 0.01, 200)])
    assert np.linalg.cond(ledoit_wolf_cov(x)) < np.linalg.cond(np.cov(x, rowvar=False))
