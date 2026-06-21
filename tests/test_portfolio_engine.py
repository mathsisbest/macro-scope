"""Weight solvers: correct shapes, and ERC genuinely equalises risk contributions."""

import numpy as np

from mmi.portfolio.engine import (
    equal_weight,
    inverse_volatility,
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
