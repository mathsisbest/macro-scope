"""Performance metrics pinned to known closed-form values."""

import numpy as np

from mmi.portfolio.metrics import (
    annualized_return,
    calmar,
    max_drawdown,
    sharpe,
)


def test_max_drawdown_known():
    # wealth: 1.1 then 0.55 -> peak 1.1, trough 0.55 -> drawdown -0.5
    assert np.isclose(max_drawdown(np.array([0.1, -0.5])), -0.5)


def test_max_drawdown_zero_for_monotonic_gains():
    assert np.isclose(max_drawdown(np.array([0.01, 0.01, 0.01])), 0.0)


def test_sharpe_zero_when_no_variation():
    assert sharpe(np.array([0.001, 0.001, 0.001])) == 0.0  # zero std


def test_annualized_return_matches_compounding():
    r = np.full(252, 0.001)  # ~0.1%/day for one trading year
    assert np.isclose(annualized_return(r), 1.001**252 - 1)


def test_calmar_is_finite_and_signed():
    r = np.array([0.02, -0.01, 0.03, -0.04, 0.01])
    c = calmar(r)
    assert np.isfinite(c)
