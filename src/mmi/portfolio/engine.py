"""Portfolio weight solvers — long-only, sum-to-1, pure functions on a covariance matrix.

Three strategies of escalating sophistication:
- ``equal_weight``       — the benchmark; 1/N.
- ``inverse_volatility`` — w_i proportional to 1/sigma_i.
- ``risk_parity``        — TRUE equal-risk-contribution (each asset contributes equally to
  portfolio variance), solved numerically. This is the proper Bridgewater-style formulation and
  is distinct from naive inverse-vol — the two coincide only when assets are uncorrelated.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def equal_weight(n: int) -> np.ndarray:
    """1/N weights."""
    return np.full(n, 1.0 / n)


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    """Weights proportional to the inverse of each asset's volatility, normalised to sum to 1."""
    inv = 1.0 / np.sqrt(np.diag(cov))
    return inv / inv.sum()


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-asset contribution to portfolio variance: ``w_i * (cov @ w)_i`` (sums to w'cov w)."""
    return weights * (cov @ weights)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution weights (long-only, sum to 1), solved with SLSQP.

    Minimises the dispersion of per-asset risk contributions; at the optimum every asset
    contributes an equal share of total portfolio variance.
    """
    n = cov.shape[0]

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        return float(np.sum((rc - rc.mean()) ** 2))

    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    weights = np.asarray(result.x, dtype=float)
    return weights / weights.sum()
