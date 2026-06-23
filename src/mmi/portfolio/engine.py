"""Portfolio weight solvers — long-only, sum-to-1, pure functions on a covariance matrix.

Four strategies of escalating sophistication:
- ``equal_weight``       — the benchmark; 1/N.
- ``inverse_volatility`` — w_i proportional to 1/sigma_i.
- ``risk_parity``        — TRUE equal-risk-contribution (each asset contributes equally to
  portfolio variance), solved numerically. This is the proper Bridgewater-style formulation and
  is distinct from naive inverse-vol — the two coincide only when assets are uncorrelated.
- ``max_sharpe``         — Markowitz tangency portfolio (max Sharpe), long-only with a per-asset
  cap. The expected-returns vector ``mu`` is supplied by the caller (a trailing-window mean for the
  honest baseline; an ML forecast later), so this module stays a pure solver.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def equal_weight(n: int) -> np.ndarray:
    """1/N weights."""
    return np.full(n, 1.0 / n)


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    """Weights proportional to the inverse of each asset's volatility, normalised to sum to 1.

    Volatilities are floored at a tiny epsilon so a zero-variance asset yields a finite (large)
    weight rather than NaN/inf.
    """
    inv = 1.0 / np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return inv / inv.sum()


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-asset contribution to portfolio variance: ``w_i * (cov @ w)_i`` (sums to w'cov w)."""
    return weights * (cov @ weights)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution weights (long-only, sum to 1), solved with SLSQP.

    Minimises the dispersion of per-asset risk contributions; at the optimum every asset
    contributes an equal share of total portfolio variance.

    The covariance is scale-normalised (divided by its mean variance) before solving. ERC weights
    are invariant to multiplying the covariance by a positive scalar, but the objective's magnitude
    is not: on *daily* covariances (diagonal ~1e-4) the raw objective is ~1e-9, so SLSQP's absolute
    ``ftol=1e-12`` is met — and its finite-difference gradients vanish into numerical noise — at the
    1/N start, returning 1/N regardless of the true risk structure (silently collapsing risk_parity
    to equal_weight). Normalising makes the contributions O(1) so convergence and gradients are
    meaningful; the optimum is mathematically unchanged.
    """
    n = cov.shape[0]
    mean_var = float(np.mean(np.diag(cov)))
    cov_n = cov / mean_var if mean_var > 0 else cov

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov_n)
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


def max_sharpe(cov: np.ndarray, mu: np.ndarray, *, max_weight: float = 0.40) -> np.ndarray:
    """Long-only max-Sharpe (tangency) weights: maximise ``(w·mu) / sqrt(w'cov w)``, summing to 1.

    A per-asset cap curbs the concentration mean-variance optimisation is prone to; it is relaxed
    to ``1/n`` when the requested cap would make a fully-invested long-only portfolio infeasible
    (too few assets). Solved with SLSQP. ``mu`` and ``cov`` may be on any consistent scale — the
    Sharpe ratio (hence the argmax) is invariant to a positive rescaling of either.
    """
    n = len(mu)
    cap = max(max_weight, 1.0 / n)

    def neg_sharpe(w: np.ndarray) -> float:
        variance = float(w @ cov @ w)
        if variance <= 0.0:
            return 0.0
        return -float(w @ mu) / np.sqrt(variance)

    result = minimize(
        neg_sharpe,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    total = float(weights.sum())
    return weights / total if total > 0 else equal_weight(n)


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of a ``(n_obs, n_assets)`` return window.

    Shrinking the sample covariance toward a scaled identity conditions the estimate so the
    max-Sharpe optimiser does not chase a near-singular sample matrix — the mean-variance
    instability that a noisy (ML-forecast) ``mu`` would otherwise amplify. Thin wrapper over
    scikit-learn's reference implementation; falls back to the sample covariance for < 2 rows.
    """
    from sklearn.covariance import ledoit_wolf

    x = np.asarray(returns, dtype=float)
    if x.shape[0] < 2:
        return np.atleast_2d(np.cov(x, rowvar=False))
    cov, _shrinkage = ledoit_wolf(x)
    return cov
