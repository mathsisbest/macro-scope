"""Backtest performance metrics on a daily return series (any array-like of simple returns).

All annualised with a 252-trading-day convention. Kept as small, individually-tested pure
functions so the numbers shown in the dashboard/LLM are auditable.
"""

from __future__ import annotations

import numpy as np

_TRADING_DAYS = 252


def annualized_return(returns: np.ndarray, periods: int = _TRADING_DAYS) -> float:
    """Compound annual growth rate (CAGR) from a daily simple-return series."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + r))
    return growth ** (periods / r.size) - 1.0


def annualized_vol(returns: np.ndarray, periods: int = _TRADING_DAYS) -> float:
    """Annualised standard deviation of daily returns."""
    r = np.asarray(returns, dtype=float)
    return float(r.std(ddof=1) * np.sqrt(periods)) if r.size > 1 else 0.0


def sharpe(returns: np.ndarray, rf: float = 0.0, periods: int = _TRADING_DAYS) -> float:
    """Annualised Sharpe ratio. ``rf`` is an annual risk-free rate (converted to per-period)."""
    r = np.asarray(returns, dtype=float)
    excess = r - rf / periods
    sd = excess.std(ddof=1)
    if r.size < 2 or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the cumulative wealth curve (a negative number)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float((cum / peak - 1.0).min())


def calmar(returns: np.ndarray, periods: int = _TRADING_DAYS) -> float:
    """CAGR divided by the absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return annualized_return(returns, periods) / abs(mdd)
