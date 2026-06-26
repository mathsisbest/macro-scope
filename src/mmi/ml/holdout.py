"""Locked-holdout carving — a shared, pure helper for the ML models.

A *locked holdout* is the LAST ``holdout_size`` time-ordered observations of a feature
frame, set aside before any cross-validation or training and scored exactly once by a model
final-fit on everything before it.  It is an **honest extra out-of-sample readout**: the
walk-forward CV (and therefore the skill gate) only ever sees the DEV portion, and the
holdout metrics are *reported, not gated*.  The holdout is NEVER used to tune the model, the
features, or the gate thresholds.

This module is intentionally tiny and DB-free so both the volatility (``rv_har``) and the
direction (``random_forest``) models share one carving rule and one small-data guard, and so
the rule is unit-testable in isolation.
"""

from __future__ import annotations

import math

# Largest holdout we ever carve: ~1 trading year.  On the small CI/sample data the 20% rule
# bites first, so this cap only matters on multi-year real data.
_MAX_HOLDOUT: int = 252
# Fraction of the (time-ordered) observations to reserve as the holdout.
_HOLDOUT_FRAC: float = 0.2
# Canonical minimum trainable observations required to attempt training.  Lives here (the
# leaf module both vol + direction models import) so the two models share ONE floor instead
# of duplicating the literal.
MIN_OBS: int = 60


def holdout_size(n_obs: int) -> int:
    """Size of the locked holdout for ``n_obs`` time-ordered observations.

    ``min(252, floor(0.2 * n_obs))`` — ~1 trading year on real data, ~20% on the smaller
    sample/CI data.  Returns 0 (no holdout) for non-positive ``n_obs``.
    """
    if n_obs <= 0:
        return 0
    return min(_MAX_HOLDOUT, math.floor(_HOLDOUT_FRAC * n_obs))


def split_indices(n_obs: int, min_dev: int) -> tuple[int, int]:
    """Return ``(dev_end, hold)`` for an ``n_obs``-row, time-ordered series.

    The DEV portion is ``rows[:dev_end]`` and the locked holdout is the tail
    ``rows[dev_end:]`` (length ``hold``).  ``hold`` is :func:`holdout_size`, UNLESS carving it
    would leave fewer than ``min_dev`` dev rows — in that case we SKIP the holdout
    (``hold == 0``, ``dev_end == n_obs``) so the caller runs CV on the full series as before.

    Pure and deterministic; the caller logs the skip.
    """
    hold = holdout_size(n_obs)
    if hold <= 0 or (n_obs - hold) < min_dev:
        return n_obs, 0
    return n_obs - hold, hold
