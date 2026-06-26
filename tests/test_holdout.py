"""Unit tests for the shared locked-holdout carving helper (mmi.ml.holdout).

The locked holdout is the LAST ``holdout_size`` time-ordered rows, set aside before any CV
and scored once by a model final-fit on everything before it.  It is an honest extra OOS
readout — reported, NEVER used to tune anything.  These tests pin the carving rule and the
small-data skip guard in isolation, with no DB and no model.
"""

from __future__ import annotations

import math

import pytest

from mmi.ml.holdout import _MAX_HOLDOUT, holdout_size, split_indices


@pytest.mark.parametrize(
    ("n_obs", "expected"),
    [
        (0, 0),
        (-5, 0),
        (10, 2),  # floor(0.2 * 10)
        (100, 20),  # floor(0.2 * 100)
        (375, 75),  # the vol-model sample-data count
        (379, 75),  # the direction-model sample-data count
        (1260, _MAX_HOLDOUT),  # 0.2 * 1260 = 252 == cap
        (5000, _MAX_HOLDOUT),  # cap bites on multi-year real data
    ],
)
def test_holdout_size_rule(n_obs: int, expected: int) -> None:
    """holdout_size == min(252, floor(0.2 * n_obs)), 0 for non-positive n."""
    assert holdout_size(n_obs) == expected


def test_holdout_size_never_exceeds_cap() -> None:
    """The holdout is capped at ~1 trading year regardless of how much data we have."""
    for n in (1260, 2520, 10_000):
        assert holdout_size(n) <= _MAX_HOLDOUT


def test_split_indices_carves_tail_when_dev_sufficient() -> None:
    """With ample data, split is (n - holdout, holdout) and the holdout is the TAIL."""
    n = 375
    dev_end, hold = split_indices(n, min_dev=60)
    assert hold == holdout_size(n) == 75
    assert dev_end == n - hold == 300
    # The split partitions the series exactly: dev = [0, dev_end), holdout = [dev_end, n).
    assert dev_end + hold == n
    # Dev comfortably clears the minimum.
    assert dev_end >= 60


def test_split_indices_skips_when_dev_would_fall_below_min() -> None:
    """If carving the holdout would leave < min_dev dev rows, the holdout is SKIPPED.

    Skipping is signalled by hold == 0 and dev_end == n, so the caller runs CV on the full
    series exactly as before — no crash, no partial holdout.
    """
    # n=70, holdout_size=floor(0.2*70)=14 -> dev=56 < min_dev=60 -> SKIP.
    n = 70
    assert holdout_size(n) == 14
    dev_end, hold = split_indices(n, min_dev=60)
    assert hold == 0, "holdout must be skipped when dev would fall below the minimum"
    assert dev_end == n, "on skip, the full series is the dev portion"


def test_split_indices_boundary_exactly_min_dev() -> None:
    """When dev would equal exactly min_dev, the holdout is KEPT (>=, not >)."""
    # Find an n where n - holdout_size(n) == min_dev exactly.
    min_dev = 60
    # n=75 -> holdout=15 -> dev=60 == min_dev (kept).
    n = 75
    assert math.floor(0.2 * n) == 15
    dev_end, hold = split_indices(n, min_dev=min_dev)
    assert hold == 15 and dev_end == 60, "dev == min_dev should keep the holdout"
