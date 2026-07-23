"""Pure deterministic skill-verdict helper for the HAR realized-volatility model.

This module is SCOPED to the volatility model (model='rv_har').  It reads rows
from a long-format model_metrics DataFrame and returns a verdict dict.  There is
NO DB, NO network, NO model import, NO recomputation here — just arithmetic on
already-persisted metrics.

Threshold constants are FIXED and must NEVER be re-tuned to make a run pass.
They encode the minimum honest skill bar agreed in docs/GO_LIVE_PLAN.md Contract E.
"""

from __future__ import annotations

import math
from typing import TypedDict

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed module constants — NEVER re-tuned to make a run pass.
# Defined in Contract E (docs/GO_LIVE_PLAN.md §2).
# ---------------------------------------------------------------------------
R2_MIN: float = 0.10  # minimum OOS R² vs persistence baseline
QLIKE_MARGIN: float = 0.01  # model QLIKE must be < (1 - QLIKE_MARGIN) × baseline_qlike
# i.e. qlike_skill_ratio < 0.99
SUSTAIN_FRAC: float = 0.6  # fraction of folds that must individually beat baseline
N_OBS_MIN: int = 250  # minimum in-sample + OOS observations for reliability


class SkillVerdict(TypedDict):
    cleared: bool
    reasons: list[str]
    oos_r2: float | None
    qlike_skill_ratio: float | None
    folds_passed: int | None
    n_folds: int | None
    n_obs: int | None


def _to_finite_float(value: float | None) -> float | None:
    """Coerce a stored metric to a finite float.

    Returns ``None`` if the value is missing (``None``), non-numeric, NaN, or
    ±inf.  This is what makes the gate FAIL CLOSED: ``float('nan') < R2_MIN`` and
    ``float('nan') >= threshold`` are BOTH ``False`` in Python, so a NaN metric
    would otherwise silently satisfy every comparison and clear the gate.  We
    treat NaN/inf exactly like a missing metric instead.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _to_finite_int(value: float | None) -> int | None:
    """Coerce a stored metric to an int, failing closed (never raising).

    Returns ``None`` for missing/NaN/±inf/non-numeric values instead of letting
    ``int(float('nan'))`` raise ``ValueError`` — preserving the contract that
    absent or partial rows yield a verdict, never an exception.
    """
    f = _to_finite_float(value)
    if f is None:
        return None
    return int(f)


def skill_verdict(
    model_metrics_df: pd.DataFrame,
    symbol: str = "SPY",
) -> SkillVerdict:
    """Return the go-live skill verdict for the HAR realized-volatility model.

    Parameters
    ----------
    model_metrics_df:
        Long-format DataFrame with columns ``(model, symbol, metric, value,
        trained_at)``.  Extra columns are silently ignored.
    symbol:
        The asset ticker to evaluate (default ``'SPY'``).

    Returns
    -------
    SkillVerdict
        ``cleared`` is ``True`` only when ALL four gate conditions are met:

        * ``oos_r2 >= R2_MIN`` (0.10)
        * ``qlike_skill_ratio < 1 - QLIKE_MARGIN`` (< 0.99)
        * ``folds_passed >= ceil(SUSTAIN_FRAC * n_folds)`` (≥ 3 of 5 by default)
        * ``n_obs >= N_OBS_MIN`` (250)

        Absent, partial, non-finite (NaN / ±inf), or implausible out-of-range
        metric values for the requested ``(model='rv_har', symbol)`` combination
        always yield ``cleared=False`` with an explicit reason string — never an
        exception.  The gate FAILS CLOSED: a NaN/inf metric is treated exactly
        like a missing one, and a finite-but-impossible value (e.g. ``oos_r2 >
        1.0``, ``n_folds < 1``, or a negative ``qlike_skill_ratio``) is rejected
        too — never silently passed.
    """
    # -----------------------------------------------------------------------
    # Guard: must have the mandatory columns.
    # -----------------------------------------------------------------------
    for col in ("model", "symbol", "metric", "value"):
        if col not in model_metrics_df.columns:
            return SkillVerdict(
                cleared=False,
                reasons=[f"model_metrics_df is missing required column '{col}'"],
                oos_r2=None,
                qlike_skill_ratio=None,
                folds_passed=None,
                n_folds=None,
                n_obs=None,
            )

    # -----------------------------------------------------------------------
    # Filter to rv_har rows for the requested symbol.
    # -----------------------------------------------------------------------
    subset = model_metrics_df[
        (model_metrics_df["model"] == "rv_har") & (model_metrics_df["symbol"] == symbol)
    ]

    if subset.empty:
        return SkillVerdict(
            cleared=False,
            reasons=[
                f"no rows found for model='rv_har', symbol='{symbol}' in "
                "model_metrics_df — model has not been trained yet"
            ],
            oos_r2=None,
            qlike_skill_ratio=None,
            folds_passed=None,
            n_folds=None,
            n_obs=None,
        )

    # -----------------------------------------------------------------------
    # Extract metric values by NAME (Contract D: metrics are rows, not columns).
    # Use the most-recent trained_at row per metric if duplicates exist.
    # -----------------------------------------------------------------------
    if "trained_at" in subset.columns:
        subset = subset.sort_values("trained_at", ascending=True)

    metric_map: dict[str, float] = (
        subset.drop_duplicates(subset=["metric"], keep="last")
        .set_index("metric")["value"]
        .to_dict()
    )

    # -----------------------------------------------------------------------
    # Coerce every required metric to a usable numeric value.  A metric that is
    # ABSENT, None, NaN, or ±inf is treated IDENTICALLY: it makes the verdict
    # FAIL CLOSED.  This is the honesty gate — a degenerate run (zero-variance
    # baseline, tiny sample) that writes NaN must NEVER silently clear, because
    # ``nan < R2_MIN`` and ``nan >= threshold`` are both False in Python.
    # -----------------------------------------------------------------------
    oos_r2_v = _to_finite_float(metric_map.get("oos_r2"))
    qlike_skill_ratio_v = _to_finite_float(metric_map.get("qlike_skill_ratio"))
    folds_passed_v = _to_finite_int(metric_map.get("folds_passed"))
    n_folds_v = _to_finite_int(metric_map.get("n_folds"))
    n_obs_v = _to_finite_int(metric_map.get("n_obs"))

    if (
        oos_r2_v is None
        or qlike_skill_ratio_v is None
        or folds_passed_v is None
        or n_folds_v is None
        or n_obs_v is None
    ):
        unusable = sorted(
            name
            for name, value in (
                ("oos_r2", oos_r2_v),
                ("qlike_skill_ratio", qlike_skill_ratio_v),
                ("folds_passed", folds_passed_v),
                ("n_folds", n_folds_v),
                ("n_obs", n_obs_v),
            )
            if value is None
        )
        return SkillVerdict(
            cleared=False,
            reasons=[
                "missing or non-finite metric(s) for model='rv_har', "
                f"symbol='{symbol}': " + ", ".join(unusable)
            ],
            oos_r2=oos_r2_v,
            qlike_skill_ratio=qlike_skill_ratio_v,
            folds_passed=folds_passed_v,
            n_folds=n_folds_v,
            n_obs=n_obs_v,
        )

    # -----------------------------------------------------------------------
    # All five metrics are present and finite (mypy narrows each to non-None).
    # -----------------------------------------------------------------------
    oos_r2: float = oos_r2_v
    qlike_skill_ratio: float = qlike_skill_ratio_v
    folds_passed: int = folds_passed_v
    n_folds: int = n_folds_v
    n_obs: int = n_obs_v

    # -----------------------------------------------------------------------
    # Domain guard: a metric can be FINITE yet physically impossible (a sign of
    # a degenerate run or an upstream bug).  These "garbage but finite" values
    # must ALSO fail closed, because each would otherwise slip through a
    # threshold comparison written assuming a sane domain:
    #   * oos_r2 > 1.0 would satisfy ``>= R2_MIN``,
    #   * n_folds < 1 makes ceil(SUSTAIN_FRAC*n_folds)=0 so the folds check is
    #     vacuously true,
    #   * a negative qlike_skill_ratio would satisfy ``< qlike_threshold``.
    # Treat out-of-range exactly like missing/non-finite: cleared=False.
    # -----------------------------------------------------------------------
    implausible: list[str] = []
    if oos_r2 > 1.0:
        implausible.append(f"oos_r2={oos_r2:.4f} > 1.0 (R² cannot exceed 1)")
    if qlike_skill_ratio < 0.0:
        implausible.append(
            f"qlike_skill_ratio={qlike_skill_ratio:.4f} < 0 "
            "(a ratio of non-negative losses cannot be negative)"
        )
    if n_folds < 1:
        implausible.append(f"n_folds={n_folds} < 1 (no cross-validation folds ran)")
    if folds_passed < 0 or folds_passed > n_folds:
        implausible.append(f"folds_passed={folds_passed} outside [0, n_folds={n_folds}]")
    if n_obs < 0:
        implausible.append(f"n_obs={n_obs} < 0")

    if implausible:
        return SkillVerdict(
            cleared=False,
            reasons=[
                "implausible (out-of-range) metric(s) for model='rv_har', "
                f"symbol='{symbol}': " + "; ".join(implausible)
            ],
            oos_r2=oos_r2,
            qlike_skill_ratio=qlike_skill_ratio,
            folds_passed=folds_passed,
            n_folds=n_folds,
            n_obs=n_obs,
        )

    # -----------------------------------------------------------------------
    # Evaluate gate conditions.
    # -----------------------------------------------------------------------
    reasons: list[str] = []

    if oos_r2 < R2_MIN:
        reasons.append(
            f"oos_r2={oos_r2:.4f} < R2_MIN={R2_MIN} — model does not beat "
            "the persistence baseline out-of-sample"
        )

    qlike_threshold = 1.0 - QLIKE_MARGIN
    if qlike_skill_ratio >= qlike_threshold:
        reasons.append(
            f"qlike_skill_ratio={qlike_skill_ratio:.4f} >= {qlike_threshold} — "
            "model QLIKE does not meaningfully improve on baseline QLIKE"
        )

    min_folds_needed = math.ceil(SUSTAIN_FRAC * n_folds)
    if folds_passed < min_folds_needed:
        reasons.append(
            f"folds_passed={folds_passed} < ceil({SUSTAIN_FRAC}×{n_folds})={min_folds_needed} — "
            "skill is not sustained across enough cross-validation folds"
        )

    if n_obs < N_OBS_MIN:
        reasons.append(
            f"n_obs={n_obs} < N_OBS_MIN={N_OBS_MIN} — "
            "insufficient observations for a reliable estimate"
        )

    cleared = len(reasons) == 0

    return SkillVerdict(
        cleared=cleared,
        reasons=reasons,
        oos_r2=oos_r2,
        qlike_skill_ratio=qlike_skill_ratio,
        folds_passed=folds_passed,
        n_folds=n_folds,
        n_obs=n_obs,
    )


def return_forecast_skill_verdict(
    model_metrics_df: pd.DataFrame,
    symbol: str = "SPY",
    model: str = "return_gb",
) -> SkillVerdict:
    """Return skill verdict for Return Forecast models (R2 > 0, Directional Accuracy > 0.50)."""
    for col in ("model", "symbol", "metric", "value"):
        if col not in model_metrics_df.columns:
            return SkillVerdict(
                cleared=False,
                reasons=[f"model_metrics_df is missing required column '{col}'"],
                oos_r2=None,
                qlike_skill_ratio=None,
                folds_passed=None,
                n_folds=None,
                n_obs=None,
            )

    subset = model_metrics_df[
        (model_metrics_df["model"] == model) & (model_metrics_df["symbol"] == symbol)
    ]
    if subset.empty:
        return SkillVerdict(
            cleared=False,
            reasons=[f"no rows found for model='{model}', symbol='{symbol}'"],
            oos_r2=None,
            qlike_skill_ratio=None,
            folds_passed=None,
            n_folds=None,
            n_obs=None,
        )

    if "trained_at" in subset.columns:
        subset = subset.sort_values("trained_at", ascending=True)

    metric_map: dict[str, float] = (
        subset.drop_duplicates(subset=["metric"], keep="last")
        .set_index("metric")["value"]
        .to_dict()
    )

    r2 = _to_finite_float(metric_map.get("r2"))
    dir_acc = _to_finite_float(metric_map.get("direction_accuracy"))

    reasons: list[str] = []
    if r2 is None or r2 <= 0:
        reasons.append(f"r2={r2} <= 0 — model does not produce positive out-of-sample R2")
    if dir_acc is None or dir_acc <= 0.50:
        reasons.append(
            f"direction_accuracy={dir_acc} <= 0.50 — directional accuracy is at or below coin flip"
        )

    return SkillVerdict(
        cleared=len(reasons) == 0,
        reasons=reasons,
        oos_r2=r2,
        qlike_skill_ratio=None,
        folds_passed=1 if len(reasons) == 0 else 0,
        n_folds=1,
        n_obs=int(metric_map.get("prediction_count", 0)),
    )

