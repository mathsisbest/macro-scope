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

        Absent or partial rows for the requested ``(model='rv_har', symbol)``
        combination always yield ``cleared=False`` with an explicit reason string
        — never an exception.
    """
    required_metrics = {
        "oos_r2",
        "qlike_skill_ratio",
        "folds_passed",
        "n_folds",
        "n_obs",
    }

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
    # Check for missing metrics before computing the verdict.
    # -----------------------------------------------------------------------
    missing = required_metrics - set(metric_map.keys())
    if missing:
        return SkillVerdict(
            cleared=False,
            reasons=[
                f"missing metric(s) for model='rv_har', symbol='{symbol}': "
                + ", ".join(sorted(missing))
            ],
            oos_r2=metric_map.get("oos_r2"),
            qlike_skill_ratio=metric_map.get("qlike_skill_ratio"),
            folds_passed=int(metric_map["folds_passed"]) if "folds_passed" in metric_map else None,
            n_folds=int(metric_map["n_folds"]) if "n_folds" in metric_map else None,
            n_obs=int(metric_map["n_obs"]) if "n_obs" in metric_map else None,
        )

    # -----------------------------------------------------------------------
    # Extract typed values.
    # -----------------------------------------------------------------------
    oos_r2: float = float(metric_map["oos_r2"])
    qlike_skill_ratio: float = float(metric_map["qlike_skill_ratio"])
    folds_passed: int = int(metric_map["folds_passed"])
    n_folds: int = int(metric_map["n_folds"])
    n_obs: int = int(metric_map["n_obs"])

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
