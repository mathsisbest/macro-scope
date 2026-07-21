"""Volatility-regime labelling (Low / Medium / High) per asset."""

from __future__ import annotations

import pandas as pd

from mmi.utils.logging import get_logger

log = get_logger("ml.regime")
_LABELS = ["Low", "Medium", "High"]


def label_regimes(con) -> pd.DataFrame:
    """Classify each (symbol, date) into a vol regime via per-symbol terciles."""
    df = con.execute(
        "select symbol, date, vol_20d from marts.fct_asset_daily where vol_20d is not null"
    ).df()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "date", "vol_20d", "regime"])

    # rank(method='first') guarantees unique edges so qcut never fails on ties.
    # transform keeps the result aligned to the original rows (no apply-on-groups warning).
    out = df.copy()
    # Each symbol needs at least 3 rows for pd.qcut(..., 3) to succeed.
    valid_symbols = out.groupby("symbol").size()
    valid_symbols = valid_symbols[valid_symbols >= 3].index
    out = out[out["symbol"].isin(valid_symbols)]

    if out.empty:
        log.warning("no symbol has enough rows for regime labelling (need >= 3 per symbol)")
        return df[["symbol", "date", "vol_20d"]].copy().assign(regime="Medium")

    out["regime"] = (
        out.groupby("symbol")["vol_20d"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=_LABELS))
        .astype(str)
    )
    log.info("labelled %d regime rows", len(out))
    return out[["symbol", "date", "vol_20d", "regime"]]
