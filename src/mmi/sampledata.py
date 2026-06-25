"""Deterministic synthetic data so the project runs end-to-end with no API keys.

Generates the *same columns* the real extractors produce, written to the ``raw`` schema.
This lets a reviewer ``make demo`` and see the whole stack immediately, while live
ingestion (``make ingest``) populates the identical tables when keys are present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmi.ingestion.loader import DuckDBLoader
from mmi.settings import load_assets
from mmi.utils.logging import get_logger

log = get_logger("sampledata")
_RNG = np.random.default_rng(42)


def _walk(start: float, n: int, vol: float) -> np.ndarray:
    """Geometric-ish random walk for plausible prices."""
    steps = _RNG.normal(0, vol, n)
    return start * np.exp(np.cumsum(steps))


def _assets() -> pd.DataFrame:
    assets = load_assets()
    today = pd.Timestamp.now(tz="UTC").normalize()
    days = 400
    dates = pd.bdate_range(end=today, periods=days)
    starts = {
        "SPY": 520,
        "QQQ": 440,
        "VEA": 50,
        "TLT": 90,
        "TIP": 108,
        "GLD": 215,
        "EURUSD": 1.08,
        "GBPUSD": 1.27,
        "BTC": 60000,  # daily BTC (Yahoo crypto_daily) — the only crypto path
    }
    rows = []
    # All price-bearing classes (incl. bonds/commodities) so the 60/40 benchmark has its legs.
    for kind in ("equities", "bonds", "commodities", "fx"):
        for sym in assets.get(kind, []):
            close = _walk(starts.get(sym, 100), len(dates), 0.012)
            for d, c in zip(dates, close, strict=False):
                o = c * (1 + _RNG.normal(0, 0.003))
                rows.append(
                    {
                        "symbol": sym,
                        "asset_class": kind,
                        "date": d,
                        "open": round(float(o), 4),
                        "high": round(float(max(o, c)) * 1.005, 4),
                        "low": round(float(min(o, c)) * 0.995, 4),
                        "close": round(float(c), 4),
                        "volume": int(abs(_RNG.normal(5e6, 1e6))) if kind == "equities" else 0,
                        "source": "sample",
                    }
                )
    # Daily crypto (BTC) — the ONLY crypto path (Yahoo BTC-USD). Deliberately starts LATER than the
    # other assets (trailing window) to mimic BTC's later inception, so the multi-window backtest's
    # staggered-start handling is genuinely exercised by `make ci`. The span (330 of 400 days)
    # stays comfortably above the 252-day lookback so the inc/ex 2015 windows are backtestable in
    # CI. Higher vol than the others but held under the +/-50% bound (assert_returns_within_bounds).
    # Carries a NONZERO volume like real Yahoo BTC-USD bars, so it survives stg_asset_prices'
    # phantom-volume-0-bar filter and lands in dim_asset / fct_asset_daily (the unified Asset
    # selector + BTC KPI + inc_btc window all need BTC present in the dbt marts).
    btc_dates = dates[-330:]
    for sym in assets.get("crypto_daily", []):
        stored = sym.split("-")[0]
        close = _walk(starts.get(stored, 60000), len(btc_dates), 0.035)
        for d, c in zip(btc_dates, close, strict=False):
            o = c * (1 + _RNG.normal(0, 0.01))
            rows.append(
                {
                    "symbol": stored,
                    "asset_class": "crypto",
                    "date": d,
                    "open": round(float(o), 2),
                    "high": round(float(max(o, c)) * 1.01, 2),
                    "low": round(float(min(o, c)) * 0.99, 2),
                    "close": round(float(c), 2),
                    "volume": int(abs(_RNG.normal(2e10, 4e9))),
                    "source": "sample",
                }
            )
    return pd.DataFrame(rows)


def _macro() -> pd.DataFrame:
    series = load_assets()["macro"]
    months = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=60, freq="MS")
    base = {"CPIAUCSL": 300.0, "UNRATE": 4.0, "DGS10": 4.3, "DGS2": 4.7, "FEDFUNDS": 5.3}
    drift = {"CPIAUCSL": 0.4, "UNRATE": 0.01, "DGS10": 0.0, "DGS2": 0.0, "FEDFUNDS": 0.0}
    rows = []
    for s in series:
        sid = s["id"]
        val = base.get(sid, 1.0)
        for d in months:
            val = val + drift.get(sid, 0.0) + _RNG.normal(0, abs(val) * 0.01 + 0.02)
            rows.append(
                {"series_id": sid, "date": d, "value": round(float(val), 3), "source": "sample"}
            )
    return pd.DataFrame(rows)


def _worldbank() -> pd.DataFrame:
    inds = load_assets()["worldbank"]
    years = [str(y) for y in range(2014, 2025)]
    countries = ["USA", "GBR", "WLD"]
    rows = []
    for ind in inds:
        for c in countries:
            for y in years:
                rows.append(
                    {
                        "indicator_id": ind["id"],
                        "country": c,
                        "date": y,
                        "value": round(float(_RNG.normal(2.5, 1.2)), 3),
                        "source": "sample",
                    }
                )
    return pd.DataFrame(rows)


def seed(con) -> dict[str, int]:
    """Populate raw.* tables with deterministic synthetic data. Returns row counts."""
    loader = DuckDBLoader(con)
    counts = {
        "raw.asset_prices": loader.upsert("raw.asset_prices", _assets(), ["symbol", "date"]),
        "raw.macro_series": loader.upsert("raw.macro_series", _macro(), ["series_id", "date"]),
        "raw.worldbank": loader.upsert(
            "raw.worldbank", _worldbank(), ["indicator_id", "country", "date"]
        ),
    }
    log.info("seeded sample data: %s", counts)
    return counts
