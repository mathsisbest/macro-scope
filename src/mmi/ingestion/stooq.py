"""Stooq extractor for daily OHLCV (equities, ETFs, FX) — no API key required."""

from __future__ import annotations

import io

import pandas as pd

from mmi.ingestion.base import Extractor
from mmi.settings import load_assets
from mmi.utils.http import get_text

_URL = "https://stooq.com/q/d/l/"


def _stooq_symbol(symbol: str, kind: str) -> str:
    """Map our symbol to Stooq's convention (US tickers get a .us suffix)."""
    return f"{symbol.lower()}.us" if kind == "equities" else symbol.lower()


class StooqExtractor(Extractor):
    source = "stooq"
    table = "raw.asset_prices"
    keys = ["symbol", "date"]
    required_columns = ["symbol", "date", "close"]
    # Stooq uses unofficial CSV endpoints — best-effort, so a transient failure must not fail
    # the ingest step. NOTE: it is still the sole producer of raw.asset_prices (which dbt
    # requires), so on a *fresh* DB a Stooq failure cascades into dbt build — see follow-up.
    required = False

    def fetch(self) -> pd.DataFrame:
        assets = load_assets()
        frames = []
        for kind in ("equities", "fx"):
            for symbol in assets.get(kind, []):
                frames.append(self._fetch_one(symbol, kind))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_one(self, symbol: str, kind: str) -> pd.DataFrame:
        csv = get_text(_URL, params={"s": _stooq_symbol(symbol, kind), "i": "d"})
        df = pd.read_csv(io.StringIO(csv))
        if df.empty or "Close" not in df.columns:
            self.log.warning("no data for %s", symbol)
            return pd.DataFrame()
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df["symbol"] = symbol
        df["asset_class"] = kind
        df["source"] = self.source
        return df[
            ["symbol", "asset_class", "date", "open", "high", "low", "close", "volume", "source"]
        ]
