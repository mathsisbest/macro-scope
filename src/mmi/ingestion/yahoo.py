"""Yahoo Finance v8 chart extractor — daily ADJUSTED-close history (total return).

Replaces the Stooq CSV path, whose free endpoint now returns a JavaScript browser-challenge
(not CSV) and so silently yielded zero rows. This uses the free, key-less v8 ``/chart`` JSON
endpoint and stores the *adjusted* close as ``close`` so downstream returns are total-return
(dividends reinvested) — which matters materially for bond/dividend ETFs (TLT/TIP/VEA).
"""

from __future__ import annotations

import pandas as pd

from mmi.ingestion.base import Extractor
from mmi.settings import load_assets
from mmi.utils.http import get_json

_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo rejects the default httpx user-agent; a browser UA returns JSON.
_HEADERS = {"User-Agent": "Mozilla/5.0"}
# config/assets.yml keys this extractor pulls into raw.asset_prices (asset_class = the key,
# except crypto_daily which folds into asset_class 'crypto' — see _fetch_one).
_KINDS = ("equities", "bonds", "commodities", "fx", "crypto_daily")


class YahooChartExtractor(Extractor):
    source = "yahoo"
    table = "raw.asset_prices"
    keys = ["symbol", "date"]
    required_columns = ["symbol", "date", "close"]
    required = True  # core price history — a *total* failure must fail the run, not pass silently
    probe_url = _URL.format(symbol="SPY")

    def probe(self) -> None:
        """Probe Yahoo chart endpoint for SPY with a minimal 1-day range."""
        get_json(self.probe_url, headers=_HEADERS, params={"range": "1d", "interval": "1d"})

    def fetch(self) -> pd.DataFrame:
        assets = load_assets()
        frames: list[pd.DataFrame] = []
        for kind in _KINDS:
            for symbol in assets.get(kind, []):
                try:
                    df = self._fetch_one(symbol, kind)
                except Exception as exc:  # noqa: BLE001 - per-symbol best-effort; total failure caught below
                    self.log.warning("yahoo: %s failed: %s", symbol, exc)
                    continue
                if df.empty:
                    self.log.warning("yahoo: no data for %s", symbol)
                else:
                    frames.append(df)
        if not frames:
            # Fail LOUD: a non-JSON / empty response must not masquerade as a successful empty load.
            raise ValueError("Yahoo returned no usable data for any symbol")
        return pd.concat(frames, ignore_index=True)

    def _fetch_one(self, symbol: str, kind: str) -> pd.DataFrame:
        yahoo_symbol = f"{symbol}=X" if kind == "fx" else symbol
        # crypto_daily holds Yahoo crypto tickers (e.g. BTC-USD); store them as a clean symbol
        # ('BTC') under asset_class 'crypto' so they join the daily price path. This is the only
        # crypto source — BTC daily via Yahoo.
        stored_symbol = symbol.split("-")[0] if kind == "crypto_daily" else symbol
        asset_class = "crypto" if kind == "crypto_daily" else kind
        # Use explicit period1/period2 (not range=max, which silently coarsens to monthly bars)
        # to get the full *daily* history. period2 far in the future -> Yahoo clamps to today.
        payload = get_json(
            _URL.format(symbol=yahoo_symbol),
            params={"period1": 0, "period2": 9999999999, "interval": "1d"},
            headers=_HEADERS,
        )
        result = (payload.get("chart") or {}).get("result")
        if not result:
            raise ValueError(f"no chart result for {yahoo_symbol}")
        node = result[0]
        timestamps = node.get("timestamp") or []
        quote = (node.get("indicators", {}).get("quote") or [{}])[0]
        adjclose = (node.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
        # Adjusted close = total return; fall back to raw close (FX has no adjclose).
        closes = adjclose if adjclose is not None else quote.get("close")
        if not timestamps or not closes:
            raise ValueError(f"no price series for {yahoo_symbol}")
        df = pd.DataFrame(
            {
                "symbol": stored_symbol,
                "asset_class": asset_class,
                "date": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": closes,
                "volume": quote.get("volume"),
                "source": self.source,
            }
        )
        return df[df["close"].notna()].reset_index(drop=True)
