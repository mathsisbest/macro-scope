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
    required = False  # gracefully skip on weekends/holidays when no new data
    probe_url = _URL.format(symbol="SPY")
    watermark_col = "date"

    def probe(self) -> None:
        """Probe Yahoo chart endpoint for SPY with a minimal 1-day range."""
        get_json(self.probe_url, headers=_HEADERS, params={"range": "1d", "interval": "1d"})

    def fetch(self, start_after: str | None = None) -> pd.DataFrame:
        assets = load_assets()
        frames: list[pd.DataFrame] = []
        for kind in _KINDS:
            for symbol in assets.get(kind, []):
                try:
                    df = self._fetch_one(symbol, kind, start_after=start_after)
                except Exception as exc:  # noqa: BLE001 - per-symbol best-effort; total failure caught below
                    self.log.warning("yahoo: %s failed: %s", symbol, exc)
                    continue
                if df.empty:
                    self.log.warning("yahoo: no data for %s", symbol)
                else:
                    frames.append(df)
        if not frames:
            if start_after:
                self.log.info("yahoo: no new data since %s", start_after)
                return pd.DataFrame()
            # Fail LOUD: a non-JSON / empty response on full load must not masquerade as a successful empty load.
            raise ValueError("Yahoo returned no usable data for any symbol")
        return pd.concat(frames, ignore_index=True)


    def _fetch_one(self, symbol: str, kind: str, start_after: str | None = None) -> pd.DataFrame:
        yahoo_symbol = f"{symbol}=X" if kind == "fx" else symbol
        # crypto_daily holds Yahoo crypto tickers (e.g. BTC-USD); store them as a clean symbol
        # ('BTC') under asset_class 'crypto' so they join the daily price path. This is the only
        # crypto source — BTC daily via Yahoo.
        stored_symbol = symbol.split("-")[0] if kind == "crypto_daily" else symbol
        asset_class = "crypto" if kind == "crypto_daily" else kind
        # Incremental: if start_after is set, fetch only data after that date.
        # Yahoo's period1 is a Unix timestamp; convert the date string to epoch.
        if start_after:
            wm_date = pd.Timestamp(start_after).normalize()
            # Add 1 day so we don't re-fetch the last loaded row
            period1 = int((wm_date + pd.Timedelta(days=1)).timestamp())
        else:
            period1 = 0
        payload = get_json(
            _URL.format(symbol=yahoo_symbol),
            params={"period1": period1, "period2": 9999999999, "interval": "1d"},
            headers=_HEADERS,
        )
        result = (payload.get("chart") or {}).get("result")
        if not result:
            raise ValueError(f"no chart result for {yahoo_symbol}")
        node = result[0]
        timestamps = node.get("timestamp") or []
        indicators = node.get("indicators") or {}
        quote_list = indicators.get("quote") or []
        quote = quote_list[0] if (quote_list and isinstance(quote_list[0], dict)) else {}
        adjclose_list = indicators.get("adjclose") or []
        adjnode = adjclose_list[0] if (adjclose_list and isinstance(adjclose_list[0], dict)) else {}
        adjclose = adjnode.get("adjclose") if isinstance(adjnode, dict) else None


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
        df = df[df["close"].notna()].reset_index(drop=True)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        from mmi.ingestion.models import YahooPriceRow

        df = super().validate(df)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return self.validate_pydantic(df, YahooPriceRow)
