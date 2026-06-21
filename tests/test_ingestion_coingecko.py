import httpx
import respx

from mmi.ingestion.coingecko import CoinGeckoExtractor
from mmi.ingestion.loader import DuckDBLoader


@respx.mock
def test_coingecko_fetch_parses_payload(con):
    respx.get("https://api.coingecko.com/api/v3/simple/price").mock(
        return_value=httpx.Response(
            200,
            json={
                "bitcoin": {
                    "usd": 65000.0,
                    "usd_market_cap": 1.2e12,
                    "usd_24h_vol": 3.0e10,
                    "last_updated_at": 1700000000,
                }
            },
        )
    )
    df = CoinGeckoExtractor(DuckDBLoader(con)).fetch()
    assert {"symbol", "ts", "price_usd"}.issubset(df.columns)
    assert df.loc[df["symbol"] == "bitcoin", "price_usd"].iloc[0] == 65000.0
