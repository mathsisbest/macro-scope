"""YahooChartExtractor stores adjusted close (total return) and fails loud on no usable data."""

import httpx
import pytest
import respx

from mmi.ingestion.loader import DuckDBLoader
from mmi.ingestion.yahoo import YahooChartExtractor

_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/SPY"


@respx.mock
def test_yahoo_uses_adjusted_close(con, monkeypatch):
    monkeypatch.setattr("mmi.ingestion.yahoo.load_assets", lambda: {"equities": ["SPY"]})
    respx.get(_CHART).mock(
        return_value=httpx.Response(
            200,
            json={
                "chart": {
                    "result": [
                        {
                            "meta": {},
                            "timestamp": [1700000000, 1700086400],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [10, 11],
                                        "high": [10, 11],
                                        "low": [10, 11],
                                        "close": [10.0, 11.0],
                                        "volume": [100, 200],
                                    }
                                ],
                                "adjclose": [{"adjclose": [9.5, 10.5]}],
                            },
                        }
                    ]
                }
            },
        )
    )
    df = YahooChartExtractor(DuckDBLoader(con)).fetch()
    assert {"symbol", "asset_class", "date", "close", "source"}.issubset(df.columns)
    assert df["close"].tolist() == [9.5, 10.5]  # adjusted close used, not raw close [10, 11]
    assert (df["asset_class"] == "equities").all()


@respx.mock
def test_yahoo_fails_loud_when_no_usable_data(con, monkeypatch):
    monkeypatch.setattr("mmi.ingestion.yahoo.load_assets", lambda: {"equities": ["SPY"]})
    respx.get(_CHART).mock(return_value=httpx.Response(200, json={"chart": {"result": None}}))
    with pytest.raises(ValueError):  # not a silent empty load
        YahooChartExtractor(DuckDBLoader(con)).fetch()
