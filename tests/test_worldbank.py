"""WorldBankExtractor: fetch returns indicator rows; probe validates connectivity."""

import httpx
import pytest
import respx

from mmi.ingestion.loader import DuckDBLoader
from mmi.ingestion.worldbank import WorldBankExtractor

_PROBE_URL = "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD"
_FETCH_URL = "https://api.worldbank.org/v2/country/USA;GBR;WLD/indicator/NY.GDP.MKTP.CD"
_INDICATOR = {"id": "NY.GDP.MKTP.CD", "label": "GDP (current US$)"}


@respx.mock
def test_probe_hits_api_and_succeeds(con):
    respx.get(_PROBE_URL).mock(return_value=httpx.Response(200, json=[{}, []]))
    WorldBankExtractor(DuckDBLoader(con)).probe()
    assert respx.calls.last.request.url.path.endswith("NY.GDP.MKTP.CD")


@respx.mock
def test_probe_raises_on_non_200(con):
    respx.get(_PROBE_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        WorldBankExtractor(DuckDBLoader(con)).probe()


@respx.mock
def test_fetch_returns_rows(con, monkeypatch):
    monkeypatch.setattr("mmi.ingestion.worldbank.load_assets", lambda: {"worldbank": [_INDICATOR]})
    body = [
        {},
        [
            {"countryiso3code": "USA", "date": "2022", "value": 25e12},
            {"countryiso3code": "GBR", "date": "2022", "value": 3e12},
        ],
    ]
    respx.get(_FETCH_URL).mock(return_value=httpx.Response(200, json=body))
    df = WorldBankExtractor(DuckDBLoader(con)).fetch()
    assert len(df) == 2
    assert list(df.columns) == ["indicator_id", "country", "date", "value", "source"]
    assert df["indicator_id"].iloc[0] == "NY.GDP.MKTP.CD"
    assert df["country"].iloc[0] == "USA"
    assert df["source"].iloc[0] == "worldbank"


@respx.mock
def test_fetch_returns_empty_when_payload_is_invalid(con, monkeypatch):
    monkeypatch.setattr("mmi.ingestion.worldbank.load_assets", lambda: {"worldbank": [_INDICATOR]})
    respx.get(_FETCH_URL).mock(return_value=httpx.Response(200, json={}))
    df = WorldBankExtractor(DuckDBLoader(con)).fetch()
    assert df.empty


@respx.mock
def test_fetch_skips_null_values(con, monkeypatch):
    monkeypatch.setattr("mmi.ingestion.worldbank.load_assets", lambda: {"worldbank": [_INDICATOR]})
    body = [
        {},
        [
            {"countryiso3code": "USA", "date": "2022", "value": 25e12},
            {"countryiso3code": "GBR", "date": "2022", "value": None},
        ],
    ]
    respx.get(_FETCH_URL).mock(return_value=httpx.Response(200, json=body))
    df = WorldBankExtractor(DuckDBLoader(con)).fetch()
    assert len(df) == 2
    assert df["value"].isna().iloc[1]
