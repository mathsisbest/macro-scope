"""get_json / get_text retry on transport errors and HTTP 429s; pass through on 2xx."""

import httpx
import pytest
import respx

from mmi.utils.http import get_json, get_text

_JSON_URL = "https://api.example.com/data.json"
_TEXT_URL = "https://api.example.com/data.csv"


@respx.mock
def test_get_json_returns_parsed_json():
    respx.get(_JSON_URL).mock(return_value=httpx.Response(200, json={"key": "val"}))
    assert get_json(_JSON_URL) == {"key": "val"}


@respx.mock
def test_get_json_passes_params():
    route = respx.get(_JSON_URL).mock(return_value=httpx.Response(200, json=[]))
    get_json(_JSON_URL, params={"foo": "bar"})
    assert route.calls.last.request.url.query == b"foo=bar"


@respx.mock
def test_get_json_raises_on_4xx():
    respx.get(_JSON_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        get_json(_JSON_URL)


@respx.mock
def test_get_json_retries_on_429():
    respx.get(_JSON_URL).mock(
        return_value=httpx.Response(429),
    )
    with pytest.raises(httpx.HTTPStatusError):
        get_json(_JSON_URL)
    assert len(respx.calls) == 4  # initial + 3 retries


@respx.mock
def test_get_json_retries_on_transport_error():
    respx.get(_JSON_URL).mock(side_effect=httpx.TransportError("connection refused"))
    with pytest.raises(httpx.TransportError):
        get_json(_JSON_URL)
    assert len(respx.calls) == 4


@respx.mock
def test_get_text_returns_body():
    respx.get(_TEXT_URL).mock(return_value=httpx.Response(200, text="a,b,c\n1,2,3"))
    assert get_text(_TEXT_URL) == "a,b,c\n1,2,3"


@respx.mock
def test_get_text_follows_redirects():
    respx.get(_TEXT_URL).mock(return_value=httpx.Response(302, headers={"Location": "/final"}))
    respx.get("https://api.example.com/final").mock(
        return_value=httpx.Response(200, text="final")
    )
    assert get_text(_TEXT_URL) == "final"
