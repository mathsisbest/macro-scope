"""redact() scrubs URL-query API keys (FRED/Gemini) and Bearer tokens; leaves the rest intact."""

from mmi.utils.redact import redact


def test_redacts_fred_api_key():
    s = (
        "Client error '400 Bad Request' for url "
        "'https://api.stlouisfed.org/fred/series/observations?series_id=DGS10"
        "&api_key=SECRETKEY123&file_type=json'"
    )
    out = redact(s)
    assert "SECRETKEY123" not in out
    assert "api_key=***" in out
    assert "series_id=DGS10" in out  # non-secret params are preserved


def test_redacts_gemini_key_param():
    s = "url 'https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key=AIzaSECRET'"
    out = redact(s)
    assert "AIzaSECRET" not in out
    assert "key=***" in out


def test_redacts_bearer_token():
    assert redact("Authorization: Bearer gsk_supersecret") == "Authorization: Bearer ***"


def test_noop_when_no_secret():
    s = "Connection error to https://stooq.com/q/d/l/?s=spy.us&i=d"
    assert redact(s) == s
