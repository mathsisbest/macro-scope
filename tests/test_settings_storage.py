"""storage_label() must never render the MotherDuck token; use_motherduck needs both vars."""

from mmi.settings import Settings

# Both casings: pydantic-settings matches env vars case-insensitively, and connect() sets the
# lowercase `motherduck_token` — clear both so these tests are order-independent.
_MD_VARS = ("MMI_MOTHERDUCK_DATABASE", "MOTHERDUCK_TOKEN", "motherduck_token")


def _clear(monkeypatch):
    for v in _MD_VARS:
        monkeypatch.delenv(v, raising=False)


def test_local_storage_label_and_use_motherduck(monkeypatch):
    _clear(monkeypatch)
    s = Settings(_env_file=None)
    assert s.use_motherduck is False
    assert s.storage_label().startswith("DuckDB")


def test_storage_label_never_leaks_token(monkeypatch):
    _clear(monkeypatch)
    secret = "md_secret_DO_NOT_LEAK_42"
    monkeypatch.setenv("MMI_MOTHERDUCK_DATABASE", "mmi")
    monkeypatch.setenv("MOTHERDUCK_TOKEN", secret)
    s = Settings(_env_file=None)
    assert s.use_motherduck is True
    label = s.storage_label()
    assert secret not in label  # security invariant: the token must never reach the UI
    assert "MotherDuck" in label and "mmi" in label


def test_use_motherduck_requires_both_vars(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("MMI_MOTHERDUCK_DATABASE", "mmi")
    assert Settings(_env_file=None).use_motherduck is False
