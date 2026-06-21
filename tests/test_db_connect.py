"""connect() routes to a local file by default, to md: only when MotherDuck is configured,
and always lets an explicit path win (so a forced-local CI run can never escalate to prod)."""

import mmi.utils.db as dbmod


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, database, read_only=False, **kw):
        self.calls.append((database, read_only))
        return object()


def _spy(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(dbmod.duckdb, "connect", spy)
    return spy


def test_defaults_to_local_file(monkeypatch, tmp_path):
    spy = _spy(monkeypatch)
    monkeypatch.setattr(dbmod.settings, "motherduck_database", "")
    monkeypatch.setattr(dbmod.settings, "motherduck_token", "")
    monkeypatch.setattr(dbmod.settings, "duckdb_path", tmp_path / "local.duckdb")
    dbmod.connect(read_only=True)
    assert spy.calls == [(str(tmp_path / "local.duckdb"), True)]


def test_routes_to_motherduck_when_configured(monkeypatch):
    spy = _spy(monkeypatch)
    monkeypatch.setenv("motherduck_token", "")  # connect() sets this; let monkeypatch clean it up
    monkeypatch.setattr(dbmod.settings, "motherduck_database", "mmi")
    monkeypatch.setattr(dbmod.settings, "motherduck_token", "tok")
    dbmod.connect(read_only=True)
    assert spy.calls == [("md:mmi", True)]  # read_only is threaded through to MotherDuck


def test_explicit_path_overrides_motherduck(monkeypatch, tmp_path):
    spy = _spy(monkeypatch)
    monkeypatch.setattr(dbmod.settings, "motherduck_database", "mmi")
    monkeypatch.setattr(dbmod.settings, "motherduck_token", "tok")
    local = tmp_path / "forced.duckdb"
    dbmod.connect(path=local)
    assert spy.calls == [(str(local), False)]  # explicit local path wins over MotherDuck
