import pytest


@pytest.fixture(autouse=True)
def _no_clock_query(monkeypatch):
    """The dispatcher context asks the dispatcher for the clock offset; tests
    never touch the network."""
    try:
        from escape.stream import es_wrappers_datahub as dh
    except ImportError:
        return
    monkeypatch.setattr(dh, "_clock_offset", lambda *a, **k: 0.0)
