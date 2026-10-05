"""Lazy creation of the module-default EventWorker in escape.stream.

These tests only construct objects -- nothing here registers a channel,
starts the event loop or touches the dispatcher.
"""
import threading

import pytest

import escape.stream.escape_stream as esn
from escape.stream import (
    EventWorker,
    LocalEventHandler,
    Stream,
    StreamSession,
    from_getter,
    get_default_eventworker,
    lab_time,
    pulse_id,
)


@pytest.fixture(autouse=True)
def _clean_default(monkeypatch):
    monkeypatch.delattr(esn, "eventworker", raising=False)


def _default():
    return esn._peek_default_eventworker()


@pytest.mark.parametrize("make", [lambda: Stream("x"), lambda: Stream.from_dispatcher("x")])
def test_stream_without_worker_gets_default(make):
    s = make()
    ew = s._source.eventWorker
    assert isinstance(ew, EventWorker)
    assert ew is _default()
    assert ew.loopThread is None  # created offline, nothing started


def test_streams_share_one_default():
    a, b = Stream("x"), Stream.from_dispatcher("y")
    assert a._source.eventWorker is b._source.eventWorker


def test_explicit_worker_wins_and_leaves_default_unset():
    ew = EventWorker(LocalEventHandler(), make_default=False)
    s = Stream("x", ew)
    assert s._source.eventWorker is ew
    assert _default() is None


def test_existing_default_is_returned_not_replaced():
    ew = EventWorker(make_default=True)
    assert get_default_eventworker() is ew
    assert Stream("x")._source.eventWorker is ew


def test_concurrent_first_calls_yield_one_worker():
    n = 16
    barrier = threading.Barrier(n)
    results = []

    def worker():
        barrier.wait()
        results.append(get_default_eventworker())

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == n
    assert len({id(r) for r in results}) == 1


def test_pulse_id_and_lab_time_without_prior_worker():
    p, t = pulse_id(), lab_time()
    assert p._source.eventWorker is _default()
    assert t._source.eventWorker is _default()


def test_from_getter_uses_and_shares_default():
    g = from_getter(lambda: 1)
    assert isinstance(g._source.eventWorker, EventWorker)
    combined = g + Stream("x")
    assert combined._source.eventWorker is g._source.eventWorker is _default()


@pytest.mark.parametrize("make_default", [True, False])
def test_session_builds_own_worker_without_creating_default(make_default):
    sess = StreamSession("localhost:9999", make_default=make_default)
    ew = sess._ew
    assert ew._eventHandler.__class__.__name__ in ("DataHubLocalEventHandler", "LocalEventHandler")
    # Either the session's own worker became the default, or none exists --
    # never a separate default-handler worker created behind its back.
    assert _default() is (ew if make_default else None)


def test_regression_is_accumulating_without_prior_worker():
    assert Stream.from_dispatcher("x")._is_accumulating() is False
