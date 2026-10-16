"""Recovering from a dead datahub receive thread, the bsread receive timeout
setting, and EventWorker.reset() / reset_default_eventworker().

Nothing here touches the dispatcher: stubs for the unit tests, a local
TestStream sender for the integration test.
"""
import collections
import os
import signal
import socket
import threading
import time

import pytest

import escape.stream.escape_stream as es
from escape.stream import es_wrappers_datahub as dh
from escape.stream import reset_default_eventworker


def wait_until(cond, timeout=5.0):
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            return False
        time.sleep(0.01)
    return True


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class StubStream:
    """datahub-stream-like: a message buffer and an optional thread."""

    def __init__(self, messages=(), thread_alive=True, has_thread=True, running=True):
        self.buf = collections.deque(messages)
        self._release = threading.Event()
        if has_thread:
            self.processing_thread = threading.Thread(target=self._release.wait, daemon=True)
            self.processing_thread.start()
            if not thread_alive:
                self._release.set()
                self.processing_thread.join()
        else:
            self.processing_thread = None
        self._running = running

    def receive(self, timeout=None):
        return self.buf.popleft() if self.buf else None

    def is_running(self):
        return self._running

    def get_run_exception(self):
        return Exception("Received None message.")

    def abort(self):
        self._release.set()

    def join(self, timeout=None):
        if self.processing_thread is not None:
            self.processing_thread.join(timeout)

    def close(self):
        self.abort()


class StubHandler:
    receive_timeout = 0.01
    bsread_receive_timeout_ms = None

    def __init__(self, stream):
        self.stream = stream


def ctx_for(stream):
    return dh.DataHubEventSourceContext(StubHandler(stream))


MSG = (1001, 0, {"a": 1.0})


# ---------------------------------------------------------------------------
# Change 1: get_event() raises once the receive thread is dead
# ---------------------------------------------------------------------------

def test_dead_thread_empty_buffer_raises():
    with pytest.raises(RuntimeError, match="Received None message"):
        ctx_for(StubStream(thread_alive=False)).get_event()


def test_dead_thread_delivers_buffered_message_first():
    ctx = ctx_for(StubStream([MSG], thread_alive=False))
    assert ctx.get_event().getEventId() == 1001
    with pytest.raises(RuntimeError):
        ctx.get_event()


def test_live_thread_empty_buffer_is_null_event():
    assert ctx_for(StubStream()).get_event().getEventId() is None


def test_no_thread_is_not_dead():
    assert ctx_for(StubStream(has_thread=False)).get_event().getEventId() is None


def test_fresh_stream_not_yet_running_is_not_dead():
    # datahub sets `running` only inside the new thread
    assert ctx_for(StubStream(running=False)).get_event().getEventId() is None


# ---------------------------------------------------------------------------
# Change 1, MultiSourceContext: a dead sub-stream is reconnected
# ---------------------------------------------------------------------------

class DyingStream:
    """Delivers *n* messages from its own thread, then the thread ends."""

    def __init__(self, channel, n, pids):
        self.channel, self.buf = channel, collections.deque()
        self.processing_thread = threading.Thread(target=self._run, args=(n, pids), daemon=True)
        self.processing_thread.start()

    def _run(self, n, pids):
        for _ in range(n):
            self.buf.append((next(pids), 0, {self.channel: 1.0}))
            time.sleep(0.002)

    def receive(self, timeout=None):
        if self.buf:
            return self.buf.popleft()
        time.sleep(0.002)
        return None

    def get_run_exception(self):
        return Exception("Received None message.")

    def abort(self):
        pass

    def join(self, timeout=None):
        self.processing_thread.join(timeout)

    def close(self):
        pass


class DyingHandler:
    receive_timeout = 0.01
    bsread_receive_timeout_ms = None

    def __init__(self, channel, n):
        self.channel, self.n, self.stream, self.connections = channel, n, None, 0
        self.pids = iter(range(10**9))

    def register_source(self, s):
        pass

    def remove_source(self, s):
        pass

    def context_manager(self):
        self.connections += 1
        self.stream = DyingStream(self.channel, self.n, self.pids)
        return dh.DataHubEventSourceContext(self)


def test_multisource_reconnects_dead_substream():
    dying = DyingHandler("a", n=20)
    steady = DyingHandler("b", n=10**6)
    handler = dh.MultiSourceEventHandler(dying, steady, timeout_pulses=1)
    with handler.context_manager() as ctx:
        seen_a = []

        def collect():
            # drain the queue (it drops events when full), bounded per poll
            for _ in range(1000):
                ev = ctx.get_event()
                if ev.getEventId() is None:
                    break
                if ev.getFromSource("a") is not None:
                    seen_a.append(ev.getEventId())
            return len(seen_a) > 30 and dying.connections >= 2

        # 20 messages per connection: more than 30 needs a reconnect.
        assert wait_until(collect, timeout=10)
        assert all(t.is_alive() for t in ctx._threads)


# ---------------------------------------------------------------------------
# Change 2: bsread receive timeout as its own setting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,args", [
    (dh.DataHubEventHandler, dict(backend="bsread")),
    (dh.DataHubLocalEventHandler, dict(host="localhost", port=1)),
])
def test_clone_keeps_both_timeouts(cls, args):
    h = cls(receive_timeout=0.5, bsread_receive_timeout_ms=10000, queue_size=50, **args)
    c = h.clone()
    assert (c.receive_timeout, c.bsread_receive_timeout_ms) == (0.5, 10000)
    assert c._stream_kwargs == {"queue_size": 50}


@pytest.mark.parametrize("ms", [10000, None])
def test_context_manager_passes_bsread_timeout(monkeypatch, ms):
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(dh, "_TolerantBsreadStream", Recorder)
    monkeypatch.setattr(dh, "BsreadStream", Recorder)
    h = dh.DataHubEventHandler(backend="bsread", bsread_receive_timeout_ms=ms)
    h.register_source("X")
    h.context_manager()
    assert seen.get("receive_timeout") == ms if ms else "receive_timeout" not in seen
    seen.clear()
    local = dh.DataHubLocalEventHandler(bsread_receive_timeout_ms=ms)
    local.context_manager()
    assert seen.get("receive_timeout") == ms if ms else "receive_timeout" not in seen


# ---------------------------------------------------------------------------
# Change 3: EventWorker.reset() / reset_default_eventworker()
# ---------------------------------------------------------------------------

class CountingCtx:
    """Fake connection; a "dead" one only yields NullEvents (old behaviour)."""

    def __init__(self, handler):
        self.h = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_event(self):
        if self.h.block is not None:
            self.h.block.wait()  # simulate a get_event() stuck in a blocking call
        time.sleep(0.002)
        if self.h.dead:
            return dh.NullEvent()
        self.h.pid += 1
        return dh.DataHubEvent(self.h.pid, 0, {c: 1.0 for c in self.h.source_ids})


class CountingHandler:
    def __init__(self):
        self.source_ids, self.dead, self.block, self.pid = [], False, None, 0
        self.contexts = 0

    def register_source(self, s):
        if s not in self.source_ids:
            self.source_ids.append(s)

    def remove_source(self, s):
        if s in self.source_ids:
            self.source_ids.remove(s)

    def clone(self):
        return CountingHandler()

    def context_manager(self):
        self.contexts += 1
        self.dead = False  # a new connection works again
        return CountingCtx(self)


@pytest.fixture
def ew():
    w = es.EventWorker(CountingHandler(), make_default=False, restart_mode="break_before_make")
    w.stall_timeout = None  # the old behaviour: nothing notices a dead stream
    yield w
    w.stopEventLoop()


def test_reset_recovers_dead_stream_in_place(ew):
    s = es.Stream("a", ew)
    s.accumulate(True)
    assert wait_until(lambda: len(s) > 10)
    ew._eventHandler.dead = True
    time.sleep(0.2)
    n = len(s)
    time.sleep(0.2)
    assert len(s) == n  # stuck
    callbacks, sources = list(ew.eventCallbacks), list(ew._eventHandler.source_ids)
    ew.reset()
    assert wait_until(lambda: len(s) > n + 10)
    assert s._source.eventWorker is ew
    assert ew.eventCallbacks == callbacks and ew._eventHandler.source_ids == sources
    s.accumulate(False)


def test_reset_twice_and_cancels_pending_restart(ew):
    s = es.Stream("a", ew)
    s.accumulate(True)
    ew._schedule_restart(delay=5.0)
    timer = ew._restart_timer
    ew.reset()
    ew.reset()
    assert ew._restart_timer is None and not timer.is_alive()
    assert ew.loopThread.is_alive()
    n = len(s)
    assert wait_until(lambda: len(s) > n + 10)
    s.accumulate(False)


def test_hard_reset_with_held_lock_and_new_handler(ew):
    s = es.Stream("a", ew)
    s.accumulate(True)
    old_handler = ew._eventHandler
    ew._restart_lock.acquire()  # a hung restart holding it forever
    t0 = time.time()
    ew.reset(hard=True)
    assert time.time() - t0 < 3
    assert ew._eventHandler is not old_handler
    assert ew._eventHandler.source_ids == old_handler.source_ids
    assert ew.loopThread.is_alive()
    n = len(s)
    assert wait_until(lambda: len(s) > n + 10)
    s.accumulate(False)


def test_stuck_loop_thread_never_dispatches_twice(ew):
    seen = []
    ew.eventCallbacks.append(lambda: seen.append(ew.event.getEventId()))
    s = es.Stream("a", ew)
    s.accumulate(True)
    assert wait_until(lambda: len(seen) > 10)
    old_handler = ew._eventHandler
    old_handler.block = threading.Event()  # the running loop gets stuck
    time.sleep(0.1)
    old_thread = ew.loopThread
    ew.reset(hard=True)  # gives up on it after a short join
    assert old_thread.is_alive()
    n = len(seen)
    assert wait_until(lambda: len(seen) > n + 20)
    old_handler.block.set()  # the stuck thread unblocks
    old_thread.join(2)
    assert not old_thread.is_alive()
    time.sleep(0.2)
    # one dispatching loop: every pulse of the new handler exactly once, in order
    new = [p for p in seen[n:]]
    assert new == sorted(set(new))
    s.accumulate(False)


def test_reset_default_without_default_creates_none(monkeypatch):
    monkeypatch.delattr(es, "eventworker", raising=False)
    assert reset_default_eventworker() is None
    assert es._peek_default_eventworker() is None


def test_reset_default_resets_the_registered_worker(monkeypatch):
    monkeypatch.delattr(es, "eventworker", raising=False)
    w = es.EventWorker(CountingHandler(), make_default=True)
    try:
        assert reset_default_eventworker() is w
        assert w.loopThread.is_alive()
    finally:
        w.stopEventLoop()


# ---------------------------------------------------------------------------
# Integration: frozen local sender, recovery without manual action
# ---------------------------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.mark.slow
# datahub's receive thread ending with "Received None message." is the point
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_frozen_sender_recovers_by_itself():
    pytest.importorskip("bsread")
    port = _free_port()
    ts = es.TestStream(port=port, interval=0.01)
    ts.start()
    ew = es.EventWorker(
        dh.DataHubLocalEventHandler(port=port, bsread_receive_timeout_ms=1000),
        make_default=False,
    )
    try:
        s = es.Stream("i0", ew)
        s.accumulate(True)
        assert wait_until(lambda: len(s) > 20, timeout=15)
        os.kill(ts._process.pid, signal.SIGSTOP)
        time.sleep(2.5)  # > bsread's 1 s receive timeout: datahub's thread dies
        os.kill(ts._process.pid, signal.SIGCONT)
        n = len(s)
        assert wait_until(lambda: len(s) > n + 20, timeout=8)
        assert ew.reconnects >= 1
        s.accumulate(False)
    finally:
        ew.stopEventLoop()
        ts.stop()
