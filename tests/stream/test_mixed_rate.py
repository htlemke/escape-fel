"""Mixed-rate channels and reused Streams in escape.stream.

A fake handler delivers only the registered channels, like the dispatcher
with ``incomplete: fill-null``: on pulses where a lower-rate channel has no
data its value is ``None``. Everything runs offline.
"""
import collections
import threading
import time

import numpy as np
import pytest

from escape.stream import escape_stream as es
from escape.stream import es_wrappers_datahub as dh


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------

class Ev:
    def __init__(self, pid, msg):
        self.pid, self.msg = pid, msg

    def getEventId(self):  # noqa: N802
        return self.pid

    def getFromSource(self, ch):  # noqa: N802
        return self.pid if ch == "pulse_id" else self.msg.get(ch)

    def get_channel_names(self):
        return list(self.msg)


def make_msg(pid, channels):
    m = {}
    for ch in channels:
        if ch == "fast":  # every pulse
            m[ch] = float(pid)
        elif ch == "slow":  # every 4th pulse
            m[ch] = float(pid) if pid % 4 == 0 else None
        elif ch == "evtset":  # array, every 2nd pulse; bit 1 set every 8th
            if pid % 2:
                m[ch] = None
            else:
                a = np.zeros(4)
                a[1] = 1.0 if pid % 8 == 0 else 0.0
                m[ch] = a
    return m


_pid_lock = threading.Lock()
_pid = [1000]


def next_pid():
    with _pid_lock:
        _pid[0] += 1
        return _pid[0]


class Ctx:
    def __init__(self, handler):
        self.h = handler

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_event(self):
        time.sleep(0.001)
        pid = next_pid()
        return Ev(pid, make_msg(pid, list(self.h.source_ids)))


class FakeHandler:
    """Like DataHubEventHandler: only registered channels arrive."""

    def __init__(self):
        self.source_ids = []
        self.connections = 0

    def register_source(self, s):
        if s != "pulse_id" and s not in self.source_ids:
            self.source_ids.append(s)

    def remove_source(self, s):
        if s in self.source_ids:
            self.source_ids.remove(s)

    def context_manager(self):
        self.connections += 1
        return Ctx(self)


@pytest.fixture
def ew():
    w = es.EventWorker(FakeHandler(), make_default=False, restart_mode="break_before_make")
    yield w
    w.stopEventLoop()


def pairs(s):
    return [(e, d) for se, sd in zip(s.eventIds, s.data) for e, d in zip(se, sd)]


def wait_until(cond, timeout=5.0):
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            return False
        time.sleep(0.01)
    return True


def run(stream, seconds=0.5, n=40):
    """Accumulate *stream* until it has *n* samples (at least *seconds*)."""
    stream.accumulate(True)
    time.sleep(seconds)
    assert wait_until(lambda: len(stream) >= n)
    stream.accumulate(False)
    return pairs(stream)


# ---------------------------------------------------------------------------
# E1: a derived value exists only on pulses where all inputs exist
# ---------------------------------------------------------------------------

def test_operator_with_lower_rate_input_has_no_stale_values(ew):
    fast, slow = es.Stream("fast", ew), es.Stream("slow", ew)
    got = run(fast - slow, n=20)
    assert got and all(pid % 4 == 0 for pid, _ in got)
    assert all(val == 0.0 for _, val in got)


def test_element_mask_of_lower_rate_array_is_not_reused(ew):
    fast, evt = es.Stream("fast", ew), es.Stream("evtset", ew)
    got = run(fast[evt[1]], n=10)
    assert got and all(pid % 8 == 0 for pid, _ in got)


def test_element_of_lower_rate_array_only_on_its_pulses(ew):
    got = run(es.Stream("evtset", ew)[1], n=20)
    assert got and all(pid % 2 == 0 for pid, _ in got)


def test_running_mean_is_held_between_input_pulses(ew):
    fast, slow = es.Stream("fast", ew), es.Stream("slow", ew)
    ratio = fast / slow.running_mean(4)
    got = run(ratio, n=40)
    # held: a value on every pulse, not just every 4th
    assert any(pid % 4 for pid, _ in got)


# ---------------------------------------------------------------------------
# E9: a missing scan parameter skips the pulse instead of failing
# ---------------------------------------------------------------------------

def test_categorize_by_lower_rate_key(ew):
    fast, slow = es.Stream("fast", ew), es.Stream("slow", ew)
    key = (slow * 0) + 1.0  # constant bin, exists every 4th pulse
    binned = es.Stream(source=fast._source, scan=es.Scan(parameters=[key]))
    got = run(binned, n=20)
    assert all(pid % 4 == 0 for pid, _ in got)
    assert not ew._callback_failures


# ---------------------------------------------------------------------------
# E2/E3: reference-counted subscriptions, per-consumer accumulation
# ---------------------------------------------------------------------------

def test_stopping_plain_stream_keeps_channel_for_derived(ew):
    fast = es.Stream("fast", ew)
    der = fast * 2
    der.accumulate(True)
    fast.accumulate(True)
    time.sleep(0.2)
    fast.accumulate(False)
    assert "fast" in ew._eventHandler.source_ids
    n = len(der)
    assert wait_until(lambda: len(der) > n + 20)
    vals = [v for _, v in pairs(der)][-20:]
    assert len(set(vals)) == 20  # fresh values, not one frozen one
    der.accumulate(False)
    assert ew._eventHandler.source_ids == []
    assert ew.subscriptions == {}


def test_legacy_register_and_stream_share_channel(ew):
    ew.registerSource("fast")
    s = es.Stream("fast", ew)
    s.accumulate(True)
    s.accumulate(False)
    assert "fast" in ew._eventHandler.source_ids  # eco's registration survives
    ew.removeSource("fast")
    assert "fast" not in ew._eventHandler.source_ids


def test_two_consumers_one_stops(ew):
    fast = es.Stream("fast", ew)
    a, b = object(), object()
    fast.accumulate(True, owner=a)
    fast.accumulate(True, owner=b)
    fast.accumulate(False, owner=a)
    assert fast._is_accumulating()
    assert fast.consumers == [b]
    fast.accumulate(False, owner=b)
    assert not fast._is_accumulating()
    assert ew.subscriptions == {}


def test_plain_accumulate_false_releases_everyone(ew):
    fast = es.Stream("fast", ew)
    fast.accumulate(True, owner=object())
    fast.accumulate(True)
    fast.accumulate(False)
    assert not fast._is_accumulating() and fast.consumers == []


def test_closing_a_plot_leaves_other_plot_running(ew):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from escape.stream import plots

    fast = es.Stream("fast", ew)
    p1 = plots.TracePlot(fast, axes=plt.subplots()[1])
    p2 = plots.TracePlot(fast, axes=plt.subplots()[1])
    assert fast._is_accumulating()
    p1._on_close(None)
    assert fast._is_accumulating() and "fast" in ew._eventHandler.source_ids
    p2._on_close(None)
    assert not fast._is_accumulating() and ew.subscriptions == {}
    plt.close("all")


def test_with_block_does_not_stop_other_consumer(ew):
    fast = es.Stream("fast", ew)
    owner = object()
    fast.accumulate(True, owner=owner)
    with fast:
        pass
    assert fast._is_accumulating()
    fast.accumulate(False, owner=owner)


# ---------------------------------------------------------------------------
# E4: repeated stream[i][mask] gives the same live object
# ---------------------------------------------------------------------------

def test_repeated_indexing_returns_same_stream(ew):
    fast, evt = es.Stream("fast", ew), es.Stream("evtset", ew)
    assert evt[1] is evt[1]
    assert fast[evt[1]] is fast[evt[1]]
    assert fast[evt[2]] is not fast[evt[1]]
    assert fast[~evt[1]] is not fast[~evt[1]]  # a new mask each time


def test_reused_stream_keeps_accumulated_data(ew):
    fast, evt = es.Stream("fast", ew), es.Stream("evtset", ew)
    first = fast[evt[1]]
    first.accumulate(True)
    assert wait_until(lambda: len(first) >= 5)
    assert len(fast[evt[1]]) >= 5
    first.accumulate(False)


# ---------------------------------------------------------------------------
# E7/E8: timeouts wait for restarts and say why
# ---------------------------------------------------------------------------

def test_timeout_message_names_channel_and_loop_state(ew):
    fast, evt = es.Stream("fast", ew), es.Stream("evtset", ew)
    s = fast[evt[3]]  # bit 3 is never set
    s.accumulate(True)
    with pytest.raises(TimeoutError) as info:
        s._wait_for_data(timeout=0.5)
    msg = str(info.value)
    assert "channels: fast, evtset" in msg and "event loop: alive" in msg
    assert "evtset: data in" in msg
    s.accumulate(False)


def test_wait_for_data_waits_for_pending_restart(ew):
    s = es.Stream("fast", ew)
    s.accumulate(True)
    ew._schedule_restart(delay=0.5)
    assert ew.stats()["restart_pending"]
    t0 = time.time()
    s._wait_for_data(timeout=5)
    assert time.time() - t0 >= 0.4  # waited for the restart first
    assert not ew.stats()["restart_pending"]
    s.accumulate(False)


def test_stats(ew):
    s = es.Stream("fast", ew)
    s.accumulate(True)
    assert wait_until(lambda: len(s) > 60)
    st = ew.stats()
    assert st["loop_alive"] and st["rate_Hz"] > 10 and st["n_callbacks"] == 1
    assert st["subscriptions"] == {"fast": 1}
    s.accumulate(False)


# ---------------------------------------------------------------------------
# E5/E6: datahub receive thread dies / restart handover
# ---------------------------------------------------------------------------

class FakeBsreadStream:
    """Mimics datahub's BsreadStream: a background thread fills a buffer and
    ends for good after *die_after* messages (as on bsread's receive
    timeout). Records whether close() ran while the thread was alive (the
    psi-datahub context-before-join order that raises ZMQError)."""

    closed_while_running = 0

    def __init__(self, channels, die_after=None, first_delay=0.0):
        self.channels = channels
        self.buf = collections.deque(maxlen=100)
        self.cond = threading.Condition()
        self.aborted = False
        self.die_after = die_after
        self.first_delay = first_delay
        self.processing_thread = threading.Thread(target=self._run, daemon=True)
        self.processing_thread.start()

    def _run(self):
        # Like bsread's receive timeout: an abort is noticed within ~0.1 s.
        deadline = time.time() + self.first_delay
        while time.time() < deadline and not self.aborted:
            time.sleep(0.05)
        n = 0
        while not self.aborted:
            if self.die_after is not None and n >= self.die_after:
                raise_exc = RuntimeError("Received None message.")
                self.run_exception = raise_exc
                return
            time.sleep(0.002)
            pid = next_pid()
            with self.cond:
                self.buf.append((pid, 0, make_msg(pid, self.channels)))
                self.cond.notify()
            n += 1

    run_exception = None

    def receive(self, timeout=None):
        with self.cond:
            if not self.buf:
                self.cond.wait(timeout)
            return self.buf.popleft() if self.buf else None

    def is_thread_running(self):
        return self.processing_thread.is_alive()

    def get_run_exception(self):
        return self.run_exception

    def abort(self):
        self.aborted = True

    def join(self, timeout=None):
        self.processing_thread.join(timeout)

    def close(self):
        if self.processing_thread.is_alive():
            FakeBsreadStream.closed_while_running += 1
        self.aborted = True


class FakeDataHubHandler(FakeHandler):
    def __init__(self, die_after=None, first_delay=0.0):
        super().__init__()
        self.receive_timeout = 0.1
        self.stream = None
        self.die_after = die_after
        self.first_delay = first_delay

    def clone(self):
        h = FakeDataHubHandler(self.die_after, self.first_delay)
        h.connections = self.connections
        return h

    def context_manager(self):
        self.connections += 1
        self.stream = FakeBsreadStream(list(self.source_ids), self.die_after, self.first_delay)
        return dh.DataHubEventSourceContext(self)


def test_dead_receive_thread_reconnects():
    ew = es.EventWorker(FakeDataHubHandler(die_after=50), make_default=False)
    try:
        s = es.Stream("fast", ew)
        s.accumulate(True)
        # 50 messages per connection: > 150 samples needs >= 3 connections.
        assert wait_until(lambda: len(s) > 150, timeout=10)
        assert ew._eventHandler.connections >= 3
        s.accumulate(False)
    finally:
        ew.stopEventLoop()


def test_stall_watchdog_reconnects():
    # A connection that never delivers anything, but whose thread stays up.
    ew = es.EventWorker(
        FakeDataHubHandler(first_delay=1e9), make_default=False, restart_mode="break_before_make"
    )
    ew.stall_timeout = 0.3
    try:
        s = es.Stream("fast", ew)
        s.accumulate(True)
        assert wait_until(lambda: ew._eventHandler.connections >= 2, timeout=5)
        s.accumulate(False)
    finally:
        ew.stopEventLoop()


def test_datahub_exit_closes_after_thread_ended():
    FakeBsreadStream.closed_while_running = 0
    h = FakeDataHubHandler()
    h.register_source("fast")
    with h.context_manager() as ctx:
        assert ctx.get_event().getEventId() is not None
    assert FakeBsreadStream.closed_while_running == 0
    assert h.stream is None


def test_make_before_break_handover_waits_for_first_event_and_never_overlaps():
    ew = es.EventWorker(FakeDataHubHandler(first_delay=0.3), make_default=False)
    try:
        seen = []
        fast = es.Stream("fast", ew)
        fast.accumulate(True)
        cb = lambda: seen.append(ew.event.getEventId())  # noqa: E731
        ew.eventCallbacks.append(cb)
        assert wait_until(lambda: len(fast) > 20)
        slow = es.Stream("slow", ew)
        slow.accumulate(True)  # channel change -> make-before-break restart
        assert ew.wait_for_restart(10)
        assert "slow" in ew._eventHandler.source_ids
        assert wait_until(lambda: len(slow) > 5)
        # Never two connections dispatching: event ids strictly increase.
        assert all(b > a for a, b in zip(seen, seen[1:]))
        fast.accumulate(False)
        slow.accumulate(False)
    finally:
        ew.stopEventLoop()


# ---------------------------------------------------------------------------
# Related lifecycle fixes
# ---------------------------------------------------------------------------

def test_stop_is_not_undone_by_pending_restart(ew):
    s = es.Stream("fast", ew)
    s.accumulate(True)
    assert wait_until(lambda: len(s) > 5)
    s.accumulate(False)  # arms a debounced restart (channel removed)
    ew.stopEventLoop()
    time.sleep(0.5)
    assert not ew.loopThread.is_alive()
    # subscribing again starts it again
    s.accumulate(True)
    assert wait_until(lambda: ew.loopThread.is_alive())
    s.accumulate(False)


def test_auto_removed_callback_releases_its_channels(ew):
    s = es.Stream("slow", ew)[2]  # fails every pulse where slow has data
    s.accumulate(True)
    assert wait_until(lambda: not s._is_accumulating(), timeout=10)
    assert s.consumers == [] and ew.subscriptions == {}
    with pytest.raises(TimeoutError, match="repeated errors.*not subscriptable"):
        s._wait_for_data(timeout=0.1)


def test_removing_last_channel_restarts_quickly():
    ew = es.EventWorker(FakeDataHubHandler(), make_default=False)
    try:
        s = es.Stream("fast", ew)
        s.accumulate(True)
        assert wait_until(lambda: len(s) > 5)
        s.accumulate(False)
        t0 = time.time()
        assert ew.wait_for_restart(10)
        assert time.time() - t0 < 3  # not the 10 s first-event timeout
    finally:
        ew.stopEventLoop()
