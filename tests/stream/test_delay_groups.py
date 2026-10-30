"""Late channels get their own dispatcher connection (delay groups).

A fake dispatcher stream reproduces what the real dispatcher does when a
late channel (a Jungfrau ROI, ~12 s) is requested together with prompt
ones: it only delivers the late channel's pulses, late, with every other
channel None. Here "late" arrives 1 s late, on every 2nd pulse.
"""
import collections
import threading
import time

import pytest

import escape.stream.escape_stream as es
from escape.stream import es_wrappers_datahub as dh

RATE = 100.0   # pulses per second
DELAY = 1.0    # s, the late channel


def pulse_now():
    return int(time.time() * RATE)


class FakeDispatcherStream:
    connections = []

    rates = {}  # channel -> deliver data only every n-th pulse (default 1)
    skew = 0.0  # s: pulse timestamps this much behind the local clock
    hold = {}   # channel -> s the dispatcher holds back a connection that has it

    def __init__(self, channels, url=None, **kwargs):
        self.channels = list(channels)
        self.buf = collections.deque(maxlen=1000)
        self.cond = threading.Condition()
        self.aborted = False
        FakeDispatcherStream.connections.append(self.channels)
        self.processing_thread = threading.Thread(target=self._run, daemon=True)
        self.processing_thread.start()

    def _run(self):
        late = "late" in self.channels
        hold = max([0.0] + [self.hold.get(c, 0.0) for c in self.channels])
        pid = pulse_now() - int(hold * RATE)  # a held-back stream starts in the past
        while not self.aborted:
            target = pid + int(hold * RATE) if not late else pid + int(DELAY * RATE)
            while pulse_now() < target and not self.aborted:
                time.sleep(0.002)
            if not late:
                msg = {c: (float(pid) if pid % self.rates.get(c, 1) == 0 else None) for c in self.channels}
            elif pid % 2 == 0:
                msg = {c: (float(pid) if c == "late" else None) for c in self.channels}
            else:
                msg = None
            if msg is not None:
                with self.cond:
                    self.buf.append((pid, int((pid / RATE - self.skew) * 1e9), msg))
                    self.cond.notify()
            pid += 1

    def receive(self, timeout=None):
        with self.cond:
            if not self.buf and timeout:
                self.cond.wait(timeout)
            return self.buf.popleft() if self.buf else None

    def get_run_exception(self):
        return None

    def abort(self):
        self.aborted = True

    def join(self, timeout=None):
        self.processing_thread.join(timeout)

    def close(self):
        self.aborted = True


@pytest.fixture
def ew(monkeypatch):
    monkeypatch.setattr(dh, "_TolerantBsreadStream", FakeDispatcherStream)
    monkeypatch.setattr(dh, "_CHANNEL_DELAY", {})
    monkeypatch.setattr(dh, "_LATE_CHANNELS", {})
    monkeypatch.setattr(dh._GroupedDispatcherContext, "ABSENT_S", 2.5)
    FakeDispatcherStream.connections = []
    FakeDispatcherStream.rates = {}
    FakeDispatcherStream.skew = 0.0
    FakeDispatcherStream.hold = {}
    w = es.EventWorker(
        dh.DataHubEventHandler(backend="bsread", late_threshold_s=0.5),
        make_default=False,
        restart_mode="break_before_make",
    )
    yield w
    w.stopEventLoop()


def pairs(s):
    return [(e, d) for se, sd in zip(s.eventIds, s.data) for e, d in zip(se, sd)]


def wait_until(cond, timeout=10.0):
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            return False
        time.sleep(0.02)
    return True


def test_prompt_only_stays_one_connection(ew):
    fast = es.Stream("fast", ew)
    fast.accumulate(True)
    assert wait_until(lambda: len(fast) > 300)
    assert len(ew._eventHandler.groups) == 1
    assert FakeDispatcherStream.connections == [["fast"]]
    fast.accumulate(False)


def test_late_channel_is_split_off_and_prompt_data_recovers(ew):
    fast, late = es.Stream("fast", ew), es.Stream("late", ew)
    fast.accumulate(True)
    late.accumulate(True)
    # split after one evaluation window of the mixed connection
    assert wait_until(lambda: len(ew._eventHandler.groups) == 2)
    groups = ew._eventHandler.groups
    assert groups[0][0] == ["fast"] and groups[1][0] == ["late"]
    assert dh._CHANNEL_DELAY["late"] == pytest.approx(DELAY, abs=0.3)
    assert dh._LATE_CHANNELS["late"] == pytest.approx(DELAY, abs=0.3)
    assert "fast" not in dh._LATE_CHANNELS
    # prompt data arrives promptly again
    n = len(fast)
    assert wait_until(lambda: len(fast) > n + 100)
    assert pulse_now() - pairs(fast)[-1][0] < 0.3 * RATE
    # and the late channel keeps coming, on its own pulses
    m = len(late)
    assert wait_until(lambda: len(late) > m + 50)
    assert all(pid % 2 == 0 and val == pid for pid, val in pairs(late)[-50:])
    fast.accumulate(False)
    late.accumulate(False)


def test_late_and_prompt_values_combine_on_the_same_pulse(ew):
    fast, late = es.Stream("fast", ew), es.Stream("late", ew)
    diff = late - fast  # both carry the pulse id: 0 only if matched correctly
    diff.accumulate(True)
    fast.accumulate(True)
    assert wait_until(lambda: len(ew._eventHandler.groups) == 2)
    n = len(diff)
    assert wait_until(lambda: len(diff) > n + 50, timeout=15)
    got = pairs(diff)[n:]
    assert got and all(val == 0.0 for _, val in got)
    assert diff._event_group() == 1 and fast._event_group() == 0
    diff.accumulate(False)
    fast.accumulate(False)


def test_learned_delay_splits_at_once(ew):
    dh._LATE_CHANNELS["late"] = DELAY
    fast, late = es.Stream("fast", ew), es.Stream("late", ew)
    ew.subscribe("test", "fast", "late")
    fast.accumulate(True)
    assert wait_until(lambda: len(fast) > 50)
    assert len(ew._eventHandler.groups) == 2
    assert ["fast", "late"] not in FakeDispatcherStream.connections  # never mixed
    fast.accumulate(False)
    ew.unsubscribe("test")


def test_running_mean_is_not_fed_twice(ew):
    dh._LATE_CHANNELS["late"] = DELAY
    fast, late = es.Stream("fast", ew), es.Stream("late", ew)
    rm = fast.running_mean(10)
    norm = late / rm
    norm.accumulate(True)
    rm.accumulate(True)
    assert wait_until(lambda: len(norm) > 50, timeout=15)
    ids = [pid for pid, _ in pairs(rm)]
    assert ids == sorted(set(ids))  # only prompt events, in order, once each
    norm.accumulate(False)
    rm.accumulate(False)


def test_plot_wait_covers_the_split(ew):
    fast, late = es.Stream("fast", ew), es.Stream("late", ew)
    late.accumulate(True)
    fast.accumulate(True)
    # fast has no data until the split (~2-3 s here); the wait must not give up at 0.5 s
    fast._wait_for_data(timeout=0.5)
    assert len(fast) >= 2
    fast.accumulate(False)
    late.accumulate(False)


def test_clock_offset_is_subtracted(ew, monkeypatch):
    # a local clock 3.5 s fast must not make prompt data look late
    monkeypatch.setattr(dh, "_clock_offset", lambda *a, **k: 3.5)
    FakeDispatcherStream.skew = 3.5
    fast, rare = es.Stream("fast", ew), es.Stream("rare", ew)
    FakeDispatcherStream.rates = {"rare": 10**9}  # never has data
    fast.accumulate(True)
    rare.accumulate(True)
    assert wait_until(lambda: dh._CHANNEL_DELAY.get("fast") is not None)
    assert abs(dh._CHANNEL_DELAY["fast"]) < 0.3
    time.sleep(3)
    assert len(ew._eventHandler.groups) == 1  # no split
    fast.accumulate(False)
    rare.accumulate(False)


def test_low_rate_channel_in_delayed_stream_is_not_split_off(ew, monkeypatch):
    # The whole stream 3 s late (as the dispatcher does with a ~2 s camera
    # in the request), plus a 0.2 Hz channel that misses whole windows:
    # it has data, so nothing must be split.
    monkeypatch.setattr(dh._GroupedDispatcherContext, "ABSENT_S", 6.0)
    FakeDispatcherStream.skew = 3.0  # everything 3 s late, clock correct
    FakeDispatcherStream.rates = {"slow": int(5 * RATE)}  # every 5 s
    fast, slow = es.Stream("fast", ew), es.Stream("slow", ew)
    fast.accumulate(True)
    slow.accumulate(True)
    time.sleep(9)
    assert len(ew._eventHandler.groups) == 1
    assert len(slow) >= 1
    fast.accumulate(False)
    slow.accumulate(False)


def test_delayed_stream_does_not_mark_prompt_channels_late(ew):
    FakeDispatcherStream.skew = 3.0  # whole stream held back, nothing missing
    fast = es.Stream("fast", ew)
    fast.accumulate(True)
    assert wait_until(lambda: dh._CHANNEL_DELAY.get("fast") is not None)
    assert dh._CHANNEL_DELAY["fast"] == pytest.approx(3.0, abs=0.3)
    assert dh._LATE_CHANNELS == {}
    fast.accumulate(False)


def test_switching_to_a_held_back_connection_repeats_no_pulse(monkeypatch):
    # Adding a slower channel (a camera: the dispatcher holds the whole
    # stream back ~2 s) makes the replacement connection start in the past:
    # pulses already dispatched must not be dispatched again. Removing it
    # skips ahead instead (an unavoidable gap).
    monkeypatch.setattr(dh, "_TolerantBsreadStream", FakeDispatcherStream)
    monkeypatch.setattr(dh, "_CHANNEL_DELAY", {})
    monkeypatch.setattr(dh, "_LATE_CHANNELS", {})
    FakeDispatcherStream.connections, FakeDispatcherStream.rates, FakeDispatcherStream.skew = [], {}, 0.0
    FakeDispatcherStream.hold = {"cam": 1.0}
    w = es.EventWorker(dh.DataHubEventHandler(backend="bsread"), make_default=False)  # make-before-break
    try:
        seen = []
        w.eventCallbacks.append(lambda: seen.append(w.event.getEventId()))
        fast = es.Stream("fast", w)
        fast.accumulate(True)
        assert wait_until(lambda: len(fast) > 100)
        cam = es.Stream("cam", w)
        cam.accumulate(True)
        assert w.wait_for_restart(15)
        assert wait_until(lambda: len(cam) > 100)
        cam.accumulate(False)
        assert w.wait_for_restart(15)
        n = len(fast)
        assert wait_until(lambda: len(fast) > n + 100)
        assert all(b > a for a, b in zip(seen, seen[1:]))  # never a pulse twice or backwards
        ids = [pid for pid, _ in pairs(fast)]
        assert ids == sorted(set(ids))
        fast.accumulate(False)
    finally:
        w.stopEventLoop()


def test_removing_the_last_prompt_channel_keeps_late_data_flowing(ew):
    dh._LATE_CHANNELS["late"] = DELAY
    ew.stopEventLoop()
    ew = es.EventWorker(dh.DataHubEventHandler(backend="bsread", late_threshold_s=0.5),
                        make_default=False)  # default make-before-break: no in-flight loss
    fast, late = es.Stream("fast", ew), es.Stream("late", ew)
    fast.accumulate(True)
    late.accumulate(True)
    assert wait_until(lambda: len(late) > 50, timeout=15)
    fast.accumulate(False)  # restart with the late channel only
    assert ew.wait_for_restart(15)
    assert ew._eventHandler.group_of("late") == 1  # number 0 stays reserved for prompt data
    last = pairs(late)[-1][0]
    assert wait_until(lambda: len(late) and pairs(late)[-1][0] > last + 50, timeout=5)
    ids = [pid for pid, _ in pairs(late)]
    gaps = [b - a for a, b in zip(ids, ids[1:])]
    assert max(gaps) < 0.5 * RATE  # no stretch lost as "already dispatched"
    late.accumulate(False)
    ew.stopEventLoop()
