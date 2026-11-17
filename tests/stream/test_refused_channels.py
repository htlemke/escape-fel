"""Names that are not channels never reach the dispatcher, and channels the
dispatcher refuses are left out instead of being asked for forever.

The real dispatcher answers a stream request with HTTP 500 ("Unable to
request stream") when none of the requested channels is one it knows, and
silently drops unknown ones next to known ones. The fake below does the same.
"""
import collections
import threading
import time

import pytest

import escape.stream.escape_stream as es
from escape.stream import es_wrappers_datahub as dh

RATE = 100.0
KNOWN = {"fast", "slow"}


class RefusingDispatcherStream:
    connections = []

    def __init__(self, channels, url=None, **kwargs):
        self.channels = list(channels)
        self.buf = collections.deque(maxlen=1000)
        self.cond = threading.Condition()
        self.aborted = False
        self.error = None
        RefusingDispatcherStream.connections.append(self.channels)
        self.processing_thread = threading.Thread(target=self._run, daemon=True)
        self.processing_thread.start()

    def _run(self):
        known = [c for c in self.channels if c in KNOWN]
        if not known:
            self.error = Exception(
                'Unable to request stream for specified channels - {"status":500,"error":"Internal Server Error"}'
            )
            return
        pid = int(time.time() * RATE)
        while not self.aborted:
            while int(time.time() * RATE) < pid and not self.aborted:
                time.sleep(0.002)
            with self.cond:
                self.buf.append((pid, int(pid / RATE * 1e9), {c: float(pid) for c in known}))
                self.cond.notify()
            pid += 1

    def receive(self, timeout=None):
        with self.cond:
            if not self.buf and timeout:
                self.cond.wait(timeout)
            return self.buf.popleft() if self.buf else None

    def get_run_exception(self):
        return self.error

    def abort(self):
        self.aborted = True

    def join(self, timeout=None):
        self.processing_thread.join(timeout)

    def close(self):
        self.aborted = True


def channel_status(channels, base_url=None):
    return {c: {"configured": c in KNOWN, "connected": c in KNOWN, "recording": c in KNOWN} for c in channels}


@pytest.fixture
def ew(monkeypatch):
    from bsread import dispatcher

    monkeypatch.setattr(dh, "_TolerantBsreadStream", RefusingDispatcherStream)
    monkeypatch.setattr(dh, "_CHANNEL_DELAY", {})
    monkeypatch.setattr(dh, "_LATE_CHANNELS", {})
    monkeypatch.setattr(dh, "_clock_offset", lambda *a, **k: 0.0)
    monkeypatch.setattr(dispatcher, "get_channel_status", channel_status)
    RefusingDispatcherStream.connections = []
    w = es.EventWorker(
        dh.DataHubEventHandler(backend="bsread", late_threshold_s=0.5),
        make_default=False,
        restart_mode="break_before_make",
    )
    yield w
    w.stopEventLoop()


def wait_until(cond, timeout=10.0):
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            return False
        time.sleep(0.02)
    return True


@pytest.mark.parametrize("name", ["(a / b)", "i[pump_on]", "x[3]", "a b", ""])
def test_a_computed_streams_name_is_refused_at_subscribe(ew, name):
    with pytest.raises(ValueError, match="not a channel name"):
        ew.subscribe(object(), name)
    with pytest.raises(ValueError, match="_channels"):
        ew.registerSource(name)
    assert name not in ew._eventHandler.source_ids


def test_real_channel_names_pass():
    for name in ("SAROP21-PBPS133:INTENSITY", "JF01T03V01:roi_intensities", "SAR-CVME-TIFALL5:EvtSet"):
        assert es.check_channel_name(name) == name


def test_a_computed_stream_subscribes_its_channels(ew):
    fast, slow = es.Stream("fast", ew), es.Stream("slow", ew)
    ratio = fast / slow
    assert ratio._channels() == ["fast", "slow"]
    ratio.accumulate(True)
    assert wait_until(lambda: len(ratio) > 50)
    assert all("(" not in c for chs in RefusingDispatcherStream.connections for c in chs)
    ratio.accumulate(False)


def test_a_refused_channel_is_left_out_once(ew, capsys):
    ew.registerSource("NOPE-XXX:YYY")  # a typo: a valid name the dispatcher does not know
    fast = es.Stream("fast", ew)
    fast.accumulate(True)
    assert wait_until(lambda: len(fast) > 50)
    out = capsys.readouterr().out
    assert out.count("the dispatcher does not know 'NOPE-XXX:YYY'") == 1
    assert ew._eventHandler.refused == {"NOPE-XXX:YYY"}
    time.sleep(1.0)
    assert ew.reconnects == 0
    assert "refused by the dispatcher" in ew.diagnose(["NOPE-XXX:YYY"])
    # no later connection asks for it again
    assert "NOPE-XXX:YYY" not in RefusingDispatcherStream.connections[-1]
    fast.accumulate(False)


def test_unsubscribing_forgets_the_refusal(ew):
    ew.registerSource("NOPE-XXX:YYY")
    fast = es.Stream("fast", ew)
    fast.accumulate(True)
    assert wait_until(lambda: "NOPE-XXX:YYY" in ew._eventHandler.refused)
    ew.removeSource("NOPE-XXX:YYY")
    assert "NOPE-XXX:YYY" not in ew._eventHandler.refused
    fast.accumulate(False)


def test_a_passing_dispatcher_error_drops_nothing(monkeypatch):
    """Every channel is known: the failure is not theirs -- reconnect as before."""
    from bsread import dispatcher

    monkeypatch.setattr(dispatcher, "get_channel_status", channel_status)
    handler = dh.DataHubEventHandler(backend="bsread")
    handler.source_ids = ["fast"]
    ctx = dh._GroupedDispatcherContext(handler)
    group = ctx.groups[0]
    assert not ctx._refuse(group, Exception("Unable to request stream for specified channels - 500"))
    assert not ctx._refuse(group, Exception("connection reset"))
    assert handler.refused == set() and group.channels == ["fast"]
