"""
datahub-based event source wrappers for escape.stream.

PSI is migrating from raw bsread (ZMQ) to Redis/Dragonfly as the live-stream
transport.  This module wraps both behind a single ``DataHubEventHandler`` API
using the `psi-datahub` abstraction package, so switching backends requires only
changing the ``backend`` keyword — no code changes elsewhere in escape.

Installation::

    conda install -c paulscherrerinstitute -c conda-forge datahub
    # or
    pip install psi-datahub

Supported backends
------------------
``'bsread'``
    Live PSI channels via the SwissFEL dispatcher → bsread SUB stream.
    This is the current (2024) production path.
    URL default: bsread.DEFAULT_DISPATCHER_URL (sf-databuffer dispatcher).

``'redis'``
    Live PSI channels via Redis/Dragonfly (next-generation backend, post-bsread).
    URL default: sf-daqsync-18:6379.

``'auto'``
    Try ``redis`` first; fall back to ``bsread`` if the redis library is missing.

For connecting to local test streams use ``DataHubLocalEventHandler``.

Quick start
-----------
::

    from escape.stream import (
        Stream, EventWorker, DataHubEventHandler, DataHubLocalEventHandler,
        TestStream,
    )

    # a) Synthetic test data (works offline)
    ts = TestStream()
    ts.start()
    ew = EventWorker(DataHubLocalEventHandler(host='localhost', port=9999))
    i0 = Stream('i0', ew)

    # b) Live SwissFEL dispatcher data
    ew = EventWorker(DataHubEventHandler(backend='bsread'))
    i0 = Stream('SARBD02-DBPM070:Q1', ew)

    # c) New Redis backend
    ew = EventWorker(DataHubEventHandler(backend='redis'))
    i0 = Stream('SARBD02-DBPM070:Q1', ew)

    # d) Special source with explicit URL
    ew = EventWorker(DataHubEventHandler(backend='bsread',
                                          url='https://dispatcher-api.psi.ch/sf-databuffer'))
"""

import queue
import threading
import time
import logging
from collections import OrderedDict, deque

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — datahub and its streaming classes
# ---------------------------------------------------------------------------

try:
    from datahub import BsreadStream, RedisStream
    _HAS_DATAHUB = True
except ImportError:
    BsreadStream = None
    RedisStream = None
    _HAS_DATAHUB = False

try:
    import redis as _redis_lib
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


# ---------------------------------------------------------------------------
# Missing channels -> None (datahub would end its receive thread instead)
# ---------------------------------------------------------------------------
#
# The dispatcher leaves a channel out of a message when it has no data for it
# (e.g. a detector that is not running, or between pulses of a lower-rate
# one). psi-datahub's Bsread.run (1.1.13) builds every message with
# ``data[channel]`` for each *requested* channel, so the KeyError ends the
# receive thread -- and with it the stream for every other channel too.
# _TolerantBsreadStream runs datahub's own run() code, but with a ``bsread``
# module whose sources return each missing requested channel as None, which
# is what ``incomplete: fill-null`` means. Reported upstream; drop this once
# datahub handles it.

class _MissingValue:
    value = None


class _FillMissingSource:
    """bsread source wrapper: fills requested channels missing from a message."""

    def __init__(self, src, channels):
        self._src = src
        self._channels = list(channels or [])

    def __enter__(self):
        self._src.__enter__()
        return self

    def __exit__(self, *exc):
        return self._src.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._src, name)

    def receive(self, *args, **kwargs):
        msg = self._src.receive(*args, **kwargs)
        if msg:
            data = msg.data.data
            for ch in self._channels:
                if ch not in data:
                    data[ch] = _MissingValue
        return msg


class _FillMissingBsread:
    """Stand-in for the ``bsread`` module as seen by datahub's Bsread.run."""

    def __init__(self, bsread_module):
        self._bsread = bsread_module

    def __getattr__(self, name):
        return getattr(self._bsread, name)

    def source(self, *args, channels=None, **kwargs):
        return _FillMissingSource(self._bsread.source(*args, channels=channels, **kwargs), channels)


def _make_tolerant_bsread_stream():
    import types

    run = getattr(BsreadStream, "run", None)
    run_globals = getattr(run, "__globals__", {})
    if "bsread" not in run_globals:  # datahub changed: use it unmodified
        return BsreadStream
    tolerant_run = types.FunctionType(
        run.__code__,
        dict(run_globals, bsread=_FillMissingBsread(run_globals["bsread"])),
        run.__name__,
        run.__defaults__,
        run.__closure__,
    )
    return type("_TolerantBsreadStream", (BsreadStream,), {"run": tolerant_run})


_TolerantBsreadStream = _make_tolerant_bsread_stream() if _HAS_DATAHUB else None


def _require_datahub():
    if not _HAS_DATAHUB:
        raise RuntimeError(
            "psi-datahub is not installed.\n"
            "Install with:\n"
            "  conda install -c paulscherrerinstitute -c conda-forge datahub\n"
            "  # or\n"
            "  pip install psi-datahub"
        )


# ---------------------------------------------------------------------------
# Event wrappers
# ---------------------------------------------------------------------------

class NullEvent:
    """Placeholder emitted when no message is available (timeout or no source)."""

    group = 0

    def getFromSource(self, source):  # noqa: N802
        return None

    def getEventId(self):  # noqa: N802
        return None

    def get_channel_names(self):  # noqa: N802
        return []


class DataHubEvent:
    """Wraps a datahub ``(pulse_id, timestamp_ns, msg_dict)`` tuple.

    ``msg_dict`` is a ``{channel_name: value}`` dict exactly as produced by
    ``BsreadStream.receive()`` and ``RedisStream.receive()``.
    The ``timestamp`` is nanoseconds since epoch (integer).
    """

    def __init__(self, pulse_id, timestamp_ns, msg, group=0):
        self._id = pulse_id
        self._ts_ns = timestamp_ns   # nanoseconds
        self._msg = msg              # {channel_name: value}
        # Delay group that delivered this event (see _GroupedDispatcherContext);
        # a Stream only processes events of its own group.
        self.group = group

    def getFromSource(self, channel):  # noqa: N802
        if channel == "pulse_id":
            return self._id
        if channel == "lab_time":
            return self._ts_ns * 1e-9 if self._ts_ns else None
        return self._msg.get(channel)

    def getEventId(self):  # noqa: N802
        return self._id

    def get_channel_names(self):  # noqa: N802
        return list(self._msg.keys())


# ---------------------------------------------------------------------------
# Context manager wrapping a live datahub stream
# ---------------------------------------------------------------------------

_DATAHUB_DEFAULT_RECEIVE_TIMEOUT_MS = 3000  # datahub's query.get("receive_timeout", 3000)


def _bsread_kwargs(handler):
    """Stream kwargs for BsreadStream/RedisStream: the handler's extra
    kwargs plus, if set, bsread's receive timeout under datahub's name for
    it (``receive_timeout``, ms -- not the handler's own receive_timeout)."""
    kwargs = dict(handler._stream_kwargs)
    if handler.bsread_receive_timeout_ms is not None:
        kwargs["receive_timeout"] = handler.bsread_receive_timeout_ms
    return kwargs


def _bsread_receive_timeout_s(handler):
    """How long datahub's receive thread can block before it notices an abort."""
    ms = getattr(handler, "bsread_receive_timeout_ms", None) or _DATAHUB_DEFAULT_RECEIVE_TIMEOUT_MS
    return ms / 1000.0


class DataHubEventSourceContext:
    """Context manager owning a started datahub streaming source.

    ``get_event()`` blocks for up to ``receive_timeout`` seconds then returns
    either a :class:`DataHubEvent` or a :class:`NullEvent`.  NullEvents are
    transparently skipped by the :class:`~escape.stream.EventWorker`.
    """

    def __init__(self, eventhandler):
        self.eventhandler = eventhandler

    def __enter__(self):
        return self

    def get_event(self):
        stream = self.eventhandler.stream
        if stream is None:
            time.sleep(0.05)
            return NullEvent()
        result = stream.receive(timeout=self.eventhandler.receive_timeout)
        if result is None:
            # Buffer empty. datahub's receive thread ends for good on any
            # error, including bsread's receive timeout ("Received None
            # message.", every psi-datahub release up to 1.1.13) -- after that
            # nothing ever arrives again. Raising makes EventWorker._serve
            # reconnect (with back-off). Messages queued before the thread
            # died are still delivered first. processing_thread, not
            # is_running(): datahub sets `running` only inside the new thread,
            # so a fresh stream can look "not running" for a moment.
            t = getattr(stream, "processing_thread", None)
            if t is not None and not t.is_alive():
                raise RuntimeError(f"datahub stream thread died: {stream.get_run_exception()!r}")
            return NullEvent()
        pulse_id, timestamp_ns, msg = result
        return DataHubEvent(pulse_id, timestamp_ns, msg)

    def __exit__(self, *_):
        stream = self.eventhandler.stream
        if stream is not None:
            try:
                # Let the receive thread leave its loop and close its socket
                # and zmq context itself before close(): Bsread.close()
                # destroys the context first, which, with the thread still
                # in recv(), gives "ZMQError: Socket operation on non-socket".
                stream.abort()
                stream.join(timeout=_bsread_receive_timeout_s(self.eventhandler) + 1.0)
                stream.close()
            except Exception as exc:
                _logger.debug("Error closing datahub stream: %s", exc)
        self.eventhandler.stream = None
        return False


# ---------------------------------------------------------------------------
# Delay groups: channels that arrive late get their own dispatcher connection
# ---------------------------------------------------------------------------
#
# The dispatcher merges all requested channels by pulse id. A channel whose
# data reaches the dispatcher late (a Jungfrau ROI channel: ~12 s) makes it
# deliver only that channel's pulses, with every other channel None. So the
# dispatcher context watches each connection: when its messages are late by a
# large, steady amount and some requested channels never have data, those
# channels move to the connection with the shortest delay (a new one if
# needed). Events of each connection are dispatched as they arrive, tagged
# with the connection's group index (0 = shortest delay); a late event also
# carries the values the earlier groups delivered for the same pulse, kept in
# a buffer until then. Learned delays are remembered for the session
# (_CHANNEL_DELAY), so a known late channel gets its own connection at once.

_CHANNEL_DELAY = {}  # channel -> last measured dispatcher delay (s)


class _DelayGroup:
    """One dispatcher connection and its delay statistics."""

    WINDOW = 100  # messages per evaluation

    def __init__(self, channels):
        self.channels = list(channels)
        self.stream = None
        self.connected = time.time()
        self.n_msgs = 0
        self.delays = deque(maxlen=self.WINDOW)
        self.seen = {}            # channel -> messages with data in this window
        self.window_msgs = 0
        self.window_start = time.time()
        self.delay = None         # median delay (s) of the last window
        self.buffer = OrderedDict()  # pulse id -> (pulse time, values), for later groups

    def sort_key(self):
        if self.delay is not None:
            return self.delay
        return max((_CHANNEL_DELAY.get(c, 0.0) for c in self.channels), default=0.0)


def _plan_groups(channels, threshold):
    """Split *channels* into delay groups from what was learned so far:
    unknown and prompt channels together, late ones clustered by delay."""
    prompt = [c for c in channels if _CHANNEL_DELAY.get(c, 0.0) < threshold]
    late = sorted((_CHANNEL_DELAY[c], c) for c in channels if c not in prompt)
    groups = [prompt] if prompt else []
    last = None
    for d, c in late:
        if last is None or d - last >= threshold:
            groups.append([])
        groups[-1].append(c)
        last = d
    return groups or [[]]


class _GroupedDispatcherContext:
    """Dispatcher context with one connection per delay group (see above)."""

    SPREAD_MAX = 1.0      # s: p90 - p10 of the delay must be below this ("steady")
    MIN_WINDOW_S = 2.0    # evaluate a window only after this long

    def __init__(self, handler):
        self.h = handler
        self.groups = [_DelayGroup(chs) for chs in _plan_groups(handler.source_ids, handler.late_threshold_s)]
        self._last_prune = time.time()

    # -- connections ---------------------------------------------------
    FIRST_DATA_WAIT = 30.0  # s a fresh connection may take to deliver a channel

    def _open(self, g):
        g.stream = None
        g.connected = time.time()
        with self.h._group_lock:
            for c in g.channels:
                self.h._awaiting[c] = g.connected + self.FIRST_DATA_WAIT
        if g.channels:
            g.stream = _TolerantBsreadStream(channels=list(g.channels), url=self.h.url, **_bsread_kwargs(self.h))

    @staticmethod
    def _close(stream, timeout):
        if stream is None:
            return
        try:
            stream.abort()
            stream.join(timeout=timeout)
            stream.close()
        except Exception as exc:
            _logger.debug("Error closing datahub stream: %s", exc)

    def _publish(self):
        """Tell the handler (and so the Streams) the current group layout."""
        self.groups.sort(key=_DelayGroup.sort_key)
        self.h._set_groups([g.channels for g in self.groups], [g.delay for g in self.groups])
        self.h.stream = self.groups[0].stream if self.groups else None

    def __enter__(self):
        try:
            for g in self.groups:
                self._open(g)
        except Exception:
            self.__exit__()
            raise
        self._publish()
        if len(self.groups) > 1:
            print(f"DataHubEventHandler: {len(self.groups)} dispatcher connections by delay: "
                  + "; ".join(", ".join(g.channels) for g in self.groups))
        return self

    def __exit__(self, *_):
        timeout = _bsread_receive_timeout_s(self.h) + 1.0
        for g in self.groups:
            if g.stream is not None:
                g.stream.abort()  # let all receive threads leave their loops at once
        for g in self.groups:
            self._close(g.stream, timeout)
            g.stream = None
        self.h.stream = None
        return False

    # -- events --------------------------------------------------------
    def get_event(self):
        deadline = time.time() + self.h.receive_timeout
        while True:
            # One connection: block in receive(). Several: poll them, the one
            # with the shortest delay first.
            wait = max(0.0, deadline - time.time()) if len(self.groups) == 1 else 0
            for i, g in enumerate(self.groups):
                if g.stream is None:
                    continue
                result = g.stream.receive(timeout=wait)
                if result is not None:
                    return self._event(i, g, result)
                t = getattr(g.stream, "processing_thread", None)
                if t is not None and not t.is_alive():
                    raise RuntimeError(f"datahub stream thread died: {g.stream.get_run_exception()!r}")
            if time.time() >= deadline:
                return NullEvent()
            time.sleep(0.001)

    def _event(self, i, g, result):
        pulse_id, timestamp_ns, msg = result
        now = time.time()
        pulse_time = timestamp_ns * 1e-9 if timestamp_ns else now
        values = {c: msg.get(c) for c in g.channels}  # a split group may still deliver others
        self._account(g, now - pulse_time, values)
        if len(self.groups) > 1:
            if i < len(self.groups) - 1:
                g.buffer[pulse_id] = (pulse_time, values)
            if i > 0:
                merged = {}
                for earlier in self.groups[:i]:
                    hit = earlier.buffer.get(pulse_id)
                    if hit is not None:
                        merged.update(hit[1])
                merged.update(values)
                values = merged
            if now - self._last_prune > 1.0:
                self._prune(now)
        if i == 0 and len(self.groups) == 1:
            return DataHubEvent(pulse_id, timestamp_ns, values)
        return DataHubEvent(pulse_id, timestamp_ns, values, group=i)

    def _prune(self, now):
        self._last_prune = now
        keep = max((g.delay or 0.0) for g in self.groups) + 5.0
        for g in self.groups:
            while g.buffer:
                pid, (t, _) = next(iter(g.buffer.items()))
                if now - t <= keep:
                    break
                g.buffer.popitem(last=False)

    # -- delay detection -----------------------------------------------
    def _account(self, g, delay, values):
        g.n_msgs += 1
        g.window_msgs += 1
        g.delays.append(delay)
        for c, v in values.items():
            if v is not None:
                g.seen[c] = g.seen.get(c, 0) + 1
                if c in self.h._awaiting:
                    self.h._awaiting.pop(c, None)
        if g.window_msgs >= _DelayGroup.WINDOW and time.time() - g.window_start >= self.MIN_WINDOW_S:
            self._evaluate(g)
            g.window_msgs, g.seen, g.window_start = 0, {}, time.time()

    def _evaluate(self, g):
        d = sorted(g.delays)
        g.delay = d[len(d) // 2]
        spread = d[int(0.9 * (len(d) - 1))] - d[int(0.1 * (len(d) - 1))]
        present = [c for c in g.channels if g.seen.get(c)]
        absent = [c for c in g.channels if not g.seen.get(c)]
        for c in present:
            _CHANNEL_DELAY[c] = g.delay
        if not self.h.auto_split:
            return
        if g.delay >= self.h.late_threshold_s and spread < self.SPREAD_MAX and present and absent:
            self._split(g, present, absent)

    def _split(self, g, present, absent):
        """Move *absent* channels of the late group *g* to the group with the
        shortest delay, or to a new connection if *g* is that group."""
        print(
            f"DataHubEventHandler: {', '.join(present)} arrive {g.delay:.1f} s late and "
            f"hold back {', '.join(absent)}; giving them separate dispatcher connections."
        )
        g.channels = present
        prompt = self.groups[0] if self.groups[0] is not g else None
        if prompt is None:
            prompt = _DelayGroup(absent)
            prompt.delay = 0.0
            self.groups.insert(0, prompt)
        else:
            prompt.channels = prompt.channels + absent
            old = prompt.stream
            threading.Thread(target=self._close, args=(old, _bsread_receive_timeout_s(self.h) + 1.0),
                             daemon=True).start()
        self._open(prompt)
        self._publish()


# ---------------------------------------------------------------------------
# DataHubEventHandler — live PSI data via dispatcher or Redis
# ---------------------------------------------------------------------------

class DataHubEventHandler:
    """Event handler for live PSI data using the datahub abstraction layer.

    Replaces :class:`~escape.stream.EventHandler_SFEL` and
    transparently supports both the current bsread/dispatcher backend and the
    upcoming Redis/Dragonfly backend.

    Parameters
    ----------
    backend : {'bsread', 'redis', 'auto'}
        Transport backend.  ``'bsread'`` uses the SwissFEL dispatcher;
        ``'redis'`` uses the Redis/Dragonfly live stream;
        ``'auto'`` picks redis if available, falls back to bsread.
    url : str, optional
        Override the default dispatcher or Redis URL.
        For bsread: a dispatcher URL (e.g. ``'https://dispatcher-api.psi.ch/sf-databuffer'``).
        For redis: ``'hostname:port'`` (e.g. ``'sf-daqsync-18:6379'``).
    receive_timeout : float
        Seconds :meth:`get_event` waits for a message before returning a
        :class:`NullEvent` (allows clean shutdown). Default 0.5 s.
    bsread_receive_timeout_ms : int, optional
        bsread's own receive timeout in ms: after this long without a
        message, datahub ends its receive thread and the EventWorker
        reconnects. ``None`` (default) keeps datahub's 3000 ms. A longer
        value (e.g. 10000) rides out short upstream stalls without a
        reconnect.
    **stream_kwargs
        Extra keyword arguments forwarded to ``BsreadStream`` / ``RedisStream``
        (e.g. ``queue_size=200``).
    """

    def __init__(self, backend="bsread", url=None, receive_timeout=0.5,
                 bsread_receive_timeout_ms=None, auto_split=True, late_threshold_s=2.0,
                 **stream_kwargs):
        _require_datahub()
        self.backend = backend
        self.url = url
        self.receive_timeout = receive_timeout
        self.bsread_receive_timeout_ms = bsread_receive_timeout_ms
        self.auto_split = auto_split
        self.late_threshold_s = late_threshold_s
        self._stream_kwargs = stream_kwargs
        self.stream = None
        self.source_ids = []
        self.group_version = 0
        self._group_of = {}
        self._group_delays = []
        self._group_lock = threading.Lock()
        self._awaiting = {}  # channel -> time until which its first data is still expected

    # ------------------------------------------------------------------
    # Delay groups (see _GroupedDispatcherContext)
    # ------------------------------------------------------------------

    def _set_groups(self, channel_lists, delays):
        with self._group_lock:
            self._group_of = {c: i for i, chs in enumerate(channel_lists) for c in chs}
            self._group_delays = list(delays)
            self.group_version += 1

    def group_of(self, channel):
        """Index of the delay group (dispatcher connection) delivering
        *channel*; 0 is the one with the shortest delay."""
        return self._group_of.get(channel, 0)

    @property
    def groups(self):
        """Current delay groups as ``[(channels, delay_s or None), ...]``."""
        with self._group_lock:
            lists = {}
            for c, i in self._group_of.items():
                lists.setdefault(i, []).append(c)
            return [(lists.get(i, []), d) for i, d in enumerate(self._group_delays)]

    def awaiting_data(self, channels):
        """True while one of *channels* is on a fresh connection that has
        not delivered data for it yet (e.g. while a late channel holds it
        back until the split), for at most 30 s."""
        now = time.time()
        return any(self._awaiting.get(c, 0) > now for c in channels)

    def expected_delay(self, channels):
        """How late data of *channels* is expected to arrive (s), from what
        was measured so far."""
        return max((_CHANNEL_DELAY.get(c, 0.0) for c in channels), default=0.0)

    def clone(self):
        """Return a fresh, independent handler with the same configuration
        (no channels registered yet, no connection) -- used by
        ``EventWorker``'s make-before-break restart to build and connect a
        replacement before tearing down the current connection. Deliberately
        a *new* handler/stream object rather than reusing ``self.stream``:
        the point is for the old and new connections to be fully
        independent while both are briefly alive."""
        return DataHubEventHandler(
            backend=self.backend,
            url=self.url,
            receive_timeout=self.receive_timeout,
            bsread_receive_timeout_ms=self.bsread_receive_timeout_ms,
            auto_split=self.auto_split,
            late_threshold_s=self.late_threshold_s,
            **self._stream_kwargs,
        )

    # ------------------------------------------------------------------
    # Channel registration (called by EventWorker)
    # ------------------------------------------------------------------

    def register_source(self, source_id):
        if source_id in ("lab_time", "pulse_id"):
            return
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        if source_id in ("lab_time", "pulse_id"):
            return
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def get_all_source_ids(self):
        """Return all currently available channel names from the backend."""
        try:
            if _use_redis(self.backend):
                if RedisStream is not None:
                    url = self.url or RedisStream.DEFAULT_URL
                    tmp = RedisStream.__new__(RedisStream)
                    tmp.host, tmp.port = url.split(":") if ":" in url else (url, 6379)
                    return tmp.search() or []
            else:
                # bsread dispatcher
                from datahub import Dispatcher
                d = Dispatcher(url=self.url)
                return d.search() or []
        except Exception as exc:
            _logger.warning("datahub channel search failed: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Context manager (called by EventWorker.eventLoop)
    # ------------------------------------------------------------------

    def context_manager(self):
        """Create and start the appropriate datahub stream."""
        if not self.source_ids:
            # No channels registered yet — yield NullEvents until some arrive
            self.stream = None
            return DataHubEventSourceContext(self)

        channels = list(self.source_ids)
        kwargs = _bsread_kwargs(self)

        try:
            if _use_redis(self.backend):
                url = self.url or RedisStream.DEFAULT_URL
                self.stream = RedisStream(channels=channels, url=url, **kwargs)
                _logger.debug("DataHubEventHandler: Redis stream on %s, channels=%s", url, channels)
            else:
                # bsread dispatcher (url=None: bsread's default dispatcher), one
                # connection per delay group; the context opens them.
                return _GroupedDispatcherContext(self)
        except Exception as exc:
            _logger.error("Failed to create datahub stream (%s): %s", self.backend, exc)
            self.stream = None

        return DataHubEventSourceContext(self)


# ---------------------------------------------------------------------------
# DataHubLocalEventHandler — local test streams (host:port bsread)
# ---------------------------------------------------------------------------

class DataHubLocalEventHandler:
    """Event handler for local / direct bsread streams.

    Connects to a ``host:port`` bsread sender (e.g. the synthetic test stream
    started by :class:`~escape.stream.TestStream`) using datahub's
    ``BsreadStream`` with a direct URL.  No dispatcher is involved and all
    channels broadcast by the sender are received automatically.

    Because the sender broadcasts all channels unconditionally, the event loop
    does **not** need to restart when new :class:`~escape.stream.Stream`
    objects are registered — they automatically receive data from the next event.

    Parameters
    ----------
    host : str
        Sender hostname (default ``'localhost'``).
    port : int
        Sender port (default ``9999``).
    receive_timeout : float
        Seconds per ``get_event()`` call (default 0.5 s).
    bsread_receive_timeout_ms : int, optional
        As for :class:`DataHubEventHandler`; ``None`` keeps datahub's 3000 ms.
    **stream_kwargs
        Forwarded to ``BsreadStream`` (e.g. ``queue_size=200``).
    """

    _needs_restart_on_register = False  # channels=[] receives all; no restart needed

    def __init__(self, host="localhost", port=9999, receive_timeout=0.5,
                 bsread_receive_timeout_ms=None, **stream_kwargs):
        _require_datahub()
        self.host = host
        self.port = port
        self.receive_timeout = receive_timeout
        self.bsread_receive_timeout_ms = bsread_receive_timeout_ms
        self._stream_kwargs = stream_kwargs
        self.stream = None
        self.source_ids = []

    def clone(self):
        """See ``DataHubEventHandler.clone``."""
        return DataHubLocalEventHandler(
            host=self.host, port=self.port, receive_timeout=self.receive_timeout,
            bsread_receive_timeout_ms=self.bsread_receive_timeout_ms,
            **self._stream_kwargs,
        )

    def register_source(self, source_id):
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def get_all_source_ids(self):
        """No discovery API for a direct sender; rely on live event keys instead.

        ``EventWorker._last_event_keys`` (populated from received events) is
        the accurate source of truth for a local stream and is what
        ``StreamSession.available()`` prefers; this is only the fallback.
        """
        return []

    def context_manager(self):
        """Connect to ``host:port`` and stream all channels."""
        url = f"{self.host}:{self.port}"
        kwargs = _bsread_kwargs(self)
        kwargs.setdefault("queue_size", 100)
        # The bsread Sender uses ZMQ PUSH; direct receivers must use PULL mode.
        # Datahub defaults to SUB (for dispatcher connections), so we override it.
        kwargs.setdefault("mode", "PULL")
        # channels=[] → datahub forwards all channels from the bsread data header.
        try:
            self.stream = BsreadStream(channels=[], url=url, **kwargs)
            _logger.debug("DataHubLocalEventHandler: BsreadStream on %s (PULL)", url)
        except Exception as exc:
            _logger.error("Failed to create local datahub stream: %s", exc)
            self.stream = None
        return DataHubEventSourceContext(self)


# ---------------------------------------------------------------------------
# MultiSourceEventHandler — merge events from several backends by pulse_id
# ---------------------------------------------------------------------------

class _PulseIdMerger:
    """Thread-safe pulse_id aligner for N independent event sources.

    Each source calls ``add(source_idx, pulse_id, timestamp_ns, msg_dict)``
    from its own thread.  Once all *n_sources* have contributed to a pulse_id
    (or ``timeout_pulses`` newer pulses have arrived, or the buffer overflows)
    the merged message is forwarded via ``callback(pulse_id, timestamp_ns, msg)``.
    """

    def __init__(self, n_sources, callback, buffer_size=200, timeout_pulses=50):
        self.n_sources = n_sources
        self.callback = callback
        self.buffer_size = buffer_size
        self.timeout_pulses = timeout_pulses
        self._data = {}   # {pulse_id: {'_ts': ns, '_nsrc': int, **channels}}
        self._lock = threading.Lock()

    def add(self, source_idx, pulse_id, timestamp_ns, msg):
        with self._lock:
            if pulse_id not in self._data:
                self._data[pulse_id] = {'_ts': timestamp_ns, '_nsrc': 0}
            entry = self._data[pulse_id]
            entry['_nsrc'] += 1
            entry.update(msg)
            self._flush()

    def _flush(self):
        if not self._data:
            return
        sorted_ids = sorted(self._data)
        latest_id = sorted_ids[-1]
        fire = []
        for pid in sorted_ids:
            entry = self._data[pid]
            complete = entry['_nsrc'] >= self.n_sources
            stale = (latest_id - pid) >= self.timeout_pulses
            overflow = len(self._data) > self.buffer_size
            if complete or stale or (overflow and pid != latest_id):
                fire.append(pid)
        for pid in sorted(fire):
            entry = self._data.pop(pid, None)
            if entry is None:
                continue
            ts = entry.pop('_ts', 0)
            entry.pop('_nsrc', None)
            try:
                self.callback(pid, ts, dict(entry))
            except Exception as exc:
                _logger.warning("_PulseIdMerger callback error: %s", exc)


class MultiSourceContext:
    """Context manager that drives N sub-handler contexts in background threads.

    Merged events are placed in a ``queue.Queue``; ``get_event()`` drains it
    with a timeout and returns a :class:`DataHubEvent` or :class:`NullEvent`.
    """

    def __init__(self, handler):
        self._handler = handler
        self._queue = queue.Queue(maxsize=handler.queue_size)
        self._stop = threading.Event()
        self._merger = _PulseIdMerger(
            n_sources=len(handler.handlers),
            callback=self._on_merged,
            timeout_pulses=handler.timeout_pulses,
        )
        self._contexts = []
        self._threads = []

    def __enter__(self):
        for i, h in enumerate(self._handler.handlers):
            ctx = h.context_manager()
            ctx.__enter__()
            self._contexts.append(ctx)
            t = threading.Thread(
                target=self._recv_loop,
                args=(i, ctx),
                daemon=True,
                name=f"escape-multisrc-{i}",
            )
            t.start()
            self._threads.append(t)
        return self

    def _recv_loop(self, source_idx, ctx):
        backoff = 1.0
        while not self._stop.is_set():
            try:
                event = ctx.get_event()
            except Exception as exc:
                # A sub-stream died (e.g. its datahub receive thread ended):
                # reconnect just that one, as EventWorker._serve does for a
                # single handler, instead of ending this thread.
                print(f"MultiSourceEventHandler: source {source_idx} reconnecting in {backoff:.0f}s (error: {exc})")
                try:
                    ctx.__exit__(None, None, None)
                except Exception as exit_exc:
                    _logger.debug("Error closing sub-context: %s", exit_exc)
                while True:
                    if self._stop.wait(backoff):
                        return
                    backoff = min(backoff * 2, 30.0)
                    try:
                        ctx = self._handler.handlers[source_idx].context_manager()
                        ctx.__enter__()
                        break
                    except Exception as conn_exc:
                        _logger.debug("Reconnect of source %d failed: %s", source_idx, conn_exc)
                self._contexts[source_idx] = ctx
                continue
            pid = event.getEventId()
            if pid is not None:
                backoff = 1.0
            if pid is not None and hasattr(event, '_msg') and hasattr(event, '_ts_ns'):
                self._merger.add(source_idx, pid, event._ts_ns, event._msg)

    def _on_merged(self, pulse_id, timestamp_ns, msg):
        try:
            self._queue.put_nowait(DataHubEvent(pulse_id, timestamp_ns, msg))
        except queue.Full:
            pass

    def get_event(self):
        try:
            return self._queue.get(timeout=self._handler.receive_timeout)
        except queue.Empty:
            return NullEvent()

    def __exit__(self, *_):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for ctx in self._contexts:
            try:
                ctx.__exit__(None, None, None)
            except Exception as exc:
                _logger.debug("Error closing sub-context: %s", exc)
        return False


class MultiSourceEventHandler:
    """Merge live events from multiple datahub backends by pulse_id.

    Each sub-handler streams its assigned channels in an independent background
    thread.  Messages with the same pulse_id are merged into a single dict
    before delivery.  Partial events are held until either all sources have
    contributed or *timeout_pulses* newer pulse IDs have arrived.

    This is most useful when channels live on different backends — for example
    bsread channels from the dispatcher **plus** special channels from a local
    sender or a second Redis instance.

    Parameters
    ----------
    *handlers : DataHubEventHandler | DataHubLocalEventHandler
        Two or more configured handler instances.
    timeout_pulses : int
        Pulse IDs to wait before forwarding a partial event.
        At 100 Hz, the default of 50 corresponds to ≈ 0.5 s.
    queue_size : int
        Depth of the internal merged-event queue.
    receive_timeout : float
        Seconds per ``get_event()`` call (propagated to sub-handlers).

    Example
    -------
    ::

        from escape.stream import (
            Stream, EventWorker,
            DataHubEventHandler, DataHubLocalEventHandler, MultiSourceEventHandler,
            TestStream,
        )

        # Start a local test stream on port 9999
        ts = TestStream(port=9999)
        ts.start()

        # Merge the local test stream with (hypothetical) live dispatcher channels
        ew = EventWorker(MultiSourceEventHandler(
            DataHubLocalEventHandler(host='localhost', port=9999),
            DataHubEventHandler(backend='bsread'),   # live SwissFEL channels
        ))

        # All channels are now available in the same event:
        i0      = Stream('i0',                    ew)   # from local stream
        energy  = Stream('SARUN18-UIND030:FELPHOTENE', ew)   # from dispatcher

    For a fully synthetic demo using a *single* local stream see
    ``example_local_stream.ipynb``.
    """

    _needs_restart_on_register = False  # sub-handlers receive all channels; no restart needed

    def __init__(self, *handlers, timeout_pulses=50, queue_size=200, receive_timeout=0.5):
        _require_datahub()
        if len(handlers) < 2:
            raise ValueError("MultiSourceEventHandler needs at least two sub-handlers.")
        self.handlers = list(handlers)
        self.timeout_pulses = timeout_pulses
        self.queue_size = queue_size
        self.receive_timeout = receive_timeout
        # Propagate the receive_timeout to sub-handlers so their get_event()
        # calls don't block longer than our own timeout.
        for h in self.handlers:
            if hasattr(h, 'receive_timeout'):
                h.receive_timeout = min(h.receive_timeout, receive_timeout)
        self.source_ids = []

    def register_source(self, source_id):
        if source_id in ("lab_time", "pulse_id"):
            return
        for h in self.handlers:
            try:
                h.register_source(source_id)
            except Exception:
                pass
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        for h in self.handlers:
            try:
                h.remove_source(source_id)
            except Exception:
                pass
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def get_all_source_ids(self):
        """Union of every sub-handler's discoverable channels."""
        ids = []
        for h in self.handlers:
            try:
                for sid in h.get_all_source_ids():
                    if sid not in ids:
                        ids.append(sid)
            except Exception as exc:
                _logger.warning("get_all_source_ids failed for %s: %s", type(h).__name__, exc)
        return ids

    def context_manager(self):
        return MultiSourceContext(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _use_redis(backend):
    """Return True if the redis backend should be used."""
    if backend == "redis":
        return True
    if backend == "auto":
        return _HAS_REDIS and RedisStream is not None
    return False
