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

    def __init__(self, pulse_id, timestamp_ns, msg):
        self._id = pulse_id
        self._ts_ns = timestamp_ns   # nanoseconds
        self._msg = msg              # {channel_name: value}

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
            return NullEvent()
        pulse_id, timestamp_ns, msg = result
        return DataHubEvent(pulse_id, timestamp_ns, msg)

    def __exit__(self, *_):
        stream = self.eventhandler.stream
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                _logger.debug("Error closing datahub stream: %s", exc)
        self.eventhandler.stream = None
        return False


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
    **stream_kwargs
        Extra keyword arguments forwarded to ``BsreadStream`` / ``RedisStream``
        (e.g. ``queue_size=200``, ``receive_timeout=1000`` for the bsread
        internal timeout in ms).
    """

    def __init__(self, backend="bsread", url=None, receive_timeout=0.5, **stream_kwargs):
        _require_datahub()
        self.backend = backend
        self.url = url
        self.receive_timeout = receive_timeout
        self._stream_kwargs = stream_kwargs
        self.stream = None
        self.source_ids = []

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
        kwargs = dict(self._stream_kwargs)

        try:
            if _use_redis(self.backend):
                url = self.url or RedisStream.DEFAULT_URL
                self.stream = RedisStream(channels=channels, url=url, **kwargs)
                _logger.debug("DataHubEventHandler: Redis stream on %s, channels=%s", url, channels)
            else:
                # bsread dispatcher: url=None → bsread uses its own default dispatcher
                url = self.url  # None means "use bsread default dispatcher URL"
                self.stream = BsreadStream(channels=channels, url=url, **kwargs)
                _logger.debug("DataHubEventHandler: BsreadStream via dispatcher, channels=%s", channels)
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
    **stream_kwargs
        Forwarded to ``BsreadStream`` (e.g. ``queue_size=200``).
    """

    _needs_restart_on_register = False  # channels=[] receives all; no restart needed

    def __init__(self, host="localhost", port=9999, receive_timeout=0.5, **stream_kwargs):
        _require_datahub()
        self.host = host
        self.port = port
        self.receive_timeout = receive_timeout
        self._stream_kwargs = stream_kwargs
        self.stream = None
        self.source_ids = []

    def register_source(self, source_id):
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def context_manager(self):
        """Connect to ``host:port`` and stream all channels."""
        url = f"{self.host}:{self.port}"
        kwargs = dict(self._stream_kwargs)
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
        while not self._stop.is_set():
            event = ctx.get_event()
            pid = event.getEventId()
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
