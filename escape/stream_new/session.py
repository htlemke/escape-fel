"""
StreamSession — interactive live-stream session object.

A ``StreamSession`` is the live-data equivalent of a scan-run object in
``escape.storage``.  Attribute or item access to a channel name creates and
auto-starts a :class:`~escape.stream_new.Stream`; no explicit
``accumulate(True)`` call is needed.

Quick start::

    ses = StreamSession('localhost:9999')   # local test / direct bsread
    # or
    ses = StreamSession('bsread')           # PSI dispatcher (current default)
    ses = StreamSession('redis')            # Redis/Dragonfly backend

    # Channel access — creates Stream and starts accumulation automatically
    i0   = ses.i0
    pump = ses.pump_on

    # Arithmetic creates derived streams (also auto-subscribed)
    ratio = ses.i / ses.i0

    # Single-line analysis — identical syntax to escape.Array
    import numpy as np
    bins = np.linspace(-1, 1, 21)
    ses.t.digitize(bins).categorize(ses.i / ses.i0).plot_med(update=0.5)

    # Inspect what is available on the backend
    ses.available()   # list of channels the backend is broadcasting

    # Inspect subscription status
    ses.status()      # prints a table: name, events, rate, ok/missing

    # Stop everything
    ses.stop()

How channel discovery works
---------------------------
For **local/direct streams** (``DataHubLocalEventHandler``), the EventWorker
receives all channels that the sender broadcasts.  After the first event the
worker populates ``eventworker._last_event_keys`` with the channel names it
saw.  ``StreamSession.available()`` reads those keys directly.

For **dispatcher streams** (``DataHubEventHandler``), ``available()`` calls
``DataHubEventHandler.get_all_source_ids()`` which queries the PSI bsread
dispatcher search API.  This returns the full list of published channels (which
can be large; filter with ``available(pattern='SARBD*')`` if needed).

A channel that does not exist in the backend will produce a ``Stream`` with
0 events.  ``ses.missing()`` returns the names of all such empty streams.
"""

import time
import threading

_RESERVED = frozenset({
    "_ew", "_streams", "_auto_accumulate",
    "available", "status", "missing", "stop", "streams",
})


class StreamSession:
    """Interactive session object — attribute access creates auto-subscribing Streams.

    Parameters
    ----------
    source : str or EventHandler
        A connection description:

        - ``'localhost:9999'`` — local bsread sender at the given host:port
        - ``'hostname'`` — local bsread at that host, default port 9999
        - ``'bsread'`` / ``'redis'`` / ``'auto'`` — datahub backend keyword
        - any ``DataHubEventHandler``, ``DataHubLocalEventHandler``,
          or ``MultiSourceEventHandler`` instance
        - ``None`` — uses the module default ``EventWorker`` if one is registered,
          otherwise creates a ``DataHubEventHandler(backend='bsread')``
    auto_accumulate : bool
        If ``True`` (default) every stream created via attribute access
        immediately starts accumulating.
    make_default : bool
        Register the internal ``EventWorker`` as the module default so that
        ``Stream('ch')`` with no worker argument also works.
    **handler_kwargs
        Extra keyword arguments forwarded to the handler constructor.
    """

    def __init__(self, source=None, auto_accumulate=True, make_default=True, **handler_kwargs):
        from .escape_stream_new import EventWorker
        from .es_wrappers_datahub import (
            DataHubEventHandler, DataHubLocalEventHandler, _HAS_DATAHUB,
        )
        from .es_wrappers_new import EventHandler_SFEL

        if source is None:
            # Re-use existing default EventWorker if present, else create one.
            import escape.stream_new.escape_stream_new as _esn
            existing = _esn.__dict__.get("eventworker")
            if existing is not None:
                ew = existing
            elif _HAS_DATAHUB:
                ew = EventWorker(DataHubEventHandler(backend="bsread", **handler_kwargs), make_default=make_default)
            else:
                ew = EventWorker(EventHandler_SFEL(**handler_kwargs), make_default=make_default)
        elif hasattr(source, "context_manager"):
            # Already a handler instance
            ew = EventWorker(source, make_default=make_default)
        elif isinstance(source, str):
            if source in ("bsread", "redis", "auto"):
                if not _HAS_DATAHUB:
                    raise RuntimeError("psi-datahub not installed; cannot use backend keyword.")
                ew = EventWorker(DataHubEventHandler(backend=source, **handler_kwargs), make_default=make_default)
            else:
                # Parse 'host:port' or bare 'hostname'
                if not _HAS_DATAHUB:
                    raise RuntimeError("psi-datahub not installed; cannot create DataHubLocalEventHandler.")
                if ":" in source:
                    host, port_str = source.rsplit(":", 1)
                    port = int(port_str)
                else:
                    host = source
                    port = 9999
                ew = EventWorker(
                    DataHubLocalEventHandler(host=host, port=port, **handler_kwargs),
                    make_default=make_default,
                )
        else:
            raise TypeError(f"source must be a string or EventHandler instance, not {type(source).__name__}")

        object.__setattr__(self, "_ew", ew)
        object.__setattr__(self, "_streams", {})
        object.__setattr__(self, "_auto_accumulate", auto_accumulate)

    # ------------------------------------------------------------------
    # Channel access
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        if name.startswith("_") or name in _RESERVED:
            raise AttributeError(name)
        return self._get_or_create(name)

    def __getitem__(self, name):
        return self._get_or_create(name)

    def _get_or_create(self, name):
        streams = object.__getattribute__(self, "_streams")
        if name in streams:
            return streams[name]
        from .escape_stream_new import Stream
        ew = object.__getattribute__(self, "_ew")
        aa = object.__getattribute__(self, "_auto_accumulate")
        s = Stream(name, ew)
        streams[name] = s
        if aa:
            s.accumulate(True)
        return s

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available(self, pattern=None, timeout=2.0):
        """Return the channel names the backend is currently broadcasting.

        For local/direct streams the list comes from the first received event;
        a short wait is applied if the event loop has not received anything yet.
        For dispatcher streams the PSI channel search API is queried.

        Parameters
        ----------
        pattern : str, optional
            Substring filter applied to the returned names.
        timeout : float
            Seconds to wait for the first event if the event loop just started.
        """
        ew = object.__getattribute__(self, "_ew")
        # Prefer live-event keys (accurate for local streams)
        if not ew._last_event_keys and timeout > 0:
            deadline = time.time() + timeout
            while not ew._last_event_keys and time.time() < deadline:
                time.sleep(0.05)
        if ew._last_event_keys:
            keys = ew._last_event_keys
        else:
            keys = ew._eventHandler.get_all_source_ids()
        if pattern:
            keys = [k for k in keys if pattern in k]
        return sorted(keys)

    def missing(self):
        """Return names of subscribed streams that have received no events yet.

        These are likely channel names that do not exist on the backend, or
        channels that have not been triggered since accumulation started.
        """
        streams = object.__getattribute__(self, "_streams")
        return [name for name, s in streams.items() if len(s) == 0]

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def status(self):
        """Print a table showing all subscribed streams and their event counts."""
        streams = object.__getattribute__(self, "_streams")
        ew = object.__getattribute__(self, "_ew")
        if not streams:
            print("StreamSession — no channels subscribed yet.")
            print(f"  Backend: {type(ew._eventHandler).__name__}")
            print("  Call ses.available() to see what channels the backend provides.")
            return
        print(f"StreamSession — {type(ew._eventHandler).__name__}")
        print(f"  Loop running: {ew.loopThread is not None and ew.loopThread.is_alive()}")
        print(f"  Rate:         {ew.runningFrequency:.1f} Hz")
        print()
        print(f"  {'Channel':<24} {'Events':>8}  {'Status'}")
        print(f"  {'─'*24} {'─'*8}  {'─'*10}")
        for name, s in streams.items():
            n = len(s)
            acc = "● live" if s._is_accumulating() else "○ stopped"
            warn = "  ⚠ no data" if n == 0 else ""
            print(f"  {name:<24} {n:>8}  {acc}{warn}")

    def __repr__(self):
        streams = object.__getattribute__(self, "_streams")
        ew = object.__getattribute__(self, "_ew")
        alive = ew.loopThread is not None and ew.loopThread.is_alive()
        return (
            f"StreamSession("
            f"handler={type(ew._eventHandler).__name__}, "
            f"channels={len(streams)}, "
            f"loop={'running' if alive else 'stopped'})"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def streams(self):
        """Dict of all streams created in this session."""
        return dict(object.__getattribute__(self, "_streams"))

    def stop(self):
        """Stop accumulation on all streams and shut down the EventWorker."""
        streams = object.__getattribute__(self, "_streams")
        for s in streams.values():
            try:
                s.accumulate(False)
            except Exception:
                pass
        ew = object.__getattribute__(self, "_ew")
        ew.stopEventLoop()


# ---------------------------------------------------------------------------
# gather() — timed multi-stream acquisition
# ---------------------------------------------------------------------------

def gather(*streams, seconds=0, n_events=None):
    """Accumulate one or more Streams for a fixed duration or event count.

    Starts accumulation on all passed streams, waits, then stops accumulation.
    Returns a dict ``{stream.name: stream}`` for convenient access.

    Parameters
    ----------
    *streams : Stream
        One or more :class:`~escape.stream_new.Stream` objects.
    seconds : float
        Duration to accumulate (default 0 — starts and returns immediately).
    n_events : int, optional
        Accumulate until at least this many events are in the first stream.
        If both *seconds* and *n_events* are given, whichever triggers first
        stops acquisition.

    Examples
    --------
    ::

        # 5-second burst on three streams
        data = gather(i0, ratio, t, seconds=5)
        data['ratio'].plot_hist()

        # Single stream as context manager:
        with i0:          # accumulate(True) on enter, accumulate(False) on exit
            time.sleep(5)

        # Multiple streams with contextlib:
        from contextlib import ExitStack
        with ExitStack() as stack:
            for s in [i0, i, t, pump]:
                stack.enter_context(s)
            time.sleep(5)
    """
    from .escape_stream_new import Stream
    for s in streams:
        if not isinstance(s, Stream):
            raise TypeError(f"gather() expects Stream objects; got {type(s).__name__}")
        s.accumulate(True)

    if seconds > 0 and n_events is None:
        time.sleep(seconds)
    elif n_events is not None:
        deadline = time.time() + (seconds if seconds > 0 else 1e9)
        first = streams[0] if streams else None
        while first is not None and len(first) < n_events and time.time() < deadline:
            time.sleep(0.05)

    for s in streams:
        s.accumulate(False)

    return {s.name: s for s in streams}
