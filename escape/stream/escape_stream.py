"""
Event Synchronous Categorisation And Processing Environment — new generation.

Improvements over escape_stream.py:
  - EventWorker uses threading.Event for clean stop signalling; fixes the
    deprecated Thread.isAlive() call (now is_alive()).
  - The event loop does not crash when no channels are registered — it simply
    yields NullEvents until a channel is added.
  - Stream API mirrors escape.Array:
        Stream('channel', ew)                      short constructor
        i[pump]                                    filter by boolean stream
        i[~pump]                                   logical NOT filter
        t.digitize(bins).categorize(i / i0)        bin by another channel
        stream.categorize(other)                   share scan structure
  - StreamContext: a context manager (and figure-close hook) that controls
    which Stream objects are accumulating.  Create it with
        ctx = StreamContext(i0, intensity)
    or use the per-object shorthand
        ctx = i0.acquire()
    then:
        with ctx:               # accumulate for the duration of the block
            ...
        ctx.tie_to_figure(fig)  # auto-stop when the figure window closes
  - EscData is kept as a backward-compatible alias for Stream.
"""

import threading
import time
import operator
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from .es_wrappers import EventHandler_SFEL, LocalEventHandler
from . import plots
from . import tools
from .testStream import createStream

try:
    from .es_wrappers_datahub import DataHubEventHandler, DataHubLocalEventHandler
    _HAS_DATAHUB = True
except Exception:
    _HAS_DATAHUB = False


# ---------------------------------------------------------------------------
# TestStream helper
# ---------------------------------------------------------------------------

class TestStream:
    """Manage a synthetic bsread test stream in a child process."""

    def __init__(self, port=9999, interval=0.01):
        self.port = port
        self.interval = interval
        self._process = None

    def start(self):
        if self._process is not None and self._process.is_alive():
            print("Test stream already running.")
            return
        from multiprocessing import Process
        self._process = Process(
            target=createStream,
            kwargs=dict(port=self.port, interval=self.interval),
            daemon=True,
        )
        self._process.start()
        print(f"Test stream started on port {self.port}.")

    def stop(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join(timeout=3)
            self._process = None
            print("Test stream stopped.")

    def __repr__(self):
        alive = self._process is not None and self._process.is_alive()
        return f"TestStream(port={self.port}, running={alive})"


# ---------------------------------------------------------------------------
# StreamContext — acquisition lifetime manager
# ---------------------------------------------------------------------------

class StreamContext:
    """Control accumulation lifetime for one or more Stream objects.

    Usage
    -----
    # as a plain context manager:
    with StreamContext(i0, intensity):
        time.sleep(10)

    # tied to a matplotlib figure:
    ctx = StreamContext(i0, intensity)
    ctx.tie_to_figure(fig)   # accumulation stops when the figure is closed
    ctx.start()

    # per-object shorthand:
    with i0.acquire():
        ...
    """

    def __init__(self, *stream_objects):
        self._objects = list(stream_objects)

    def start(self):
        for obj in self._objects:
            obj.accumulate(True)

    def stop(self):
        for obj in self._objects:
            obj.accumulate(False)

    def tie_to_figure(self, fig):
        """Stop accumulation when *fig* is closed."""
        fig.canvas.mpl_connect("close_event", lambda _e: self.stop())
        return self

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def add(self, stream):
        self._objects.append(stream)


# ---------------------------------------------------------------------------
# Scan — lightweight per-step bookkeeping
# ---------------------------------------------------------------------------

class Scan:
    """Partitions a live Stream's events into steps keyed by one or more parameters.

    ``parameters`` is normally one or more other ``Stream`` objects (e.g. from
    ``key_stream.digitize(bins).categorize(other)``), but each parameter only
    needs to duck-type ``.name`` and ``._getEventData()`` — it does not have to
    be a real bsread-backed ``Stream``. A small hand-written object with a
    mutable ``.value`` returned from ``_getEventData()`` works today as a way
    to bin live data by an *externally driven* state (e.g. "which step of an
    external scan loop is currently open") rather than by another channel's
    live value::

        class StepIndexSource:
            name = "step_index"
            def __init__(self):
                self.value = 0
            def _getEventData(self):
                return self.value

        step_src = StepIndexSource()
        shared_scan = Scan(parameters=[step_src])   # open, dynamic bins
        binned = Stream(source=raw_stream._source, scan=shared_scan)
        # elsewhere: step_src.value = next_step_index

    Multiple ``Stream``s may safely share one such open-ended, growing
    ``Scan`` (each gets its own ``DataManager``, which grows its own per-step
    buffers to match the shared step index regardless of which Stream's data
    reaches a given step first).
    """

    def __init__(self, parameters=None, values=None, precision=None, sortValues=True):
        self._parameters = parameters
        self._sortValues = sortValues

        if parameters is None:
            self._parameterNames = None
            self._precision = None
            self._values = [None]
        else:
            self._parameterNames = [tp.name for tp in parameters]
            self._applyPrecision(precision or {})
            self._values = values if values is not None else []

    def _applyPrecision(self, precision):
        if isinstance(precision, np.ndarray):
            self._precision = precision
        else:
            self._precision = np.zeros(len(self._parameters))
            for key, val in precision.items():
                if key in self._parameterNames:
                    self._precision[self._parameterNames.index(key)] = val

    def _roundParameters(self, parValues):
        pv = np.asarray(parValues, dtype=float)
        ind = self._precision.nonzero()[0]
        if len(ind):
            pv[ind] = np.round(pv[ind] / self._precision[ind]) * self._precision[ind]
        return tuple(pv)

    def _isValid(self):
        if self._parameters is None:
            return True
        parValues = [tp._getEventData() for tp in self._parameters]
        return not np.isnan(parValues).any()

    def _append(self):
        """Return (is_new_step, step_index).  None, None if scan values are invalid."""
        if self._parameters is None:
            return False, 0
        parValues = [tp._getEventData() for tp in self._parameters]
        if np.isnan(parValues).any():
            return None, None
        parValues = self._roundParameters(parValues)
        if parValues in self._values:
            return False, self._values.index(parValues)
        self._values.append(parValues)
        return True, len(self._values) - 1

    def keys(self):
        return self._parameterNames

    def copy(self):
        return Scan(
            parameters=self._parameters,
            values=list(self._values),
            precision=self._precision,
        )

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        try:
            arr = np.asarray(self._values)
            if isinstance(item, (slice, int)):
                return arr.T[item]
            if isinstance(item, str):
                return arr.T[self._parameterNames.index(item)]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# DataManager — per-step circular buffers
# ---------------------------------------------------------------------------

class DataManager:
    def __init__(self, data=None, eventIds=None, maxlen=1000, scan=None):
        if scan is None:
            scan = Scan()
        self.scan = scan
        if data is None:
            self._data = [deque(maxlen=maxlen) for _ in range(len(scan._values))]
            self._eventIds = [deque(maxlen=maxlen) for _ in range(len(scan._values))]
        else:
            self._data = data
            self._eventIds = eventIds
        # True, uncapped per-step event counts -- deliberately tracked separately
        # from len(self._data[step]), which is bounded by each deque's maxlen and
        # so silently stops growing (reporting a flat "maxlen" plateau) once a step
        # has received more than maxlen events. lens() below is the "how many
        # samples are currently retained for stats" view (matches what mean()/
        # median()/centerPerc() actually average over); counts() is the "how many
        # events has this step *ever* received" view that HistPlot needs.
        self._counts = [0] * len(self._data)
        self._lastEventId = None

    def append(self, data, eventId, index=None):
        if eventId is None or eventId == self._lastEventId:
            return
        self._lastEventId = eventId
        if index is None:
            doappend, index = self.scan._append()
            if doappend is None:  # invalid/NaN parameter values this event
                return
        # Grow *this* DataManager's own lists to cover `index`, rather than
        # trusting `doappend` (whether the *shared* scan._values grew) -- when
        # multiple Streams share one growing Scan, whichever Stream's data
        # reaches a new step first grows scan._values on behalf of all of
        # them, so a later Stream would see doappend=False for a step its own
        # _data/_eventIds haven't been extended to yet, and index it out of
        # range.
        maxlen = self._data[0].maxlen if self._data else 1000
        while len(self._data) <= index:
            self._data.append(deque(maxlen=maxlen))
            self._eventIds.append(deque(maxlen=maxlen))
            self._counts.append(0)
        step_data = self._data[index]
        if step_data and np.shape(data) != np.shape(step_data[-1]):
            # The channel's per-event shape changed mid-accumulation (e.g. a
            # detector ROI count got reconfigured) -- mean()/std()/median()/
            # centerPerc() etc. all eventually do np.asarray(step_data), which
            # raises an inhomogeneous-shape ValueError once both the old and
            # new shape are present in the same step. Drop the stale,
            # differently-shaped samples rather than let that happen; this
            # step's stats just have fewer samples right after the change,
            # the same way a step naturally does right after it opens.
            step_data.clear()
            self._eventIds[index].clear()
        step_data.append(data)
        self._eventIds[index].append(eventId)
        self._counts[index] += 1

    def _getDataShape(self):
        lens = self.lens()
        if not lens or max(lens) == 0:
            return None
        best = lens.index(max(lens))
        return np.shape(list(self._data[best])[0])

    def __len__(self):
        return sum(self.lens())

    def lens(self):
        """Number of samples currently retained per step (bounded by maxlen).

        This is the sample count actually behind mean()/median()/centerPerc()
        -- once a step exceeds maxlen events, older samples are evicted and
        this stays capped at maxlen. For the true, uncapped per-step event
        count, use counts() instead.
        """
        result = []
        for n, (te, td) in enumerate(zip(self._eventIds, self._data)):
            if len(te) == len(td):
                result.append(len(te))
            else:
                print(f"DataManager: mismatch in step {n}")
                result.append(min(len(te), len(td)))
        return result

    def counts(self):
        """True, uncapped number of events ever appended to each step."""
        return list(self._counts)

    data = property(lambda self: self._data)
    eventIds = property(lambda self: self._eventIds)


# ---------------------------------------------------------------------------
# Event sources
# ---------------------------------------------------------------------------

class EventSource:
    """Wraps a named bsread channel."""

    def __init__(self, sourceId, eventWorker, unit="a.u."):
        self.name = sourceId
        self.unit = unit
        if eventWorker is None and "eventworker" in globals():
            eventWorker = globals()["eventworker"]
        self.eventWorker = eventWorker

    def getEventData(self):  # noqa: N802
        return self.eventWorker.event.getFromSource(self.name)


class FilteredEventSource:
    """Event source that only yields data when a mask stream is truthy.

    Used by ``Stream.filter()`` and ``Stream.__getitem__(mask_stream)``.
    The mask is evaluated from the current event, so the mask stream does not
    need to be separately accumulating (though its bsread channel must be
    subscribed; for LocalEventHandler all channels are always received).
    """

    def __init__(self, source, mask_stream, inner_stream=None):
        self._inner = source
        self._mask = mask_stream
        # The Stream `source` was taken from, when known (Stream.filter() always
        # passes it) -- kept alongside the raw `_inner` Source so graph.py can
        # walk the *Stream* graph (picking up e.g. its _graph_parent) instead of
        # just the raw channel(s) `_inner` ultimately resolves to. Optional/None
        # for FilteredEventSource instances built by hand from a bare Source.
        self._inner_stream = inner_stream
        # Include mask name in channel label for clarity
        self.name = f"{source.name}[{mask_stream.name}]"
        self.unit = source.unit
        # The eventWorker for a filtered stream is the same as the data stream's
        self.eventWorker = source.eventWorker

    def getEventData(self):  # noqa: N802
        mask_val = self._mask._getEventData()
        if mask_val:
            return self._inner.getEventData()
        return None


class ProcSource:
    """Source backed by a processing pipeline node."""

    def __init__(self, procObj, eventWorker, returnIndex=0, name=None, unit="a.u."):
        self.name = name or "none"
        self.unit = unit
        self.eventWorker = eventWorker
        self.procObj = procObj
        self.returnIndex = returnIndex

    def getEventData(self):  # noqa: N802
        if self.procObj.getEventData():
            self.procObj.updateChildren(self)
        # ret_values is still None before the very first successful compute
        # (e.g. its own dependencies were invalid/None on every event so far,
        # as with a running_mean over a not-yet-True filter mask) -- report
        # "no data yet" the same way any other source does, rather than
        # crashing on ret_values[self.returnIndex].
        if self.procObj.ret_values is None:
            return None
        return self.procObj.ret_values[self.returnIndex]


class FileSource:
    """Placeholder for indexed file sources."""
    pass


def _collect_source_channels(source, seen=None):
    """Recursively collect the real, subscribable channel names a Source
    ultimately depends on.

    A plain :class:`EventSource`'s ``.name`` *is* a real channel. A
    :class:`FilteredEventSource` or :class:`ProcSource`'s ``.name`` is a
    synthetic display string (e.g. ``"i[pump_on]"`` or
    ``"SAR-CVME-TIFALL5:EvtSet[25]"``), not something the transport can
    subscribe to -- registering it as a channel would ask the dispatcher for
    a channel that doesn't exist. Those recurse into whatever real channel(s)
    actually feed them instead, so ``derived.accumulate(True)`` correctly
    subscribes every real channel the computation depends on, however deep.
    """
    if seen is None:
        seen = []
    if isinstance(source, EventSource):
        if source.name not in seen:
            seen.append(source.name)
    elif isinstance(source, FilteredEventSource):
        _collect_source_channels(source._inner, seen)
        _collect_source_channels(source._mask._source, seen)
    elif isinstance(source, ProcSource):
        proc = source.procObj
        for arg, is_esc in zip(proc.args, proc.args_is_esc):
            if is_esc:
                _collect_source_channels(arg._source, seen)
        for key, val in proc.kwargs.items():
            if proc.kwargs_is_esc.get(key):
                _collect_source_channels(val._source, seen)
    # FileSource / anything else: nothing registerable.
    return seen


# ---------------------------------------------------------------------------
# EventWorker — runs the bsread event loop in a background thread
# ---------------------------------------------------------------------------

class EventWorker:
    """Manages a background thread that drives bsread and calls registered callbacks.

    Parameters
    ----------
    eventHandler : EventHandler_SFEL or LocalEventHandler
        Handler that knows how to create the bsread Source context manager.
    make_default : bool
        If True, register this instance in the module globals so that
        EventSource() objects can find it without an explicit reference.
    restart_mode : "make_before_break" or "break_before_make"
        How a channel-set change (see ``registerSource``/``removeSource``)
        replaces the underlying connection. ``"make_before_break"``
        (default) connects and warms up a replacement *before* tearing down
        the current one -- no gap in received events, and the old
        connection's teardown (which, for some handlers, is itself not
        perfectly clean -- e.g. a known ordering bug in psi-datahub's
        ``Bsread.close()``) happens only after the replacement is already
        serving. This briefly holds two live connections open at once
        (roughly the time to construct+enter the new one, typically well
        under a second) -- pass ``"break_before_make"`` to opt out of that
        entirely and go back to the old stop-then-start behavior (a real
        gap in received events, but never more than one connection open).
        Falls back to ``"break_before_make"`` automatically, per restart,
        if the current handler doesn't support ``.clone()`` or the
        replacement doesn't come up within ``connect_timeout``
        (see ``_restart_make_before_break``) -- so this is safe to leave at
        the default even for a handler that doesn't support cloning yet.
    """

    def __init__(self, eventHandler=None, make_default=True, restart_mode="make_before_break"):
        if eventHandler is None:
            if _HAS_DATAHUB:
                eventHandler = DataHubEventHandler(backend="bsread")
            else:
                eventHandler = EventHandler_SFEL()
        self._eventHandler = eventHandler
        self.eventCallbacks = []
        self.sources = []
        self.event = None
        self._stop_event = threading.Event()
        self.loopThread = None
        self._lastTime = time.time()
        self.runningFrequency = 0.0
        self._last_event_keys = []       # channel names seen in the last event
        self._restart_timer = None
        self._restart_lock = threading.Lock()
        self._callback_failures = {}     # cb -> {"count": int, "last_log": float}
        self.restart_mode = restart_mode
        self._pulse_id_stream = None     # lazily-created, cached -- see .pulse_id
        self._lab_time_stream = None     # lazily-created, cached -- see .lab_time

        if make_default:
            globals()["eventworker"] = self
            print("EventWorker registered as module default.")

    def _needs_restart(self):
        """Return True if the handler needs a stop/restart when a source is added.

        Handlers that subscribe to explicit channel lists (e.g. bsread dispatcher)
        must be restarted so the new channel is included in the subscription.
        Handlers that receive all channels (local / multi-source) set
        ``_needs_restart_on_register = False`` to skip the expensive cycle.
        """
        return getattr(self._eventHandler, "_needs_restart_on_register", True)

    def _source_ids_snapshot(self):
        """``set`` of the handler's currently registered channel ids, or
        ``None`` if the handler doesn't expose ``source_ids`` in a
        comparable way (unknown handler type) -- see ``_changed``."""
        ids = getattr(self._eventHandler, "source_ids", None)
        return set(ids) if ids is not None else None

    def _changed(self, before):
        """Whether the handler's channel set actually differs from the
        ``before`` snapshot -- used to skip a restart entirely when a
        register/remove call was a no-op (e.g. re-registering an
        already-subscribed channel, or removing one a sibling Stream still
        needs). If the handler's channel set can't be introspected,
        conservatively assumes it changed (matches the old, unconditional
        behavior -- never a regression, just no new savings for that
        handler type)."""
        after = self._source_ids_snapshot()
        if before is None or after is None:
            return True
        return before != after

    def registerSource(self, sourceID):  # noqa: N802
        before = self._source_ids_snapshot()
        self._eventHandler.register_source(sourceID)
        if not self._changed(before):
            if not self.loopThread or not self.loopThread.is_alive():
                self.startEventLoop()
            return
        if self._needs_restart():
            self._schedule_restart()   # debounced: batches rapid channel additions
        elif not self.loopThread or not self.loopThread.is_alive():
            self.startEventLoop()

    def removeSource(self, sourceID):  # noqa: N802
        before = self._source_ids_snapshot()
        self._eventHandler.remove_source(sourceID)
        if not self._changed(before):
            return
        if self._needs_restart():
            self._schedule_restart()

    def registerSources(self, *sourceIDs):  # noqa: N802
        """Register several sources in one stop/start cycle (no extra debounce needed)."""
        before = self._source_ids_snapshot()
        for sid in sourceIDs:
            self._eventHandler.register_source(sid)
        if not self._changed(before):
            if not self.loopThread or not self.loopThread.is_alive():
                self.startEventLoop()
            return
        if self._needs_restart():
            self._schedule_restart()
        elif not self.loopThread or not self.loopThread.is_alive():
            self.startEventLoop()

    def _schedule_restart(self, delay=0.15):
        """Cancel any pending restart timer and arm a new one (debounce)."""
        with self._restart_lock:
            if self._restart_timer is not None:
                self._restart_timer.cancel()
            self._restart_timer = threading.Timer(delay, self._do_restart)
            self._restart_timer.daemon = True
            self._restart_timer.start()

    def _do_restart(self):
        with self._restart_lock:
            self._restart_timer = None
        if self.restart_mode == "make_before_break":
            if self._restart_make_before_break():
                return
            # Handler doesn't support .clone(), or the replacement didn't
            # come up in time -- fall back to the always-available path.
        self._restart_break_before_make()

    def _restart_break_before_make(self):
        """Stop the current connection, then start a new one.

        Simple and always available (no requirements on the handler), but
        there is a real gap in received events between the two -- and the
        old connection's teardown (whatever that costs for this handler --
        see the ``restart_mode`` docstring) happens with nothing yet
        replacing it.
        """
        self.stopEventLoop()
        self.startEventLoop()

    def _restart_make_before_break(self, connect_timeout=5.0):
        """Connect a replacement connection *before* tearing down the
        current one, then hand over sequentially (never both dispatching
        events at once -- see ``_serve``'s ``active_event`` gate).

        Returns True once the replacement is live and serving. Returns
        False if make-before-break wasn't possible at all for this restart
        (no usable ``.clone()``, or the replacement didn't come up in
        time), so the caller (``_do_restart``) falls back to running
        ``_restart_break_before_make`` itself.

        Not possible when the current handler has no ``.clone()`` (unknown
        handler type, or one -- like ``MultiSourceEventHandler`` -- that
        doesn't support it yet). Falls back the same way if the replacement
        doesn't finish connecting within ``connect_timeout``: holding a
        stuck half-connected replacement open indefinitely would be worse
        than just accepting the ordinary restart gap.

        This does not, and cannot, avoid an exception a misbehaving
        handler's own teardown raises on the *old* connection (e.g. the
        known ordering bug in psi-datahub's ``Bsread.close()``, which
        destroys its zmq context before joining its own background thread)
        -- that happens on the old connection's own shutdown regardless of
        whether a replacement was already ready. What this buys: no gap in
        event reception up to the point the old connection is stopped (the
        residual gap, while the old connection's thread actually exits, is
        bounded by its shutdown latency -- typically its ``receive_timeout``
        -- rather than a full reconnect), and the old connection's teardown
        runs only after the replacement is already serving, not instead of
        it.
        """
        old_handler = self._eventHandler
        clone_fn = getattr(old_handler, "clone", None)
        if clone_fn is None:
            return False
        try:
            new_handler = clone_fn()
        except Exception as exc:
            print(
                f"EventWorker: make-before-break clone() failed ({exc}); "
                "falling back to break-before-make for this restart."
            )
            return False

        for sid in list(getattr(old_handler, "source_ids", [])):
            new_handler.register_source(sid)

        new_stop_event = threading.Event()
        active_event = threading.Event()
        ready_event = threading.Event()

        new_thread = threading.Thread(
            target=self._serve,
            args=(new_handler, new_stop_event),
            kwargs=dict(active_event=active_event, ready_event=ready_event),
            daemon=True,
        )
        new_thread.start()

        if not ready_event.wait(timeout=connect_timeout):
            new_stop_event.set()
            new_thread.join(timeout=5.0)
            print(
                "EventWorker: make-before-break replacement connection did "
                f"not come up within {connect_timeout}s; falling back to "
                "break-before-make for this restart."
            )
            return False

        # Replacement is live (connected -- see _serve's ready_event, set
        # right after context_manager() is entered, not gated on having
        # actually received a first event yet -- see its docstring for why).
        # Swap references so any concurrent registerSource/removeSource call
        # targets the right (new) handler from here on.
        old_thread = self.loopThread
        old_stop_event = self._stop_event
        self._eventHandler = new_handler
        self._stop_event = new_stop_event
        self.loopThread = new_thread

        # Stop the old connection and wait for its thread to actually exit
        # BEFORE promoting the new one -- guarantees the two never dispatch
        # events at the same time (which could otherwise double-process an
        # overlapping pulse during the handover).
        old_stop_event.set()
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=5.0)
        active_event.set()
        return True

    # Consecutive failures of one callback before it's auto-removed from
    # eventCallbacks (equivalent to that Stream calling accumulate(False)).
    _CALLBACK_FAILURE_LIMIT = 20
    # Minimum seconds between repeated log lines for the *same* callback,
    # once it has already logged once.
    _CALLBACK_LOG_INTERVAL = 5.0

    @staticmethod
    def _callback_label(cb):
        """Best-effort human-readable label for a failing eventCallbacks entry."""
        owner = getattr(cb, "__self__", None)
        name = getattr(owner, "name", None)
        if name is not None:
            return repr(name)
        return repr(cb)

    def _handle_callback_error(self, cb, exc):
        """Log a callback exception, identified and rate-limited per-callback.

        Unbounded, unidentified ``print()`` spam here is what let a data-loss
        bug in a single Stream's callback hide as unreadable console noise
        instead of a diagnosable failure — see the escape.stream bug reports.
        """
        state = self._callback_failures.setdefault(cb, {"count": 0, "last_log": 0.0})
        state["count"] += 1
        now = time.time()
        label = self._callback_label(cb)
        if state["count"] == 1 or (now - state["last_log"]) >= self._CALLBACK_LOG_INTERVAL:
            print(
                f"EventWorker: callback for {label} failed "
                f"(x{state['count']} so far): {exc}"
            )
            state["last_log"] = now
        if state["count"] >= self._CALLBACK_FAILURE_LIMIT:
            print(
                f"EventWorker: callback for {label} failed "
                f"{state['count']} times in a row -- removing it from "
                f"eventCallbacks (equivalent to that Stream's accumulate(False))."
            )
            try:
                self.eventCallbacks.remove(cb)
            except ValueError:
                pass
            self._callback_failures.pop(cb, None)

    def eventLoop(self):  # noqa: N802
        self._serve(self._eventHandler, self._stop_event)

    def _serve(self, eventHandler, stop_event, active_event=None, ready_event=None):
        """Connect *eventHandler* and dispatch events until *stop_event* is
        set. This is ``eventLoop``'s actual implementation, generalized so
        ``_restart_make_before_break`` can run a second, independent one
        concurrently against a *different* handler/stop_event while the
        primary loop (``self._eventHandler``/``self._stop_event``) keeps
        running unaffected.

        *active_event*, when given, starts this loop in "warm-up" mode: it
        connects and, the moment the connection is live, sets *ready_event*
        (if given) -- but does not touch ``self.event``/``self.eventCallbacks``
        dispatch until *active_event* is set by the caller. Once set, it
        stays set (dispatch is permanent from then on, including across any
        later reconnect within this same call) -- this is what lets a
        replacement connection be verified live *before* the old one is
        torn down, without the two ever dispatching events at the same time
        (which would double-process an overlapping pulse). When
        *active_event* is ``None`` (the normal primary-loop case, via
        ``eventLoop``), this gate is skipped entirely and behavior is
        identical to before this method existed.

        "Live", for *ready_event*, means *eventHandler.context_manager()*
        was constructed and its context entered successfully -- not that a
        first real event has actually been received yet. Waiting for an
        actual event would be a stronger guarantee but risks hanging
        indefinitely on a channel that's merely quiet right now; this is a
        deliberate, documented tradeoff, not an oversight.
        """
        backoff = 1.0
        while not stop_event.is_set():
            try:
                ctx = eventHandler.context_manager()
                with ctx as s:
                    backoff = 1.0  # reset on successful connect
                    if ready_event is not None:
                        ready_event.set()
                    while not stop_event.is_set():
                        if active_event is not None and not active_event.is_set():
                            # Still warming up: stay connected (keeps the
                            # subscription alive and ready) but don't touch
                            # shared state or dispatch callbacks yet.
                            s.get_event()
                            time.sleep(0.001)
                            continue
                        self.event = s.get_event()
                        now = time.time()
                        dt = now - self._lastTime
                        if dt > 0:
                            self.runningFrequency = 1.0 / dt
                        self._lastTime = now
                        if self.event.getEventId() is not None:
                            keys = self.event.get_channel_names()
                            if keys:
                                self._last_event_keys = keys
                            for cb in list(self.eventCallbacks):
                                try:
                                    cb()
                                    self._callback_failures.pop(cb, None)
                                except Exception as exc:
                                    self._handle_callback_error(cb, exc)
                        time.sleep(0.001)
            except Exception as exc:
                if stop_event.is_set():
                    break
                print(f"EventWorker reconnecting in {backoff:.0f}s (error: {exc})")
                stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, 30.0)

    def startEventLoop(self):  # noqa: N802
        self._stop_event.clear()
        self.loopThread = threading.Thread(target=self.eventLoop, daemon=True)
        self.loopThread.start()

    def stopEventLoop(self):  # noqa: N802
        self._stop_event.set()
        if self.loopThread is not None and self.loopThread.is_alive():
            self.loopThread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # pulse_id / lab_time -- always-available pseudo-channels
    # ------------------------------------------------------------------
    #
    # Every bsread/datahub event already carries a pulse ID and a wall-clock
    # timestamp regardless of which real channels were requested -- see
    # EventSource.getEventData() -> Event_SFEL.getFromSource()/DataHubEvent.
    # getFromSource(), which special-case "pulse_id"/"lab_time". These
    # properties just save typing Stream('pulse_id', ew) by hand and, more
    # importantly, cache the result per EventWorker so repeated use (e.g. as
    # plot_corr()'s default x-axis) shares one Stream/accumulation rather
    # than creating a new one -- and duplicating it does not need a separate
    # accumulate(True): both pseudo-channels ride along with any other
    # channel already being received, no dispatcher subscription needed.

    @property
    def pulse_id(self):
        """Cached live Stream of each event's integer pulse ID."""
        if self._pulse_id_stream is None:
            self._pulse_id_stream = Stream("pulse_id", self)
        return self._pulse_id_stream

    @property
    def lab_time(self):
        """Cached live Stream of each event's wall-clock time (seconds since epoch)."""
        if self._lab_time_stream is None:
            self._lab_time_stream = Stream("lab_time", self)
        return self._lab_time_stream


# ---------------------------------------------------------------------------
# StreamBinning — returned by Stream.digitize(), applied via .categorize()
# ---------------------------------------------------------------------------

class StreamBinning:
    """Binning template created by ``Stream.digitize(bins)``.

    Mirrors ``Array.digitize()`` in the static escape API.  Call
    ``.categorize(other_stream)`` to create a new Stream whose data is
    accumulated in the defined bins.

    Example
    -------
    ratio_vs_t = t.digitize(np.linspace(-2, 2, 41)).categorize(i / i0)
    ratio_vs_t.accumulate(True)
    ratio_vs_t.plot_med()
    """

    def __init__(self, key_stream, bins):
        self.key_stream = key_stream
        self.bins = np.asarray(bins, dtype=float)

    def categorize(self, target_stream):
        """Return a new Stream: *target_stream*'s data accumulated per bin.

        Each incoming event for *target_stream* is assigned to a bin based on
        the current value of the key stream (the one ``.digitize()`` was called
        on).

        Parameters
        ----------
        target_stream : Stream

        Returns
        -------
        Stream
        """
        scan = digitizeScan(self.key_stream, self.bins)
        result = Stream(source=target_stream._source, scan=scan)
        # `result._source is target_stream._source` (identical object -- the
        # data is untouched, only the scan grouping differs), so a graph walk
        # based on `_source` alone would silently treat `result` as if it
        # independently recomputed target_stream's whole upstream chain,
        # rather than as "target_stream's data, regrouped by key_stream/bins".
        # `_graph_parent` records that relationship explicitly; see
        # escape.stream.graph.build_graph(), which treats its presence as
        # authoritative over the (otherwise misleading, in this one case)
        # `_source`-based node it would build by default.
        result._graph_parent = (
            "categorize",
            {"data": target_stream, "key": self.key_stream, "bins": self.bins},
        )
        return result

    def __repr__(self):
        return (
            f"StreamBinning(key={self.key_stream.name!r}, "
            f"n_bins={len(self.bins)-1}, "
            f"range=[{self.bins[0]:.3g}, {self.bins[-1]:.3g}])"
        )


# ---------------------------------------------------------------------------
# Running (windowed, live) statistics -- Stream.running_mean() et al.
# ---------------------------------------------------------------------------

def _broadcast_weight(w, vals):
    """Reshape a per-event scalar weight vector (N,) to broadcast against
    per-event *array*-valued samples of shape (N, *event_shape)."""
    if vals.ndim > 1:
        return w.reshape((-1,) + (1,) * (vals.ndim - 1))
    return w


def _weighted_mean(vals, w, skipnan=False):
    """vals: (N, *event_shape); w: (N,). Returns an array of shape event_shape
    (or a scalar when event_shape is ()) -- reduces along axis 0 only, so this
    works the same whether each sample is a scalar or an array."""
    w_eff = np.broadcast_to(_broadcast_weight(w, vals), vals.shape).astype(float)
    if skipnan:
        # A NaN weight is just as invalid as a NaN value -- if only vals were
        # masked, w_eff would still carry the NaN weight into the sum below
        # (nan * True == nan), silently poisoning the whole reduction.
        valid = ~np.isnan(vals) & ~np.isnan(w_eff)
        w_eff = np.where(valid, w_eff, 0.0)
        vals = np.where(valid, vals, 0.0)
    wsum = np.sum(w_eff, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sum(w_eff * vals, axis=0) / wsum
    return np.where(wsum > 0, out, np.nan)


def _weighted_std(vals, w, skipnan=False):
    w_eff = np.broadcast_to(_broadcast_weight(w, vals), vals.shape).astype(float)
    if skipnan:
        valid = ~np.isnan(vals) & ~np.isnan(w_eff)
        w_eff = np.where(valid, w_eff, 0.0)
        vals = np.where(valid, vals, 0.0)
    wsum = np.sum(w_eff, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.sum(w_eff * vals, axis=0) / wsum
        var = np.sum(w_eff * (vals - mean) ** 2, axis=0) / wsum
    return np.where(wsum > 0, np.sqrt(var), np.nan)


def _weighted_median(vals, w, skipnan=False):
    """Weighted median via cumulative-weight crossing of the 50% mark, computed
    independently per element position when vals is array-valued (axis 0 is
    the event/window axis; any trailing axes are the per-event array shape).

    Not interpolated (picks the crossing sample itself) -- a simple, standard
    enough definition for a live monitoring statistic; exact tie-breaking
    conventions vary across "weighted median" definitions in the literature.
    """
    w_eff = np.broadcast_to(_broadcast_weight(w, vals), vals.shape).astype(float)
    if skipnan:
        valid = ~np.isnan(vals) & ~np.isnan(w_eff)
        w_eff = np.where(valid, w_eff, 0.0)
        vals = np.where(valid, vals, np.inf)  # sort invalid entries to the end per position
    order = np.argsort(vals, axis=0)
    v_sorted = np.take_along_axis(vals, order, axis=0)
    w_sorted = np.take_along_axis(w_eff, order, axis=0)
    cw = np.cumsum(w_sorted, axis=0)
    total = cw[-1]
    with np.errstate(invalid="ignore"):
        idx = np.sum(cw < total / 2.0, axis=0)
    idx = np.clip(idx, 0, vals.shape[0] - 1)
    out = np.take_along_axis(v_sorted, np.expand_dims(np.asarray(idx), axis=0), axis=0)[0]
    return np.where(total > 0, out, np.nan)


class _RunningStat:
    """Stateful callable behind Stream.running_mean()/running_std()/etc.

    Holds its own bounded buffer of the last up to *N_acc* (value[, weight])
    pairs, recomputing the requested statistic on every call -- one call per
    real event, via wrapFunc_singleOutput/ProcObj's per-pulse-ID dedup. Unlike
    a plain deque(maxlen=...), the window is *not* fixed at construction: the
    owning Stream's writable ``.N_acc`` attribute is read fresh on every call,
    so it can be changed at any time (grown or shrunk) without recreating
    anything or losing already-buffered samples -- see Stream.running_mean().
    """

    _UNWEIGHTED_FUNCS = {
        "mean": (np.mean, np.nanmean),
        "std": (np.std, np.nanstd),
        "median": (np.median, np.nanmedian),
    }

    def __init__(self, stat, skipnan=False):
        self.stat = stat
        self.skipnan = skipnan
        self.owner = None  # wired up after the result Stream is created
        self._values = deque()
        self._weights = deque()

    def __call__(self, value, weight=None):
        # Same reasoning as DataManager.append()'s shape guard: a channel's
        # per-event shape can change mid-run (e.g. a detector ROI count
        # reconfigured), and _compute()'s np.asarray(self._values) raises an
        # inhomogeneous-shape ValueError once both shapes are in the window.
        # Drop the stale window and start fresh from this event instead.
        if self._values and np.shape(value) != np.shape(self._values[-1]):
            self._values.clear()
            self._weights.clear()
        self._values.append(value)
        if weight is not None:
            self._weights.append(weight)
        n_acc = max(int(self.owner.N_acc), 1) if self.owner is not None else len(self._values)
        while len(self._values) > n_acc:
            self._values.popleft()
        while len(self._weights) > n_acc:
            self._weights.popleft()
        return self._compute()

    def _compute(self):
        # vals: (N, *event_shape) -- N is the current window length, event_shape
        # is whatever shape each individual sample has (empty for a scalar
        # channel, e.g. (256,) for an array-valued one like an EvtSet or a
        # waveform). Every reduction below is explicitly axis=0 so only the
        # window/event axis collapses -- an array-valued channel's running
        # mean/std/median/mad is itself an array of that same shape, not a
        # single number.
        vals = np.asarray(self._values, dtype=float)
        w = np.asarray(self._weights, dtype=float) if self._weights else None

        if self.stat == "mad":
            if w is None:
                med_func = np.nanmedian if self.skipnan else np.median
                med = med_func(vals, axis=0)
                return med_func(np.abs(vals - med), axis=0)
            med = _weighted_median(vals, w, skipnan=self.skipnan)
            return _weighted_median(np.abs(vals - med), w, skipnan=self.skipnan)

        if w is None:
            plain, nan = self._UNWEIGHTED_FUNCS[self.stat]
            func = nan if self.skipnan else plain
            return func(vals, axis=0)

        if self.stat == "mean":
            return _weighted_mean(vals, w, skipnan=self.skipnan)
        if self.stat == "std":
            return _weighted_std(vals, w, skipnan=self.skipnan)
        return _weighted_median(vals, w, skipnan=self.skipnan)

    def _graph_label(self):
        """Human-readable op label for escape.stream.graph.build_graph().

        `N_acc` lives on the *result* Stream (``self.owner``), not as a
        ProcObj arg/kwarg -- deliberately, so it stays live-tunable without
        rebuilding anything (see running_mean's docstring). That means it is
        invisible to a walk over ProcObj.args/kwargs; this hook is read by
        the graph walker instead, at build time, so the label always shows
        the *current* window size rather than a stale snapshot.
        """
        label = ("nan" if self.skipnan else "") + self.stat
        n_acc = self.owner.N_acc if self.owner is not None else "?"
        return f"running_{label}(N_acc={n_acc})"


# ---------------------------------------------------------------------------
# Stream — main user-facing data object
# ---------------------------------------------------------------------------

class Stream:
    """Live accumulator for a single data channel.

    Mirrors ``escape.Array`` for interactive streaming analysis.

    Parameters
    ----------
    name_or_source : str or EventSource / ProcSource
        Channel name (string, preferred) or a pre-built source object
        (backward-compatible long form).
    eventworker : EventWorker, optional
        Required when *name_or_source* is a string.  Falls back to the module-
        level default registered by ``EventWorker(make_default=True)``.
    unit : str
        Physical unit used in plot axis labels.
    source : EventSource | ProcSource, keyword-only
        Used by internal code that already has a source object.
    dataManager : DataManager, optional, keyword-only
    scan : Scan, optional, keyword-only
        Scan definition; default is a no-scan (single step).

    Examples
    --------
    Short constructor (preferred)::

        ew = EventWorker(LocalEventHandler())
        i0   = Stream('i0',      ew, unit='counts')
        i    = Stream('i',       ew, unit='counts')
        t    = Stream('t',       ew, unit='ps')
        pump = Stream('pump_on', ew)

    Filtering (mirrors ``Array[bool_mask]``)::

        i_on  = i[pump]     # pump-on events only
        i_off = i[~pump]    # pump-off events  (~pump is logical NOT)

    Arithmetic → new derived Stream::

        ratio = i / i0      # per-shot ratio; fully live

    Binning by another channel (mirrors ``Array.digitize().categorize()``)::

        bins = np.linspace(-2, 2, 41)
        ratio_vs_t = t.digitize(bins).categorize(ratio)
        ratio_vs_t.accumulate(True)
        ratio_vs_t.plot_med()   # live median per delay bin

    Backward-compatible long form still works::

        i0 = Stream(source=EventSource('i0', ew, unit='counts'))
    """

    _isesc = True  # sentinel used by isesc() / ProcObj

    def __init__(
        self,
        name_or_source=None,
        eventworker=None,
        unit="a.u.",
        *,
        source=None,
        dataManager=None,
        scan=None,
    ):
        # Short form: Stream('channel', ew, unit='...')
        if isinstance(name_or_source, str):
            if eventworker is None:
                eventworker = globals().get("eventworker")
            source = EventSource(name_or_source, eventworker, unit=unit)
        # Legacy / internal: first positional arg is already a source object
        elif name_or_source is not None and source is None:
            source = name_or_source

        if scan is None:
            scan = Scan()
        self._source = source
        self.unit = source.unit
        self.name = source.name
        self.scan = scan
        if dataManager is None:
            dataManager = DataManager(scan=scan)
        self._dataManager = dataManager
        self.data = self._dataManager.data
        self.eventIds = self._dataManager.eventIds
        self._lastEventId = None

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dispatcher(cls, channel_name, eventworker=None, unit="a.u."):
        """Explicit-name spelling of the short constructor, ``Stream(channel_name, ew)``.

        A named channel is *already* dispatcher-sourced whenever *eventworker*
        uses a dispatcher-backed handler (``EventHandler_SFEL``,
        ``DataHubEventHandler``) -- this classmethod adds no new behavior,
        it exists for symmetry with :meth:`from_tcp` so example code can say
        which kind of source it means without the reader having to know
        which handler *eventworker* was built with.

        Parameters
        ----------
        channel_name : str
        eventworker : EventWorker, optional
            Falls back to the module-level default, same as the short
            constructor.
        unit : str
        """
        return cls(channel_name, eventworker, unit=unit)

    @classmethod
    def from_tcp(cls, address, field, unit="a.u.", mode="SUB"):
        """Read one field of a raw ``tcp://host:port`` bsread stream as a Stream.

        For a source that isn't a dispatcher channel *name* at all -- most
        notably a cam_server pipeline's output (``ratio.to_pipeline_server(...)``
        returns exactly such an address) -- rather than a named channel the
        dispatcher can resolve. Opens its **own** dedicated
        :class:`EventWorker`/background thread pointed at that one address
        (``make_default=False`` -- it must not replace whatever default
        EventWorker the rest of your session is using for dispatcher
        channels); calling this more than once for the same address opens a
        separate connection each time rather than sharing one.

        Parameters
        ----------
        address : str
            ``"tcp://host:port"`` (or ``"host:port"``), e.g. the
            *stream_address* returned by ``to_pipeline_server()`` or
            ``cam_server.PipelineClient.create_instance_from_config``.
        field : str
            Which key of the stream's (possibly multi-field) message to read
            -- e.g. the *name*/*output_name* a ``to_pipeline_server()`` call
            published under. A pipeline publishing several fields needs one
            ``from_tcp(address, field=...)`` call per field; they'll share
            nothing (see above) unless you pass the same *eventworker*
            explicitly to each -- pass ``eventworker=`` (positional slot
            after *unit* isn't exposed; construct once via
            ``DirectStreamEventHandler`` and reuse it, or accept one
            connection per field for simplicity).
        unit : str
        mode : str
            ``"SUB"`` (default) for a PUB-publishing sender -- correct for a
            cam_server pipeline output. ``"PULL"`` for a PUSH-publishing one.

        Examples
        --------
        ::

            instance_id, stream_address = ratio.to_pipeline_server("test_htlemke_ratio")
            result = Stream.from_tcp(stream_address, "test_htlemke_ratio")
            result.plot()   # or .accumulate(True), like any other Stream

        See Also
        --------
        DirectStreamEventHandler : the handler this builds on
            (``escape.stream.es_wrappers``).
        """
        from .es_wrappers import DirectStreamEventHandler

        addr = address.split("//")[-1]  # strip an optional "tcp://" prefix
        host, port = addr.rsplit(":", 1)
        ew = EventWorker(DirectStreamEventHandler(host, port, mode=mode), make_default=False)
        return cls(field, ew, unit=unit)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def shape(self):
        return self._dataManager._getDataShape()

    def lens(self):
        """Samples currently retained per scan step (see DataManager.lens)."""
        return self._dataManager.lens()

    def counts(self):
        """True, uncapped event count per scan step (see DataManager.counts)."""
        return self._dataManager.counts()

    def __len__(self):
        return len(self._dataManager)

    def _getEventDataRaw(self):
        return self._source.getEventData()

    def _getEventData(self):
        if self.scan._isValid():
            return self._getEventDataRaw()
        return None

    def _appendEventData(self):
        eventId = self._source.eventWorker.event.getEventId()
        if eventId is None or eventId == self._lastEventId:
            return
        data = self._getEventData()
        if data is not None:
            self._dataManager.append(data, eventId)
        self._lastEventId = eventId

    def _update(self):
        self._appendEventData()

    # ------------------------------------------------------------------
    # Accumulation control
    # ------------------------------------------------------------------

    def accumulate(self, do_accumulate=None):
        """Start or stop accumulation.  Toggle with no argument.

        For a derived Stream (arithmetic, :meth:`filter`/``[mask]``,
        :meth:`element`/``[i]``, :meth:`categorize`, ...) this transparently
        subscribes every *real* channel the computation ultimately depends
        on -- there is no need to separately ``accumulate(True)`` a mask
        stream or an upstream operand just so a derived Stream built from it
        receives data.
        """
        if do_accumulate is None:
            do_accumulate = not self._is_accumulating()
            print(f"Toggling accumulation {'ON' if do_accumulate else 'OFF'} for {self.name!r}")

        ew = self._source.eventWorker
        if do_accumulate:
            channels = _collect_source_channels(self._source)
            if channels:
                ew.registerSources(*channels)
            if self._appendEventData not in ew.eventCallbacks:
                ew.eventCallbacks.append(self._appendEventData)
        else:
            try:
                ew.eventCallbacks.remove(self._appendEventData)
            except ValueError:
                pass
            # Only unregister for a plain, directly-named channel -- a derived
            # source's dependencies may still be in use by sibling Streams, and
            # _collect_source_channels doesn't reference-count consumers.
            if isinstance(self._source, EventSource):
                ew.removeSource(self._source.name)

    def _is_accumulating(self):
        return self._appendEventData in self._source.eventWorker.eventCallbacks

    def acquire(self):
        """Return a StreamContext that manages this object's accumulation lifetime."""
        return StreamContext(self)

    def __enter__(self):
        """Start accumulation when used as a ``with`` statement."""
        self.accumulate(True)
        return self

    def __exit__(self, *_):
        """Stop accumulation when the ``with`` block exits."""
        self.accumulate(False)
        return False

    def to_frame(self):
        """Return accumulated data as a :class:`pandas.DataFrame`.

        The returned DataFrame has columns ``event`` (integer, 0-based within each
        step), ``scan_step`` (0-based step index), ``scan_value`` (the parameter
        value for that step, or NaN when no scan is defined), and a column named
        after the stream holding the measured value.

        Requires ``pandas``.
        """
        import pandas as pd

        rows = []
        if self.scan._parameters is None:
            for ev_idx, v in enumerate(self.data[0]):
                rows.append({"event": ev_idx, "scan_step": 0, "scan_value": float("nan"), self.name: v})
        else:
            for step_idx, step_data in enumerate(self.data):
                try:
                    step_val = float(self.scan[step_idx][0]) if len(self.scan[step_idx]) else float("nan")
                except Exception:
                    step_val = float("nan")
                for ev_idx, v in enumerate(step_data):
                    rows.append({"event": ev_idx, "scan_step": step_idx, "scan_value": step_val, self.name: v})
        return pd.DataFrame(rows)

    def to_array(self, index="pulse_id", name=None, unit=None):
        """Snapshot this Stream's currently accumulated data as a static ``escape.Array``.

        A one-shot conversion of whatever has been accumulated *so far* --
        not a live link back to this Stream -- bridging into the
        escape.storage ecosystem (index-aligned arithmetic, HDF5/zarr
        persistence, ``DataSet``, ...). Uses the same per-step "step/hist"
        data access ``HistPlot``/``ValueHistPlot``/:meth:`to_frame` already
        go through (``self.data``, ``self.lens()``, ``self.eventIds``) --
        including their caveat: a step beyond ``maxlen`` samples only has its
        *retained* (not full historical) samples included, same as
        :meth:`lens`.

        Parameters
        ----------
        index : "pulse_id" or array-like
            Per-event index for the resulting Array. ``"pulse_id"``
            (default) uses this Stream's own already-recorded event IDs
            (``self.eventIds``) -- ``DataManager.append()`` stores the event
            ID (bsread's pulse ID; see ``Event_SFEL.getEventId()``)
            alongside every sample already, for *every* accumulating
            Stream. There is no separate ``pulse_id`` Stream to build or
            accumulate for this, and nothing gets freshly (re-)subscribed
            from the dispatcher to produce it -- it's already there,
            unconditionally, the moment this Stream accumulates anything.
            Pass an explicit array only to override it (length must match
            this Stream's total accumulated sample count).
        name, unit : str, optional
            Default to this Stream's own ``.name``/``.unit``.

        Returns
        -------
        escape.Array
            With ``step_lengths``/``parameter`` set from this Stream's scan
            structure if it has one (mirroring ``escape.storage.Scan``'s
            ``{param_name: {"values": [...]}}`` shape) -- a single implicit
            step otherwise.
        """
        from escape import Array

        data = np.asarray([v for step in self.data for v in step])
        if isinstance(index, str) and index == "pulse_id":
            idx = np.asarray([v for step in self.eventIds for v in step])
        else:
            idx = np.asarray(index)
            if len(idx) != len(data):
                raise ValueError(
                    f"to_array: explicit index has {len(idx)} entries, but "
                    f"{len(data)} samples are currently accumulated."
                )

        name = name or self.name
        unit = unit or self.unit
        if self.scan._parameters is None:
            return Array(data=data, index=idx, name=name, unit=unit)

        step_lengths = self.lens()
        parameter = {p: {"values": list(self.scan[p])} for p in self.scan.keys()}
        return Array(
            data=data, index=idx, step_lengths=step_lengths, parameter=parameter,
            name=name, unit=unit,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def mean(self):
        return [np.mean(td, axis=0) for td in self.data]

    def std(self):
        return [np.std(td, axis=0) for td in self.data]

    def median(self):
        return [np.median(td, axis=0) for td in self.data]

    def centerPerc(self, perc=68.3):
        pervals = [50 - perc / 2.0, 50 + perc / 2.0]
        # np.percentile raises IndexError on an empty step (unlike mean/median/std,
        # which just emit a RuntimeWarning and return nan) -- match their behavior
        # so one still-empty scan step doesn't blow up a whole-array caller such as
        # Plot._getplotData().
        return [
            np.percentile(td, pervals, axis=0) if len(td) else np.full(2, np.nan)
            for td in self.data
        ]

    # ------------------------------------------------------------------
    # Live running statistics
    # ------------------------------------------------------------------

    def _running_stat(self, stat, N_acc, weights, skipnan):
        engine = _RunningStat(stat, skipnan=skipnan)
        label = ("nan" if skipnan else "") + stat
        args = [self] if weights is None else [self, weights]
        name = f"{self.name}_running_{label}"
        if weights is not None:
            name += f"_weighted_by_{weights.name}"
        p = ProcObj(
            engine,
            args=args,
            returns_is_esc=[True],
            returns_names=[name],
            returns_units=[self.unit],
            # A running stat spans the whole accumulation window, not a scan
            # step -- like wrapFunc_singleOutput(), it deliberately does not
            # inherit self.scan.
            scan=Scan(),
        )
        result = p.createChildren()[0]
        result.N_acc = N_acc  # plain, freely reassignable attribute -- see _RunningStat
        engine.owner = result
        return result

    def running_mean(self, N_acc=100, weights=None):
        """Live windowed mean, recomputed every event over the last ``N_acc`` samples.

        Returns a new Stream whose data is the current running mean -- one
        value per pulse, not a one-off snapshot. The window size is read from
        the *returned* Stream's ``.N_acc`` attribute on every event, so it can
        be changed at any time, mid-run, without recreating anything::

            rm = i0.running_mean(N_acc=200)
            rm.accumulate(True)
            ...
            rm.N_acc = 50   # takes effect from the next event onward

        Parameters
        ----------
        N_acc : int
            Window size: number of most recent samples averaged over.
        weights : Stream, optional
            Per-event weight stream (same window). If given, computes a
            weighted mean ``sum(w*x)/sum(w)`` over the window instead of a
            plain mean. Accumulating the result auto-subscribes both this
            Stream's and *weights*' real channel(s) (see :meth:`accumulate`).

        See Also
        --------
        running_nanmean, running_std, running_median, running_mad
        """
        return self._running_stat("mean", N_acc, weights, skipnan=False)

    def running_nanmean(self, N_acc=100, weights=None):
        """Like :meth:`running_mean`, but ignores nan-valued samples in the window."""
        return self._running_stat("mean", N_acc, weights, skipnan=True)

    def running_std(self, N_acc=100, weights=None):
        """Live windowed standard deviation (population, ddof=0), analogous to :meth:`running_mean`."""
        return self._running_stat("std", N_acc, weights, skipnan=False)

    def running_nanstd(self, N_acc=100, weights=None):
        """Like :meth:`running_std`, but ignores nan-valued samples in the window."""
        return self._running_stat("std", N_acc, weights, skipnan=True)

    def running_median(self, N_acc=100, weights=None):
        """Live windowed median, analogous to :meth:`running_mean`.

        With *weights*, uses a cumulative-weight-crossing weighted median
        (see the module-level ``_weighted_median`` helper) rather than the
        plain unweighted ``numpy.median``.
        """
        return self._running_stat("median", N_acc, weights, skipnan=False)

    def running_nanmedian(self, N_acc=100, weights=None):
        """Like :meth:`running_median`, but ignores nan-valued samples in the window."""
        return self._running_stat("median", N_acc, weights, skipnan=True)

    def running_mad(self, N_acc=100, weights=None):
        """Live windowed median absolute deviation, analogous to :meth:`running_mean`.

        ``median(abs(x - median(x)))`` over the window (unscaled -- multiply
        by ~1.4826 for a normal-consistent std estimate). With *weights*,
        both medians involved are weighted.
        """
        return self._running_stat("mad", N_acc, weights, skipnan=False)

    def running_nanmad(self, N_acc=100, weights=None):
        """Like :meth:`running_mad`, but ignores nan-valued samples in the window."""
        return self._running_stat("mad", N_acc, weights, skipnan=True)

    # ------------------------------------------------------------------
    # Array-mirroring API: filtering and categorisation
    # ------------------------------------------------------------------

    def filter(self, mask_stream):
        """Return a new Stream that only accumulates events where *mask_stream* is truthy.

        Mirrors ``escape.Array`` boolean indexing.  For live bsread streams the
        mask is evaluated per-event from the current bsread message without
        requiring the mask stream to be separately accumulating.

        Parameters
        ----------
        mask_stream : Stream
            Boolean-valued stream.  Use ``~pump`` for logical NOT (gives the
            complement of a 0/1 channel as True/False).

        Returns
        -------
        Stream

        Note
        ----
        Calling ``.accumulate(True)`` on the *returned* filtered Stream (or on
        anything built from it) automatically subscribes both *mask_stream*'s
        and the filtered channel's real underlying channels -- there's no need
        to separately call ``mask_stream.accumulate(True)``.

        See Also
        --------
        Stream.__getitem__ : ``i[pump]`` is shorthand for ``i.filter(pump)``.
        """
        filtered_src = FilteredEventSource(self._source, mask_stream, inner_stream=self)
        return Stream(source=filtered_src)

    def element(self, index):
        """Select a single element from this Stream's per-event array value.

        For an array-valued channel (e.g. a boolean event-code set such as
        ``SAR-CVME-TIFALL5:EvtSet``), returns a new scalar-valued Stream whose
        data is ``raw_value[index]`` for each event -- live, not a one-off
        snapshot. The result is a normal Stream, so it can itself be used as
        a filter mask on another Stream: ``other[evtset.element(25)]`` (or
        the ``evtset[25]`` shorthand via :meth:`__getitem__`) accumulates
        *other*'s events only where bit 25 of the event set is set.

        Calling ``.accumulate(True)`` on the result (or on anything built from
        it, e.g. ``other[evtset[25]]``) automatically subscribes the real
        underlying channel (``evtset``'s) -- there's no need to separately
        call ``evtset.accumulate(True)``.

        Parameters
        ----------
        index : int
            Index into the per-event array. Ordinary numpy indexing rules
            apply (negative indices count from the end).

        Returns
        -------
        Stream
        """
        # `index` is passed as an explicit (non-esc) ProcObj arg rather than
        # closed over in the lambda, so it shows up as a real constant node
        # to anything walking the ProcObj graph (see escape.stream.graph) --
        # a value baked into a closure is otherwise invisible to that walk.
        index_func = lambda arr, idx: arr[idx]
        # Opt into escape.stream.pipeline_codegen: given the already-computed
        # source expressions for [arr, idx], return the equivalent Python
        # expression for this op.
        index_func._graph_codegen = lambda arg_exprs: f"{arg_exprs[0]}[{arg_exprs[1]}]"
        p = ProcObj(
            index_func,
            args=[self, index],
            returns_is_esc=[True],
            returns_names=[f"{self.name}[{index}]"],
            returns_units=[self.unit],
            scan=self.scan,
        )
        return p.createChildren()[0]

    def __getitem__(self, key):
        """Index a Stream.

        Parameters
        ----------
        key : Stream
            Boolean mask — returns ``self.filter(key)``.  Use ``~mask`` for
            logical NOT.
        key : slice
            Returns accumulated data as a numpy array.  ``s[:100]`` gives the
            first 100, ``s[-200:]`` the last 200, across all scan steps.
        key : int
            Returns ``self.element(key)`` -- a new live Stream of that single
            element from this Stream's per-event array value. Only meaningful
            when this Stream's raw value is itself an array (e.g. an
            event-code-set channel); the slice form above is for accumulated
            *sample* access on any Stream and is a different operation.
        """
        if isinstance(key, Stream):
            return self.filter(key)
        if isinstance(key, slice):
            all_data = [v for step in self.data for v in step]
            return np.array(all_data[key])
        if isinstance(key, (int, np.integer)):
            return self.element(key)
        raise TypeError(
            f"Stream index must be a Stream (for filtering), a slice (for "
            f"data access), or an int (to select one element of a per-event "
            f"array value), not {type(key).__name__}"
        )

    def digitize(self, bins):
        """Create a binning template for this channel's values.

        Mirrors ``escape.Array.digitize()``.  Returns a :class:`StreamBinning`
        whose ``.categorize(other)`` produces a new Stream binned by *self*'s
        values.

        Parameters
        ----------
        bins : array-like of float
            Monotonically increasing bin edges.

        Returns
        -------
        StreamBinning

        Example
        -------
        ratio_vs_t = t.digitize(np.linspace(-2, 2, 41)).categorize(i / i0)
        """
        return StreamBinning(self, np.asarray(bins, dtype=float))

    def categorize(self, other_stream):
        """Apply this Stream's scan structure to *other_stream*.

        Mirrors ``escape.Array.categorize()``.  Returns a new Stream that
        accumulates *other_stream*'s data using *self*'s scan step assignments.

        Multiple Streams may share the same ``Scan`` this way — including one
        built against a hand-written, non-bsread parameter source for
        externally driven categorization (see the :class:`Scan` docstring) —
        each safely keeps its own accumulated data in step with the others.

        Parameters
        ----------
        other_stream : Stream

        Returns
        -------
        Stream
        """
        result = Stream(source=other_stream._source, scan=self.scan)
        # See the identical note in StreamBinning.categorize() -- `_source` is
        # shared with other_stream, so `_graph_parent` is what actually
        # records the "regrouped by self's scan" relationship for graph.py.
        result._graph_parent = (
            "categorize_scan",
            {"data": other_stream, "scan_from": self},
        )
        return result

    def categorizeBy(self, key_stream, binning_def, side="left"):
        """Bin *self* by the value of *key_stream* (legacy method).

        Prefer ``key_stream.digitize(bins).categorize(self)`` for new code.
        """
        if isinstance(binning_def, float):
            s = Scan(
                parameters=[key_stream],
                precision={key_stream.name: binning_def},
            )
        elif np.iterable(binning_def):
            s = digitizeScan(key_stream, binning_def)
        else:
            raise ValueError("binning_def must be a float or an iterable of edges")
        result = Stream(source=self._source, scan=s)
        # See the identical note in StreamBinning.categorize().
        result._graph_parent = (
            "categorizeBy",
            {"data": self, "key": key_stream, "binning": binning_def},
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline offload
    # ------------------------------------------------------------------

    def to_pipeline_server(
        self, name, pipeline_client=None, output_name=None,
        additional_config=None, redeploy=True,
    ):
        """Run this Stream's computation on the PSI pipeline server (``cam_server``)
        instead of in this client, and publish the result as a new bsread stream.

        Walks this Stream's computation graph (:func:`escape.stream.graph.build_graph`),
        auto-generates the equivalent ``process(data, pulse_id, timestamp,
        parameters)`` script
        (:func:`escape.stream.pipeline_codegen.generate_process_script`), and
        deploys it as a ``pipeline_type: "stream"`` instance on cam_server,
        subscribed to exactly the real channels this Stream depends on. This
        is the automated version of the manual workflow demonstrated in
        ``example_pipeline_offload.ipynb`` -- validated against the real
        Bernina cam_server; read that notebook first if any of this is
        surprising.

        Requires the optional ``cam_server`` package (``pip install
        cam_server``) and ``networkx`` (see :mod:`escape.stream.graph`) --
        both imported lazily, so merely importing ``escape.stream`` needs
        neither.

        Parameters
        ----------
        name : str
            The pipeline instance id *and* (unless *output_name* is given)
            the key under which the result is published in the output
            stream. Keep this short and unambiguous -- e.g. prefixed
            ``test_<you>_...`` for anything experimental -- since it is a
            real, visible name on a server shared with other beamline work.
        pipeline_client : cam_server.PipelineClient, optional
            An existing client to reuse (e.g. already pointed at a
            non-default server address). Defaults to a fresh
            ``PipelineClient()`` (``http://sf-daqsync-01:8889/`` unless
            overridden).
        output_name : str, optional
            Output field name, if different from *name*.
        additional_config : dict, optional
            Extra keys merged into the instance config (e.g.
            ``dispatcher_url``) -- see ``cam_server/pipeline/utils.py``'s
            ``connect_to_stream()`` for what a ``"stream"``-type instance
            understands.
        redeploy : bool
            If an instance named *name* is already running, stop it first
            and create a fresh one (default). This is the only redeploy
            strategy implemented right now -- cam_server also supports a
            cheaper in-place hot-reload (``set_function_script`` /
            ``reload=True``) when only the script body changed and the
            channel wiring didn't, but that path isn't wired up here yet.

        Returns
        -------
        (instance_id, stream_address)
            *stream_address* is a raw ``tcp://host:port`` bsread address --
            **not** a dispatcher-registered channel name. Read it directly:
            ``bsread.Source(host=..., port=...)`` (see
            ``cam_server_client.utils.get_host_port_from_stream_address`` to
            parse it, and Part 4 of the example notebook to read it). There
            is currently no automatic way to wrap that address back into a
            client-side ``Stream`` (the existing ``EventSource`` only knows
            how to subscribe to dispatcher channel *names*, not raw
            addresses) -- a natural next piece, not yet built.

        What can and can't be translated
        ---------------------------------
        Covers exactly what was validated live: real channels, constants,
        arithmetic operators, ``element()``/indexing, opposite-mask
        ``filter()``/``[mask]`` pairs (all stateless, recomputed fresh every
        event), and the plain (unweighted, non-nan-skipping)
        ``running_mean``/``running_std``/``running_median``/``running_mad``
        family (the one *stateful* op, translated to an explicit
        module-level ``deque`` since cam_server's ``"stream"`` pipeline type
        gives ``process()`` no ``init``/state argument of its own -- unlike
        its ``"custom"`` type, see the mapping table in the example
        notebook). Anything else -- ``categorize()``/``digitize()`` (a
        scan-step concept with no meaning in a stateless per-event server
        script), weighted or nan-skipping running stats, or a function this
        module doesn't recognize -- raises ``NotImplementedError`` naming
        exactly what's missing, rather than silently deploying code that
        doesn't match what this Stream actually computes.

        Making a custom function pipeline-compatible
        ----------------------------------------------
        If you write your own derived-Stream constructor (like ``element()``)
        and want ``to_pipeline_server()`` to handle it, give the function you
        pass to ``ProcObj`` a ``_graph_codegen`` attribute -- a callable that
        takes the already-generated source expression for each *positional*
        ``ProcObj`` arg (in order, constants included, rendered as their
        ``repr()``) and returns the Python expression computing this node's
        value from them, as plain text. This only works for **stateless**
        ops -- computed fresh every event from this event's own inputs, nothing
        remembered across calls. ``element()``'s own implementation is the
        template::

            index_func = lambda arr, idx: arr[idx]
            index_func._graph_codegen = lambda arg_exprs: f"{arg_exprs[0]}[{arg_exprs[1]}]"
            p = ProcObj(index_func, args=[self, index], ...)

        ``generate_process_script`` wraps the returned expression in a
        ``None``-propagation guard automatically (if any *Stream-valued* arg
        is ``None`` this event, the whole node is ``None`` too) -- your
        ``_graph_codegen`` callable only needs to return the "happy path"
        expression, not handle ``None`` itself.

        If your op needs to remember something **across** events (a window,
        a counter, an EMA, ...) that's a fundamentally different, harder
        case -- ``_graph_codegen`` alone can't express it, since it only
        emits one expression, not a persistent module-level variable plus an
        update rule. There's no generic protocol for that yet; the only
        stateful op supported today (``_RunningStat``, behind
        ``running_mean``/etc.) is special-cased directly in
        ``pipeline_codegen._emit_running_stat`` rather than going through a
        generic hook -- follow that function as the template if you need a
        second stateful op, or write the ``process()`` script for that
        Stream by hand instead of calling ``to_pipeline_server()`` on it.

        Examples
        --------
        ::

            isref = evtset.element(5)
            ratio = i[~isref] / i[isref].running_mean(N_acc=5)
            instance_id, stream_address = ratio.to_pipeline_server("test_htlemke_ratio")
            ...
            pipeline_client.stop_instance(instance_id)   # always clean up
        """
        from .pipeline_codegen import generate_process_script

        if pipeline_client is None:
            from cam_server import PipelineClient
            pipeline_client = PipelineClient()

        script = generate_process_script(self, output_name=output_name or name)
        channels = _collect_source_channels(self._source)

        config = {
            "pipeline_type": "stream",
            "function": f"{name}.py",
            "bsread_channels": channels,
        }
        if additional_config:
            config.update(additional_config)

        if redeploy and pipeline_client.is_instance_running(name):
            pipeline_client.stop_instance(name)

        pipeline_client.set_user_script(config["function"], script)
        return pipeline_client.create_instance_from_config(config, instance_id=name)

    # ------------------------------------------------------------------
    # Live plots
    # ------------------------------------------------------------------

    def _wait_for_data(self, timeout=5):
        deadline = time.time() + timeout
        while len(self) < 2:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for data from {self.name!r}.")
            time.sleep(0.05)

    def _peek_shape(self):
        """Shape of the most recently accumulated raw sample, () if scalar
        or if no data has arrived yet."""
        for step in self.data:
            if len(step):
                return np.shape(step[-1])
        return ()

    def plot_hist(self, update=0.5, axes=None, timeout=5, n_bins=50, N_acc=100):
        """Value-distribution or scan-count histogram with live updates.

        Routes to :class:`ValueHistPlot` (no scan) or :class:`HistPlot` (scan)
        for a scalar Stream. For an **array-valued** Stream (e.g. a per-event
        waveform or an event-code-set channel), routes instead to
        :class:`~escape.stream.plots.WaterfallPlot` -- a 2D live image, rows =
        the last *N_acc* events, columns = array index -- since neither a
        value histogram nor a scan-step-count histogram is meaningful
        per array element.
        """
        self.accumulate(True)
        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.figure
        self._wait_for_data(timeout)
        if np.prod(self._peek_shape(), dtype=int) > 1:
            fig.suptitle(f"{self.name}  (live, last {N_acc} events)")
            hp = plots.WaterfallPlot(self, N_acc=N_acc, axes=axes, update_interval=update or 0.5)
        else:
            has_scan = self.scan._parameters is not None
            fig.suptitle(f"{self.name}  histogram" if has_scan else f"{self.name}  value distribution")
            hp = plots.HistPlot(self, axes=axes) if has_scan else plots.ValueHistPlot(self, axes=axes, n_bins=n_bins)
        hp.plot()
        if update:
            hp.start(interval=update)
        self._histPlot = hp
        return hp

    def plot_med(self, update=0.5, axes=None, timeout=5, peak_overlay=True):
        """Median + percentile bands with live updates.

        peak_overlay : bool
            Show a live peak-analysis overlay (center + FWHM, see
            ``escape.stream.plots.find_peak``) on top of the median line.
            Defaults to ``True``; pass ``False`` to opt out (e.g. for a
            channel that isn't peak-shaped).
        """
        self.accumulate(True)
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{self.name}  median")
        self._wait_for_data(timeout)
        mp = plots.Plot(self, axes=axes, peak_overlay=peak_overlay)
        mp.plot()
        if update:
            mp.start(interval=update)
        self._medPlot = mp
        return mp

    def plot(self, rate_Hz=1.0, n_history=200, axes=None, timeout=5):
        """Live view of the current value, redrawn at *rate_Hz*.

        The generic fallback live plot: ignores scan structure entirely,
        always shows the most recently accumulated sample(s) -- exactly what
        you want for a derived, no-scan Stream such as a live-normalized
        detector trace (``(a[~mask] / a[mask].running_mean(N_acc=50)).plot()``).

        For an **array-valued** Stream shows the single latest snapshot as a
        line (x = array index). For a **scalar** Stream shows a rolling trend
        of the last *n_history* samples (x = recent event index). See
        :class:`~escape.stream.plots.TracePlot`.

        Parameters
        ----------
        rate_Hz : float
            Redraw rate in Hz (``rate_Hz=1`` -> once per second). ``0``/
            ``None`` draws once and does not start live updates.
        n_history : int
            Scalar Streams only: how many recent samples to show as a trend.
        """
        self.accumulate(True)
        if axes is None:
            fig, axes = plt.subplots()
        self._wait_for_data(timeout)
        interval = 1.0 / rate_Hz if rate_Hz else 1.0
        tp = plots.TracePlot(self, axes=axes, n_history=n_history, update_interval=interval)
        tp.plot()
        if rate_Hz:
            tp.start(interval=interval)
        self._tracePlot = tp
        return tp

    def plot_corr(self, xVar=None, Npoints=300, update=0.5, axes=None, timeout=5,
                  default_x="lab_time", N_acc=100):
        """Correlation / trend plot against *xVar*, live-updating.

        With no *xVar*, defaults to a live ``lab_time`` Stream (wall-clock
        seconds) bound to this Stream's own EventWorker -- i.e. a live "value
        vs time" trend plot with no arguments needed. Pass
        ``default_x='pulse_id'`` to default to pulse ID instead, or an
        explicit Stream for *xVar* to correlate against any other channel.

        The representation is chosen automatically once data is available:

        - scalar vs scalar -> the classic matched-by-pulse-ID scatter
          (:class:`~escape.stream.plots.PlotCorrelation`).
        - array vs scalar (either side) -> a 2D history image
          (:class:`~escape.stream.plots.WaterfallPlot`), rows ordered by the
          scalar Stream's live value (e.g. array-valued channel vs time).
        - array vs array, same shape -> every element of the last *Npoints*
          matched events, pooled into one dense scatter -- a true
          element-by-element correlation isn't otherwise representable as a
          single 2D plot.
        - array vs array, different shape -> raises ``ValueError``; not
          representable.
        """
        ew = self._source.eventWorker
        if xVar is None:
            if default_x == "pulse_id":
                xVar = ew.pulse_id
            elif default_x == "lab_time":
                xVar = ew.lab_time
            else:
                raise ValueError("default_x must be 'lab_time' or 'pulse_id'")

        self.accumulate(True)
        xVar.accumulate(True)
        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.figure
        self._wait_for_data(timeout)
        xVar._wait_for_data(timeout)

        y_arr = np.prod(self._peek_shape(), dtype=int) > 1
        x_arr = np.prod(xVar._peek_shape(), dtype=int) > 1

        if y_arr and not x_arr:
            fig.suptitle(f"{self.name}  vs  {xVar.name}  (live)")
            cp = plots.WaterfallPlot(self, x_stream=xVar, N_acc=N_acc, axes=axes, update_interval=update or 0.5)
        elif x_arr and not y_arr:
            fig.suptitle(f"{xVar.name}  vs  {self.name}  (live)")
            cp = plots.WaterfallPlot(xVar, x_stream=self, N_acc=N_acc, axes=axes, update_interval=update or 0.5)
        elif y_arr and x_arr:
            y_shape, x_shape = self._peek_shape(), xVar._peek_shape()
            if y_shape != x_shape:
                raise ValueError(
                    f"plot_corr: {self.name!r} (shape {y_shape}) and {xVar.name!r} "
                    f"(shape {x_shape}) are both array-valued with different "
                    f"shapes -- element-by-element correlation isn't defined."
                )
            fig.suptitle(f"{self.name}  vs  {xVar.name}  (live, pooled elements)")
            cp = plots.PlotCorrelation(xVar, self, Nlast=Npoints, axes=axes, flatten_arrays=True)
        else:
            fig.suptitle(f"{self.name}  vs  {xVar.name}")
            cp = plots.PlotCorrelation(xVar, self, Nlast=Npoints, axes=axes)
        cp.plot()
        if update:
            cp.start(interval=update)
        self._corrPlot = cp
        return cp

    def __repr__(self):
        n = len(self)
        steps = len(self.scan)
        scan_info = (
            f", {steps} scan step{'s' if steps != 1 else ''}"
            if self.scan._parameters
            else ""
        )
        return f"Stream({self.name!r}, unit={self.unit!r}, n={n}{scan_info})"


# ---------------------------------------------------------------------------
# ProcObj — lazy function evaluation over Stream operands
# ---------------------------------------------------------------------------

class ProcObj:
    def __init__(
        self,
        func,
        args=(),
        kwargs=None,
        returns_is_esc=None,
        returns_names=None,
        returns_units=None,
        scan=None,
        scanIndex=None,
    ):
        if kwargs is None:
            kwargs = {}
        if returns_is_esc is None:
            returns_is_esc = [True]
        self.func = func
        self.args = list(args)
        self.kwargs = kwargs
        self.returns_is_esc = returns_is_esc
        self.returns_names = returns_names
        self.returns_units = returns_units
        self.args_is_esc = [isesc(a) for a in args]
        self.kwargs_is_esc = {k: isesc(v) for k, v in kwargs.items()}
        self.ret_values = None
        self._last_processed_eventId = None
        self.children = None

        if scan is None:
            if scanIndex is not None:
                self.scan = self._escArgs()[scanIndex].scan.copy()
            else:
                self.scan = Scan()
        else:
            self.scan = scan

        self.eventWorker = self._resolveEventWorker()

    def _escArgs(self):
        return [a for a, ie in zip(self.args, self.args_is_esc) if ie]

    def _resolveEventWorker(self):
        esc_objs = self._escArgs() + [v for k, v in self.kwargs.items() if self.kwargs_is_esc[k]]
        if not esc_objs:
            return None
        workers = [obj._source.eventWorker for obj in esc_objs]
        assert all(w is workers[0] for w in workers), \
            "All Stream operands must share the same EventWorker."
        return workers[0]

    def createChildren(self):  # noqa: N802
        self.children = []
        for idx, isesc_ in enumerate(self.returns_is_esc):
            if isesc_:
                name = (self.returns_names or [])[idx] if self.returns_names else "none"
                unit = (self.returns_units or [])[idx] if self.returns_units else "none"
                self.children.append(
                    Stream(
                        source=ProcSource(
                            self,
                            name=name,
                            unit=unit,
                            eventWorker=self.eventWorker,
                            returnIndex=idx,
                        ),
                        scan=self.scan.copy(),
                    )
                )
        return self.children

    def getEventData(self):  # noqa: N802
        eid = self.eventWorker.event.getEventId()
        if eid == self._last_processed_eventId:
            return False
        args = [
            a._getEventData() if ie else a
            for a, ie in zip(self.args, self.args_is_esc)
        ]
        kwargs = {
            k: v._getEventData() if self.kwargs_is_esc[k] else v
            for k, v in self.kwargs.items()
        }
        self._last_processed_eventId = eid
        if any(a is None for a in args) or any(v is None for v in kwargs.values()):
            return False
        result = self.func(*args, **kwargs)
        self.ret_values = result if isinstance(result, tuple) else (result,)
        return True

    def updateChildren(self, caller):  # noqa: N802
        for child in self.children:
            if child._source is not caller:
                child._update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def isesc(obj):
    return bool(getattr(obj, "_isesc", False))


def initStreamInstances(eventWorker=None):  # noqa: N802
    """Create a Stream for every channel currently available from the handler.

    Returns a namespace so you can access channels as attributes, e.g. ``ns.i0``.
    """
    if eventWorker is None:
        eventWorker = EventWorker()
    sources = eventWorker._eventHandler.get_all_source_ids()
    out = {sid: Stream(source=EventSource(sid, eventWorker)) for sid in sources}
    eventWorker.startEventLoop()
    return tools.Dict2obj(out)


# backward-compatible alias
def initEscDataInstances(eventWorker=None):  # noqa: N802
    return initStreamInstances(eventWorker)


def _resolve_eventworker(eventworker, func_name):
    if eventworker is not None:
        return eventworker
    eventworker = globals().get("eventworker")
    if eventworker is None:
        raise RuntimeError(
            f"No default EventWorker registered; pass one explicitly: {func_name}(ew)."
        )
    return eventworker


def pulse_id(eventworker=None):
    """Module-level shorthand for ``eventworker.pulse_id`` -- a cached live
    Stream of each event's integer pulse ID.

    Uses the module-default EventWorker (the last one created with
    ``make_default=True``) if *eventworker* is omitted.
    """
    return _resolve_eventworker(eventworker, "pulse_id").pulse_id


def lab_time(eventworker=None):
    """Module-level shorthand for ``eventworker.lab_time`` -- a cached live
    Stream of each event's wall-clock time (seconds since epoch).

    Uses the module-default EventWorker (the last one created with
    ``make_default=True``) if *eventworker* is omitted.
    """
    return _resolve_eventworker(eventworker, "lab_time").lab_time


# ---------------------------------------------------------------------------
# Digitize / binning utilities
# ---------------------------------------------------------------------------

def digitize(data, edges, side="left"):
    data = np.atleast_1d(data)
    edges = np.asarray(edges)
    assert (np.diff(edges) >= 0).all(), "edges must be monotonically increasing"
    indices = edges.searchsorted(data, side=side)
    indout = (indices == 0) | (indices == len(edges))
    edgelower = np.full_like(data, np.nan, dtype=float)
    edgeupper = np.full_like(data, np.nan, dtype=float)
    edgelower[~indout] = edges[indices[~indout] - 1]
    edgeupper[~indout] = edges[indices[~indout]]
    bincenter = (edgeupper + edgelower) / 2.0
    return np.squeeze(bincenter), np.squeeze(edgelower), np.squeeze(edgeupper)


def digitizeEsc(escdata, edges, side="left"):  # noqa: N802
    po = ProcObj(
        digitize,
        args=[escdata, edges],
        returns_is_esc=[True, True, True],
        returns_names=[
            f"{escdata.name}_bincenter",
            f"{escdata.name}_edgelower",
            f"{escdata.name}_edgeupper",
        ],
        returns_units=[escdata.unit] * 3,
        scan=escdata.scan,
    )
    return po.createChildren()


def digitizeScan(escdata, edges, side="left"):  # noqa: N802
    escdats = digitizeEsc(escdata, edges, side=side)
    values = [
        (sum(edges[n : n + 2]) / 2.0, edges[n], edges[n + 1])
        for n in range(len(edges) - 1)
    ]
    return Scan(escdats, values=values)


def wrapFunc_singleOutput(func, name=None, unit=None, scan=None):  # noqa: N802
    name = name or "none"
    unit = unit or "none"

    def newFunc(*args, **kwargs):
        p = ProcObj(
            func,
            args=args,
            kwargs=kwargs,
            returns_is_esc=[True],
            returns_names=[name],
            returns_units=[unit],
            scan=Scan(),
        )
        return p.createChildren()[0]

    return newFunc


# ---------------------------------------------------------------------------
# Operator overloading on Stream
# ---------------------------------------------------------------------------

def _wrapOperatorJoin(func, symbol):
    def newFunc(*args):
        names = [getattr(a, "name", type(a).__name__) for a in args]
        units = [getattr(a, "unit", "no unit") for a in args]
        sep = f" {symbol} "
        p = ProcObj(
            func,
            args=args,
            returns_is_esc=[True],
            returns_names=[("(" + sep.join(names) + ")")],
            returns_units=[("(" + sep.join(units) + ")")],
            scan=args[0].scan,
        )
        return p.createChildren()[0]
    return newFunc


def _wrapOperatorSingle(func, symbol):
    def newFunc(*args):
        name = args[0].name
        p = ProcObj(
            func,
            args=args,
            returns_is_esc=[True],
            returns_names=[f"({symbol} {name})"],
            returns_units=[f"({symbol} {args[0].unit})"],
            scan=args[0].scan,
        )
        return p.createChildren()[0]
    return newFunc


_operatorsJoin = [
    (operator.add, "+"), (operator.truediv, "/"), (operator.floordiv, "//"),
    (operator.and_, "&"), (operator.xor, "^"), (operator.or_, "|"),
    (operator.pow, "**"), (operator.lshift, "<<"), (operator.mod, "%"),
    (operator.mul, "*"), (operator.rshift, ">>"), (operator.sub, "-"),
    (operator.lt, "<"), (operator.le, "<="), (operator.eq, "=="),
    (operator.ne, "!="), (operator.ge, ">="), (operator.gt, ">"),
]

_operatorsSingle = [
    (operator.neg, "-"), (operator.pos, "pos"),
]

for _opJoin, _sym in _operatorsJoin:
    setattr(Stream, f"__{_opJoin.__name__}__", _wrapOperatorJoin(_opJoin, _sym))
for _opSing, _sym in _operatorsSingle:
    setattr(Stream, f"__{_opSing.__name__}__", _wrapOperatorSingle(_opSing, _sym))

# ~stream uses logical NOT (not_ returns True/False) rather than bitwise invert
# so that i[~pump] works correctly when pump_on is a float 0.0/1.0 channel.
Stream.__invert__ = _wrapOperatorSingle(operator.not_, "not")

# EscData alias — keeps existing notebooks / code working unchanged
EscData = Stream
