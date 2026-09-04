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
        self._lastEventId = None

    def append(self, data, eventId, index=None):
        if eventId is None or eventId == self._lastEventId:
            return
        self._lastEventId = eventId
        if index is None:
            doappend, index = self.scan._append()
        if doappend is None:
            return
        if doappend:
            self._data.append(deque(maxlen=self._data[0].maxlen if self._data else 1000))
            self._eventIds.append(deque(maxlen=self._eventIds[0].maxlen if self._eventIds else 1000))
        self._data[index].append(data)
        self._eventIds[index].append(eventId)

    def _getDataShape(self):
        lens = self.lens()
        if not lens or max(lens) == 0:
            return None
        best = lens.index(max(lens))
        return np.shape(list(self._data[best])[0])

    def __len__(self):
        return sum(self.lens())

    def lens(self):
        result = []
        for n, (te, td) in enumerate(zip(self._eventIds, self._data)):
            if len(te) == len(td):
                result.append(len(te))
            else:
                print(f"DataManager: mismatch in step {n}")
                result.append(min(len(te), len(td)))
        return result

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

    def __init__(self, source, mask_stream):
        self._inner = source
        self._mask = mask_stream
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
        return self.procObj.ret_values[self.returnIndex]


class FileSource:
    """Placeholder for indexed file sources."""
    pass


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
    """

    def __init__(self, eventHandler=None, make_default=True):
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

    def registerSource(self, sourceID):  # noqa: N802
        self._eventHandler.register_source(sourceID)
        if self._needs_restart():
            self._schedule_restart()   # debounced: batches rapid channel additions
        elif not self.loopThread or not self.loopThread.is_alive():
            self.startEventLoop()

    def removeSource(self, sourceID):  # noqa: N802
        self._eventHandler.remove_source(sourceID)
        if self._needs_restart():
            self._schedule_restart()

    def registerSources(self, *sourceIDs):  # noqa: N802
        """Register several sources in one stop/start cycle (no extra debounce needed)."""
        for sid in sourceIDs:
            self._eventHandler.register_source(sid)
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
        self.stopEventLoop()
        self.startEventLoop()

    def eventLoop(self):  # noqa: N802
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                ctx = self._eventHandler.context_manager()
                with ctx as s:
                    backoff = 1.0  # reset on successful connect
                    while not self._stop_event.is_set():
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
                                except Exception as exc:
                                    print(f"EventWorker callback error: {exc}")
                        time.sleep(0.001)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                print(f"EventWorker reconnecting in {backoff:.0f}s (error: {exc})")
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2, 30.0)

    def startEventLoop(self):  # noqa: N802
        self._stop_event.clear()
        self.loopThread = threading.Thread(target=self.eventLoop, daemon=True)
        self.loopThread.start()

    def stopEventLoop(self):  # noqa: N802
        self._stop_event.set()
        if self.loopThread is not None and self.loopThread.is_alive():
            self.loopThread.join(timeout=5.0)


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
        return Stream(source=target_stream._source, scan=scan)

    def __repr__(self):
        return (
            f"StreamBinning(key={self.key_stream.name!r}, "
            f"n_bins={len(self.bins)-1}, "
            f"range=[{self.bins[0]:.3g}, {self.bins[-1]:.3g}])"
        )


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
    # Data access
    # ------------------------------------------------------------------

    def shape(self):
        return self._dataManager._getDataShape()

    def lens(self):
        return self._dataManager.lens()

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
        """Start or stop accumulation.  Toggle with no argument."""
        if do_accumulate is None:
            do_accumulate = not self._is_accumulating()
            print(f"Toggling accumulation {'ON' if do_accumulate else 'OFF'} for {self.name!r}")

        ew = self._source.eventWorker
        if do_accumulate:
            ew.registerSource(self._source.name)
            if self._appendEventData not in ew.eventCallbacks:
                ew.eventCallbacks.append(self._appendEventData)
        else:
            try:
                ew.eventCallbacks.remove(self._appendEventData)
            except ValueError:
                pass
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
        return [np.percentile(td, pervals, axis=0) for td in self.data]

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
        For SwissFEL dispatcher-based connections, make sure the mask channel is
        subscribed (call ``mask_stream.accumulate(True)`` or register it
        separately).  With LocalEventHandler all channels are always received.

        See Also
        --------
        Stream.__getitem__ : ``i[pump]`` is shorthand for ``i.filter(pump)``.
        """
        filtered_src = FilteredEventSource(self._source, mask_stream)
        return Stream(source=filtered_src)

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
        """
        if isinstance(key, Stream):
            return self.filter(key)
        if isinstance(key, slice):
            all_data = [v for step in self.data for v in step]
            return np.array(all_data[key])
        raise TypeError(
            f"Stream index must be a Stream (for filtering) or a slice "
            f"(for data access), not {type(key).__name__}"
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

        Parameters
        ----------
        other_stream : Stream

        Returns
        -------
        Stream
        """
        return Stream(source=other_stream._source, scan=self.scan)

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
        return Stream(source=self._source, scan=s)

    # ------------------------------------------------------------------
    # Live plots
    # ------------------------------------------------------------------

    def _wait_for_data(self, timeout=5):
        deadline = time.time() + timeout
        while len(self) < 2:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for data from {self.name!r}.")
            time.sleep(0.05)

    def plot_hist(self, update=0.5, axes=None, timeout=5, n_bins=50):
        """Value-distribution or scan-count histogram with live updates.

        Routes to :class:`ValueHistPlot` (no scan) or :class:`HistPlot` (scan).
        """
        self.accumulate(True)
        has_scan = self.scan._parameters is not None
        if axes is None:
            fig, axes = plt.subplots()
            title = (
                f"{self.name}  histogram"
                if has_scan
                else f"{self.name}  value distribution"
            )
            axes.figure.suptitle(title)
        self._wait_for_data(timeout)
        if has_scan:
            hp = plots.HistPlot(self, axes=axes)
        else:
            hp = plots.ValueHistPlot(self, axes=axes, n_bins=n_bins)
        hp.plot()
        if update:
            hp.start(interval=update)
        self._histPlot = hp
        return hp

    def plot_med(self, update=0.5, axes=None, timeout=5):
        """Median + percentile bands with live updates."""
        self.accumulate(True)
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{self.name}  median")
        self._wait_for_data(timeout)
        mp = plots.Plot(self, axes=axes)
        mp.plot()
        if update:
            mp.start(interval=update)
        self._medPlot = mp
        return mp

    def plot_corr(self, xVar, Npoints=300, update=0.5, axes=None, timeout=5):
        """Scatter-plot correlation against *xVar* with live updates."""
        self.accumulate(True)
        xVar.accumulate(True)
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{self.name}  vs  {xVar.name}")
        self._wait_for_data(timeout)
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
