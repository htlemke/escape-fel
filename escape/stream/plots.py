"""
Live-updating matplotlib plots for escape stream data.

Improvements over the original plots.py:
  - Periodic updates are driven by the GUI backend's own timer
    (``fig.canvas.new_timer()``, e.g. a real ``QTimer`` under Qt, Tk's
    ``after()`` under TkAgg) instead of a plain background ``threading.Thread``
    calling ``replot()``/``draw_idle()`` directly.  GUI toolkits require
    widgets to be touched only from the thread running their event loop;
    driving updates from an arbitrary Python thread violates that and — under
    Qt in particular — silently fails to repaint on its own, only catching up
    when some unrelated event (a resize, a window move) forces Qt to repaint
    anyway.  A backend timer callback runs on the correct thread by
    construction, so redraws happen every tick, independently, without ever
    blocking the GUI's own event loop (the callback itself is a fast, plain
    numpy + matplotlib update — no I/O, no waiting). Data accumulation
    continues to happen on the separate ``EventWorker`` background thread,
    untouched by this.
  - Each plot connects to matplotlib's 'close_event' so that:
      a) the update timer stops automatically, and
      b) the EscData object(s) stop accumulating.
    This fulfils the "stop acquisition when figure is destroyed" requirement.
  - Plots expose start() / stop() and a StreamContext through .context so the user
    can also drive acquisition lifetime manually or tie several plots to one context.
  - If the inline backend is detected a helpful warning is printed (live updates
    require an interactive backend such as ipympl / widget, Qt, or Tk).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_safe(fig):
    """Schedule a redraw. Safe to call from a backend timer callback (GUI thread)."""
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


def _draw_step_band(ax, x, y, yerr, label=None, alpha=0.3, y_floor=0.0):
    """Step line (``ax.step(..., where='mid')``) + matching stepped error
    band (``ax.fill_between(..., step='mid')``) -- shared by HistPlot and
    ValueHistPlot, which are otherwise near-identical. Same underlying
    matplotlib idiom as ``escape.plot_utilities.errortube(..., step='mid')``;
    kept as a separate, dependency-light copy here rather than importing
    that module, since ``escape.plot_utilities`` pulls in
    ``ipywidgets``/``dask``/``IPython`` that ``escape.stream`` doesn't
    otherwise require. Returns ``dict(line=..., err=...)`` for ``replot()``.
    """
    line = ax.step(x, y, where="mid", label=label)[0]
    color = line.get_color()
    lo = np.maximum(y - yerr, y_floor) if y_floor is not None else y - yerr
    err = ax.fill_between(x, y + yerr, lo, color=color, alpha=alpha, step="mid", lw=0)
    return {"line": line, "err": err}


def _redraw_step_band(ax, drawn, x, y, yerr, alpha=0.3, y_floor=0.0):
    """In-place update counterpart to :func:`_draw_step_band` for ``replot()``."""
    color = drawn["err"].get_facecolor()
    drawn["err"].remove()
    lo = np.maximum(y - yerr, y_floor) if y_floor is not None else y - yerr
    drawn["err"] = ax.fill_between(x, y + yerr, lo, color=color, alpha=alpha, step="mid", lw=0)
    drawn["line"].set_xdata(x)
    drawn["line"].set_ydata(y)
    return drawn


def _warn_inline():
    backend = matplotlib.get_backend()
    if "inline" in backend.lower():
        print(
            "WARNING: live plot updates require an interactive matplotlib backend.\n"
            "  In Jupyter, run  %matplotlib widget  (ipympl) before creating plots."
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _LivePlotBase:
    """Common infrastructure for all live-plot classes."""

    # Subclasses must set self.fig before calling _connect_close_event().
    # They must implement plot() and replot().

    def __init__(self, escdata_list, axes=None, update_interval=0.5):
        self._escdata = list(escdata_list)
        self.axes = axes
        self.fig = axes.get_figure() if axes is not None else None
        self.update_interval = update_interval
        self.isUpdating = False
        self._timer = None
        self.drawn = None
        self.autoscale = True

    # ------------------------------------------------------------------
    # Figure / close-event wiring
    # ------------------------------------------------------------------

    def _connect_close_event(self):
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        _warn_inline()

    def _on_close(self, event):
        self.stop()
        for ed in self._escdata:
            ed.accumulate(False)

    # ------------------------------------------------------------------
    # Update timer
    # ------------------------------------------------------------------
    #
    # Driven by fig.canvas.new_timer(), a backend-native timer (QTimer under
    # Qt, Tk's after() under TkAgg, ...) that fires on the GUI event loop
    # itself.  This is the only thread-safe way to touch a live figure
    # repeatedly: Qt/Tk widgets may only be accessed from the thread running
    # their event loop, and a plain background thread calling replot()/
    # draw_idle() silently fails to trigger a real repaint under Qt (it only
    # catches up when some unrelated GUI event forces one). On a
    # non-interactive backend (e.g. Agg) new_timer().start() is a documented
    # no-op, which is correct there -- there is no window to refresh.

    def _on_timer(self):
        try:
            self.replot()
        except Exception as exc:
            print(f"Live plot update error: {exc}")
        # Returning None (not 0/False) keeps the timer's callback registered.

    def start(self, interval=None):
        """Start periodic updates via the GUI backend's own timer."""
        if interval is not None:
            self.update_interval = interval
        if self.fig is None or self.fig.canvas is None:
            return
        if self._timer is None:
            self._timer = self.fig.canvas.new_timer(interval=int(self.update_interval * 1000))
            self._timer.add_callback(self._on_timer)
        else:
            self._timer.interval = int(self.update_interval * 1000)
        self.isUpdating = True
        self._timer.start()

    # Back-compat alias used by EscData.plot_* helpers.
    updateContinuously = start  # noqa: N815

    def stop(self):
        """Stop periodic updates."""
        self.isUpdating = False
        if self._timer is not None:
            self._timer.stop()

    # ------------------------------------------------------------------
    # Context / StreamContext integration
    # ------------------------------------------------------------------

    def tie_to_figure(self, fig=None):
        """Tie acquisition stop to figure close (chainable)."""
        if fig is None:
            fig = self.fig
        if fig is not None:
            fig.canvas.mpl_connect("close_event", self._on_close)
        return self

    def _autoscale(self):
        if self.autoscale and self.axes is not None:
            self.axes.relim()
            self.axes.autoscale_view(True, True, True)


# ---------------------------------------------------------------------------
# HistPlot — scan-step event-count histogram
# ---------------------------------------------------------------------------

class HistPlot(_LivePlotBase):
    """Step-count histogram (how many events collected per scan step).

    Parameters
    ----------
    data : EscData
        The data source being plotted.
    axes : matplotlib.axes.Axes, optional
        Target axes; a new figure is created if omitted.
    label : str, optional
    scanVariable : int
        Which scan parameter to use as the x-axis (default 0).
    alpha : float
        Fill transparency for the error band.
    autosetAxlabel : bool
        Automatically label axes from source metadata.
    """

    def __init__(
        self,
        data,
        axes=None,
        label=None,
        scanVariable=0,
        alpha=0.3,
        autosetAxlabel=True,
        update_interval=0.5,
    ):
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{data.name}  histogram")
        super().__init__([data], axes=axes, update_interval=update_interval)
        self.data = data
        self.label = label if label is not None else data.name
        self.scanVariable = scanVariable
        self.alpha = alpha
        self.autosetAxlabel = autosetAxlabel
        self._connect_close_event()

    def _getplotData(self):  # noqa: N802
        x = np.atleast_1d(np.asarray(self.data.scan[self.scanVariable]))
        # No-scan case: scan[0] returns a 0-d / None array — return empty.
        if x.dtype == object or (x.ndim == 1 and len(x) == 1 and x[0] is None):
            return np.array([]), np.array([]), np.array([])
        # True, uncapped event count per step -- NOT lens() (samples currently
        # retained), which is bounded by each step's deque maxlen (1000 by
        # default) and so plateaus at maxlen instead of showing the real count
        # once any step receives more events than that.
        y = np.asarray(self.data.counts(), dtype=float)
        yerr = np.sqrt(y)
        if len(x) == 0:
            return x, y, yerr
        sorter = x.argsort()
        return x[sorter], y[sorter], yerr[sorter]

    def plot(self):
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        self.drawn = _draw_step_band(self.axes, x, y, yerr, label=self.label, alpha=self.alpha)
        if self.autosetAxlabel and self.data.scan._parameters:
            par = self.data.scan._parameters[self.scanVariable]
            self.axes.set_xlabel(f"{par.name} / {par.unit}")
            self.axes.set_ylabel("event count")
        _draw_safe(self.fig)

    def replot(self):
        if self.drawn is None:
            return
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        self.drawn = _redraw_step_band(self.axes, self.drawn, x, y, yerr, alpha=self.alpha)
        self._autoscale()
        _draw_safe(self.fig)


# ---------------------------------------------------------------------------
# Plot — median + percentile error bands
# ---------------------------------------------------------------------------

class Plot(_LivePlotBase):
    """Median with percentile error bands vs scan variable.

    Parameters
    ----------
    data : EscData
    axes : matplotlib.axes.Axes, optional
    label : str, optional
    scanVariable : int
    errPercentiles : list of float
        Percentile widths to display as shaded bands (default: 1-sigma and 2-sigma).
    step : bool
        Use a step-plot style instead of a line.
    alpha : float
    autosetAxlabel : bool
    """

    def __init__(
        self,
        data,
        axes=None,
        label=None,
        scanVariable=0,
        errPercentiles=None,
        step=False,
        alpha=0.3,
        autosetAxlabel=True,
        update_interval=0.5,
    ):
        if errPercentiles is None:
            errPercentiles = [69.3, 95.0]
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{data.name}  median")
        super().__init__([data], axes=axes, update_interval=update_interval)
        self.data = data
        self.label = label if label is not None else data.name
        self.scanVariable = scanVariable
        self.errPercentiles = errPercentiles
        self.step = step
        self.alpha = alpha
        self.autosetAxlabel = autosetAxlabel
        self._connect_close_event()

    def _getplotData(self):  # noqa: N802
        try:
            x = np.asarray(self.data.scan[self.scanVariable])
            if len(x) == 0:
                return [], [], []
            sorter = x.argsort()
            y = np.asarray(self.data.median())[sorter]
            lens = np.asarray(self.data.lens())[sorter]
            yerr = []
            for perc in self.errPercentiles:
                band = (np.asarray(self.data.centerPerc(perc)).T)[:, sorter]
                # Shrink band by sqrt(n) to show uncertainty on median
                band = (band - y) / np.sqrt(np.maximum(lens, 1)) + y
                yerr.append(band)
            return x[sorter], y, yerr
        except Exception:
            return [], [], []

    def plot(self):
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        if self.step:
            line = self.axes.step(x, y, where="mid", label=self.label)[0]
        else:
            line = self.axes.plot(x, y, ".-", label=self.label)[0]
        color = line.get_color()
        errs = []
        for band in yerr:
            errs.append(
                self.axes.fill_between(
                    x, band[0], band[1],
                    color=color, alpha=self.alpha,
                    step="mid" if self.step else None,
                    lw=0,
                )
            )
        self.drawn = dict(err=errs, line=line)
        if self.autosetAxlabel and self.data.scan._parameters:
            par = self.data.scan._parameters[self.scanVariable]
            self.axes.set_xlabel(f"{par.name} / {par.unit}")
            self.axes.set_ylabel(f"{self.data.name} / {self.data.unit}")
        _draw_safe(self.fig)

    def replot(self):
        if self.drawn is None:
            return
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        new_errs = []
        for band, coll in zip(yerr, self.drawn["err"]):
            color = coll.get_facecolor()
            coll.remove()
            new_errs.append(
                self.axes.fill_between(
                    x, band[0], band[1],
                    color=color, alpha=self.alpha,
                    step="mid" if self.step else None,
                    lw=0,
                )
            )
        self.drawn["err"] = new_errs
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        self._autoscale()
        _draw_safe(self.fig)


# ---------------------------------------------------------------------------
# PlotCorrelation — event-matched scatter plot
# ---------------------------------------------------------------------------

class PlotCorrelation(_LivePlotBase):
    """Scatter plot of two channels matched by pulse-ID, showing the last N events.

    Parameters
    ----------
    data_x, data_y : EscData
        The x- and y-channel sources.
    Nlast : int
        Number of most recent matched events to display.
    axes : matplotlib.axes.Axes, optional
    label : str, optional
    autosetAxlabel : bool
    """

    def __init__(
        self,
        data_x,
        data_y,
        Nlast=200,
        axes=None,
        label=None,
        autosetAxlabel=True,
        update_interval=0.5,
        flatten_arrays=False,
    ):
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{data_y.name}  vs  {data_x.name}")
        super().__init__([data_x, data_y], axes=axes, update_interval=update_interval)
        self.data_x = data_x
        self.data_y = data_y
        self.Nlast = Nlast
        self.label = label if label is not None else data_y.name
        self.autosetAxlabel = autosetAxlabel
        # When both Streams are array-valued (same shape), a genuine
        # element-by-element correlation plot doesn't exist in 2D -- instead,
        # pool every element of every matched event together into one dense
        # scatter (rendered with low alpha; see plot()/replot()).
        self.flatten_arrays = flatten_arrays
        self._connect_close_event()

    def _getplotData(self):  # noqa: N802
        def flatten(ll):
            return [item for sub in ll for item in sub]

        xi = np.asarray(flatten(self.data_x.eventIds))
        yi = np.asarray(flatten(self.data_y.eventIds))
        xd = np.asarray(flatten(self.data_x.data))
        yd = np.asarray(flatten(self.data_y.data))

        if len(xi) == 0 or len(yi) == 0:
            return np.array([]), np.array([])

        xsel = np.isin(xi, yi)
        ysel = np.isin(yi, xi)
        xi, xd = xi[xsel], xd[xsel]
        yi, yd = yi[ysel], yd[ysel]

        if len(xi) == 0:
            return np.array([]), np.array([])

        mx = xi.max()
        xd = xd[xi > mx - self.Nlast]
        xi = xi[xi > mx - self.Nlast]
        yd = yd[yi > mx - self.Nlast]
        yi = yi[yi > mx - self.Nlast]

        xd = xd[np.argsort(xi)]
        yd = yd[np.argsort(yi)]
        if self.flatten_arrays:
            # xd/yd are (n_matched_events, *array_shape) here -- both operands
            # were checked to share array_shape before this mode is used, so
            # a plain elementwise ravel keeps (x_element, y_element) paired.
            xd = xd.reshape(-1)
            yd = yd.reshape(-1)
        return xd, yd

    def plot(self):
        x, y = self._getplotData()
        style = dict(alpha=0.15, ms=2) if self.flatten_arrays else {}
        line = self.axes.plot(x, y, ".", label=self.label, **style)[0]
        self.drawn = dict(line=line)
        if self.autosetAxlabel:
            self.axes.set_xlabel(f"{self.data_x.name} / {self.data_x.unit}")
            self.axes.set_ylabel(f"{self.data_y.name} / {self.data_y.unit}")
        _draw_safe(self.fig)

    def replot(self):
        if self.drawn is None:
            return
        x, y = self._getplotData()
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        self._autoscale()
        _draw_safe(self.fig)


# ---------------------------------------------------------------------------
# ValueHistPlot — value-distribution histogram (for no-scan EscData)
# ---------------------------------------------------------------------------

class ValueHistPlot(_LivePlotBase):
    """Distribution histogram of collected scalar values.

    Use this (or let ``EscData.plot_hist`` route to it automatically) when
    the EscData has no scan, i.e. all events land in a single step.

    Parameters
    ----------
    data : EscData
    axes : matplotlib.axes.Axes, optional
    label : str, optional
    n_bins : int
        Number of histogram bins (default 50).  Bins are recomputed from the
        current data range on each update.
    alpha : float
        Fill transparency for Poisson error band.
    """

    def __init__(
        self,
        data,
        axes=None,
        label=None,
        n_bins=50,
        alpha=0.35,
        update_interval=0.5,
    ):
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{data.name}  distribution")
        super().__init__([data], axes=axes, update_interval=update_interval)
        self.data = data
        self.label = label if label is not None else data.name
        self.n_bins = n_bins
        self.alpha = alpha
        self._connect_close_event()

    def _flat(self):
        return [v for step in self.data.data for v in step]

    def _getplotData(self):  # noqa: N802
        flat = self._flat()
        if len(flat) < 2:
            return np.array([]), np.array([]), np.array([])
        arr = np.asarray(flat, dtype=float)
        counts, edges = np.histogram(arr, bins=self.n_bins)
        centers = (edges[:-1] + edges[1:]) / 2
        return centers, counts.astype(float), np.sqrt(counts.astype(float))

    def plot(self):
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        self.drawn = _draw_step_band(self.axes, x, y, yerr, label=self.label, alpha=self.alpha)
        self.axes.set_xlabel(f"{self.data.name} / {self.data.unit}")
        self.axes.set_ylabel("counts")
        _draw_safe(self.fig)

    def replot(self):
        if self.drawn is None:
            return
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        # Bin count stays fixed (n_bins) but edges shift as data range grows —
        # _redraw_step_band() removes and recreates the fill_between for us.
        self.drawn = _redraw_step_band(self.axes, self.drawn, x, y, yerr, alpha=self.alpha)
        self._autoscale()
        _draw_safe(self.fig)


# ---------------------------------------------------------------------------
# TracePlot — generic "current value, redrawn periodically" plot
# ---------------------------------------------------------------------------

class TracePlot(_LivePlotBase):
    """Live view of a Stream's current value(s) -- works for scalar or
    array-valued data, and ignores scan structure entirely.

    For an **array-valued** Stream (e.g. a derived per-event waveform/trace,
    or a boolean event-code set), plots the single latest sample as a line,
    x = array index -- exactly "just redraw the current value periodically".

    For a **scalar** Stream, plots a rolling trend of the last ``n_history``
    samples, x = recent event index -- a live trend recorder.

    Parameters
    ----------
    data : Stream
    axes : matplotlib.axes.Axes, optional
    label : str, optional
    n_history : int
        Scalar case only: how many recent samples to show as a trend.
    """

    def __init__(
        self,
        data,
        axes=None,
        label=None,
        n_history=200,
        autosetAxlabel=True,
        update_interval=1.0,
    ):
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{data.name}  (live)")
        super().__init__([data], axes=axes, update_interval=update_interval)
        self.data = data
        self.label = label if label is not None else data.name
        self.n_history = n_history
        self.autosetAxlabel = autosetAxlabel
        self._connect_close_event()

    def _flat(self):
        return [v for step in self.data.data for v in step]

    def _getplotData(self):  # noqa: N802
        flat = self._flat()
        if not flat:
            return None, None, False
        latest = np.asarray(flat[-1])
        if latest.ndim >= 1 and latest.size > 1:
            return np.arange(latest.size), latest.astype(float), True
        window = np.asarray(flat[-self.n_history:], dtype=float)
        start = len(flat) - len(window)
        return np.arange(start, start + len(window)), window, False

    def plot(self):
        x, y, is_array = self._getplotData()
        if x is None:
            return
        style = "-" if is_array else ".-"
        line = self.axes.plot(x, y, style, label=self.label)[0]
        self.drawn = dict(line=line, is_array=is_array)
        if self.autosetAxlabel:
            self.axes.set_xlabel("array index" if is_array else "recent event #")
            self.axes.set_ylabel(f"{self.data.name} / {self.data.unit}")
        _draw_safe(self.fig)

    def replot(self):
        if self.drawn is None:
            return
        x, y, is_array = self._getplotData()
        if x is None:
            return
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        self._autoscale()
        _draw_safe(self.fig)


# ---------------------------------------------------------------------------
# WaterfallPlot — 2D live view of an array-valued Stream's recent history
# ---------------------------------------------------------------------------

class WaterfallPlot(_LivePlotBase):
    """2D image of an array-valued Stream's recent history: one row per
    event (most recent ``N_acc``), one column per array element.

    This is the array-valued equivalent of ``HistPlot`` -- ``plot_hist()``
    routes here automatically once it discovers the data is array-shaped.

    With ``x_stream`` given (e.g. ``pulse_id``/``lab_time``, or any other
    scalar Stream), rows are ordered/labeled by that stream's live values
    instead of plain arrival order -- this is also what ``plot_corr()`` uses
    to represent "array Stream vs. scalar Stream" correlation, since a
    per-element scatter isn't meaningful there.

    Parameters
    ----------
    data : Stream
        Array-valued Stream to display.
    x_stream : Stream, optional
        Scalar Stream whose per-event value labels each row (e.g. time).
    N_acc : int
        Number of most recent events (rows) to keep.
    axes : matplotlib.axes.Axes, optional
    cmap : str
    """

    def __init__(
        self,
        data,
        x_stream=None,
        N_acc=100,
        axes=None,
        cmap="viridis",
        autosetAxlabel=True,
        update_interval=0.5,
    ):
        if axes is None:
            fig, axes = plt.subplots()
            fig.suptitle(f"{data.name}  (live, last {N_acc} events)")
        escdata = [data] if x_stream is None else [data, x_stream]
        super().__init__(escdata, axes=axes, update_interval=update_interval)
        self.data = data
        self.x_stream = x_stream
        self.N_acc = N_acc
        self.cmap = cmap
        self.autosetAxlabel = autosetAxlabel
        self._connect_close_event()

    def _rows(self):
        flat = [v for step in self.data.data for v in step]
        rows = np.asarray(flat[-self.N_acc:], dtype=float) if flat else np.empty((0, 0))
        if self.x_stream is None or rows.size == 0:
            return rows, None
        xflat = [v for step in self.x_stream.data for v in step]
        xvals = np.asarray(xflat[-self.N_acc:], dtype=float) if xflat else None
        if xvals is None or len(xvals) != len(rows):
            xvals = None  # event counts of the two Streams diverged -- fall back to plain order
        return rows, xvals

    def plot(self):
        rows, xvals = self._rows()
        if rows.size == 0:
            return
        extent = [0, rows.shape[1], rows.shape[0], 0]
        im = self.axes.imshow(rows, aspect="auto", origin="upper", cmap=self.cmap, extent=extent)
        cbar = self.fig.colorbar(im, ax=self.axes, label=f"{self.data.name} / {self.data.unit}")
        self.drawn = dict(im=im, cbar=cbar, xvals=xvals)
        if self.autosetAxlabel:
            self.axes.set_xlabel("array index")
            if self.x_stream is not None:
                self.axes.set_ylabel(f"{self.x_stream.name} / {self.x_stream.unit} (row order)")
            else:
                self.axes.set_ylabel(f"recent event # (0 = oldest of last {self.N_acc})")
        _draw_safe(self.fig)

    def replot(self):
        if self.drawn is None:
            return
        rows, xvals = self._rows()
        if rows.size == 0:
            return
        im = self.drawn["im"]
        if rows.shape != im.get_array().shape:
            im.set_extent([0, rows.shape[1], rows.shape[0], 0])
        im.set_data(rows)
        im.set_clim(np.nanmin(rows), np.nanmax(rows))
        _draw_safe(self.fig)
