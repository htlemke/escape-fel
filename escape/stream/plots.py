"""
Live-updating matplotlib plots for escape stream data.

Improvements over the original plots.py:
  - Drawing calls use fig.canvas.draw_idle() instead of plt.draw() / plt.draw_all().
    draw_idle() schedules the redraw on the GUI thread, making it safe to call from
    the background update thread.  Works correctly with Qt, Tk, and the ipympl
    (%matplotlib widget) backends used in Jupyter.
  - Each plot connects to matplotlib's 'close_event' so that:
      a) the background update thread stops automatically, and
      b) the EscData object(s) stop accumulating.
    This fulfils the "stop acquisition when figure is destroyed" requirement.
  - Plots expose start() / stop() and a StreamContext through .context so the user
    can also drive acquisition lifetime manually or tie several plots to one context.
  - If the inline backend is detected a helpful warning is printed (live updates from
    background threads require an interactive backend such as ipympl / widget).
"""

import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_safe(fig):
    """Schedule a redraw on the GUI thread without blocking the caller."""
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


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

    _min_sleep = 0.01

    def __init__(self, escdata_list, axes=None, update_interval=0.5):
        self._escdata = list(escdata_list)
        self.axes = axes
        self.fig = axes.get_figure() if axes is not None else None
        self.update_interval = update_interval
        self.isUpdating = False
        self._update_thread = None
        self.drawn = None
        self._last_draw_dur = 0.0
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
    # Update thread
    # ------------------------------------------------------------------

    def _update_loop(self):
        while self.isUpdating:
            t0 = time.time()
            try:
                self.replot()
            except Exception as exc:
                print(f"Live plot update error: {exc}")
            elapsed = time.time() - t0
            self._last_draw_dur = elapsed
            sleep_t = max(self.update_interval - elapsed, self._min_sleep)
            time.sleep(sleep_t)

    def start(self, interval=None):
        """Start the background update thread."""
        if interval is not None:
            self.update_interval = interval
        if self.isUpdating:
            return
        self.isUpdating = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    # Back-compat alias used by EscData.plot_* helpers.
    updateContinuously = start  # noqa: N815

    def stop(self):
        """Stop the background update thread."""
        self.isUpdating = False
        t = self._update_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)

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
        y = np.asarray(self.data.lens(), dtype=float)
        yerr = np.sqrt(y)
        if len(x) == 0:
            return x, y, yerr
        sorter = x.argsort()
        return x[sorter], y[sorter], yerr[sorter]

    def plot(self):
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            return
        line = self.axes.step(x, y, where="mid", label=self.label)[0]
        color = line.get_color()
        err = self.axes.fill_between(
            x,
            y + yerr,
            np.maximum(y - yerr, 0),
            color=color,
            alpha=self.alpha,
            step="mid",
            lw=0,
        )
        self.drawn = dict(err=err, line=line)
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
        color = self.drawn["err"].get_facecolor()
        self.drawn["err"].remove()
        self.drawn["err"] = self.axes.fill_between(
            x,
            y + yerr,
            np.maximum(y - yerr, 0),
            color=color,
            alpha=self.alpha,
            step="mid",
            lw=0,
        )
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
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
        return xd, yd

    def plot(self):
        x, y = self._getplotData()
        line = self.axes.plot(x, y, ".", label=self.label)[0]
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
        line = self.axes.step(x, y, where="mid", label=self.label)[0]
        color = line.get_color()
        err = self.axes.fill_between(
            x,
            np.maximum(y - yerr, 0),
            y + yerr,
            color=color,
            alpha=self.alpha,
            step="mid",
            lw=0,
        )
        self.drawn = dict(err=err, line=line)
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
        # remove and recreate fill_between; update step line data directly.
        color = self.drawn["err"].get_facecolor()
        self.drawn["err"].remove()
        self.drawn["err"] = self.axes.fill_between(
            x,
            np.maximum(y - yerr, 0),
            y + yerr,
            color=color,
            alpha=self.alpha,
            step="mid",
            lw=0,
        )
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        self._autoscale()
        _draw_safe(self.fig)
