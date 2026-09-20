"""Live plots and parameter panels for Arrays with lineage.

Opt-in module (not imported by ``escape/__init__``). The machinery that
records how an Array was made lives in :mod:`escape.storage.lineage`; this
module is its front end for notebooks (ipympl figures + ipywidgets)::

    from escape import live

    res = sig / i0.filter_interactive(0.5, 2)   # histogram with draggable limits
    res.plot(live=True)                         # redraws when the limits change
    live.panel(res)                             # input fields for every Param upstream

    # your own function: pass it a Param
    @escape.escaped
    def scale(x, factor): return x * factor
    f = live.Param(2.0, "factor")
    scale(sig, f).plot(live=True, params="all")   # plot + a 'factor' input field

Anything that draws an Array can be made live: ``live.live_plot(res, draw)``
with ``draw(ax, current_array)``.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from .storage import lineage as _lineage
from .storage.lineage import Param, batch, describe, upstream_params

logger = logging.getLogger(__name__)

__all__ = [
    "Param",
    "batch",
    "describe",
    "upstream_params",
    "LivePlot",
    "live_plot",
    "panel",
    "set_enabled",
    "is_enabled",
]


def set_enabled(flag=True):
    """Switch lineage recording on/off globally (default on). Off: Arrays
    stop recording how they were made (Params passed to operations still work,
    they just don't make results live). Also settable with the environment
    variable ``ESCAPE_LINEAGE=0`` before importing escape."""
    _lineage.ENABLED = bool(flag)


def is_enabled():
    return _lineage.ENABLED


# ---------------------------------------------------------------------------
# Parameter input fields
# ---------------------------------------------------------------------------


def _select_params(array, params):
    """The Params of ``array``'s lineage picked by ``params``: ``None`` -> none,
    ``"all"`` -> all, else names and/or Param objects."""
    if params is None or params is False:
        return []
    available = upstream_params(array)
    if params is True or (isinstance(params, str) and params == "all"):
        return available
    if isinstance(params, (str, Param)):
        params = [params]
    chosen = []
    for want in params:
        if isinstance(want, Param):
            chosen.append(want)
            continue
        match = [p for p in available if p.name == want]
        if not match:
            raise ValueError(f"no Param named {want!r} upstream; available: {[p.name for p in available]}")
        chosen.append(match[0])
    return chosen


def _widget_for(p):
    import ipywidgets as w

    v = p.value
    kw = dict(description=p.name, style={"description_width": "initial"}, layout=w.Layout(width="95%"))
    if isinstance(v, (bool, np.bool_)):
        widget = w.Checkbox(value=bool(v), indent=False, **kw)
    elif isinstance(v, (int, np.integer)):
        widget = w.IntText(value=int(v), **kw)
    elif isinstance(v, (float, np.floating)):
        if p.bounds:
            lo, hi = p.bounds
            widget = w.FloatSlider(value=float(v), min=lo, max=hi, step=(hi - lo) / 200,
                                   readout_format=".4g", continuous_update=False, **kw)
        else:
            widget = w.FloatText(value=float(v), **kw)
    else:  # arrays etc.: shown, not edited here (the tool that made it edits it)
        widget = w.Text(value=_lineage._short_val(v), disabled=True, **kw)
        return widget, lambda: setattr(widget, "value", _lineage._short_val(p.value)), None

    def to_widget():
        if widget.value != p.value:
            widget.value = p.value

    def to_param(change):
        p.set(change["new"])

    return widget, to_widget, to_param


def panel(obj, params="all"):
    """Input fields for Params, two-way bound: editing a field sets the Param
    (results/live plots downstream update), and a Param changed elsewhere --
    e.g. by dragging a histogram span -- updates its field.

    ``obj`` is an Array (its lineage is shown and ``params`` picks from its
    upstream Params: ``"all"`` or names) or a list of Params. Returns the
    ipywidgets box; display it (last expression in a cell) or call
    ``IPython.display.display``.
    """
    try:
        import ipywidgets as w
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("live.panel needs ipywidgets (pip install ipywidgets)") from exc

    if isinstance(obj, (list, tuple)):
        plist, header = list(obj), None
    else:
        plist, header = _select_params(obj, params), describe(obj)
    rows = []
    if header:
        rows.append(w.HTML(f"<code>{_escape_html(header)}</code>"))
    removers = []
    for p in plist:
        widget, to_widget, to_param = _widget_for(p)
        if to_param is not None:
            widget.observe(to_param, names="value")
        removers.append(p.observe(to_widget))
        rows.append(widget)
    if not plist:
        rows.append(w.HTML("<i>no tunable parameters upstream</i>"))
    box = w.VBox(rows)
    box._param_removers = removers
    return box


def _escape_html(text):
    import html

    return html.escape(text)


# ---------------------------------------------------------------------------
# Live plot
# ---------------------------------------------------------------------------


class LivePlot:
    """A figure that redraws itself when the Params behind an Array change.

    Usually created through ``array.plot(live=True)`` or
    ``scan.plot(live=True)``; use :func:`live_plot` to make any drawing
    function live.

    Parameters
    ----------
    array : escape.Array
        The result being shown; its lineage is what gets re-evaluated.
    draw : callable
        ``draw(ax, current_array)`` -- draws on a *cleared* ``ax`` (it is
        called again on every update, with the freshly evaluated Array).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; default: a new figure.
    params : "all" | list of names/Params, optional
        Also show these Params as input fields below the plot (notebook).
    """

    def __init__(self, array, draw, ax=None, params=None):
        self.array = array
        self.draw = draw
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.fig = ax.figure
        self._base_axes = list(self.fig.axes)
        self._removers = []
        self.panel = None
        for p in upstream_params(array):
            self._removers.append(p.observe(self.refresh))
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.close())
        # Keep alive even if the caller drops the return value.
        self.fig._escape_live_plots = getattr(self.fig, "_escape_live_plots", []) + [self]
        self.refresh()
        if params:
            self.panel = panel(array, params=params)
            try:
                from IPython import get_ipython
                from IPython.display import display

                if get_ipython() is not None:
                    display(self.panel)
            except ImportError:  # pragma: no cover
                pass

    def refresh(self):
        """Re-evaluate the Array with the current Param values and redraw."""
        if not plt.fignum_exists(self.fig.number):
            self.close()
            return
        for extra in [a for a in self.fig.axes if a not in self._base_axes]:
            extra.remove()  # e.g. colorbars added by the previous draw
        self.ax.clear()
        plt.sca(self.ax)
        try:
            self.draw(self.ax, self.array.evaluate())
        except Exception as exc:  # e.g. a selection that leaves no events
            logger.debug("live plot could not draw", exc_info=True)
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"cannot draw: {exc}", transform=self.ax.transAxes,
                         ha="center", va="center", color="tab:red", wrap=True)
        self.fig.canvas.draw_idle()

    def close(self):
        """Stop following the Params (also happens when the figure closes)."""
        for remove in self._removers:
            remove()
        self._removers = []


def live_plot(array, draw, ax=None, params=None):
    """Make a drawing function live; see :class:`LivePlot`."""
    return LivePlot(array, draw, ax=ax, params=params)
