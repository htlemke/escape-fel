"""Shared "pick a line, optionally pick an x-range" logic for escape's
axes-attached analysis tools (:mod:`escape.fit_gui`, :mod:`escape.freq_gui`).

Not public API -- both modules build their own domain logic (fitting,
spectral analysis) on top of this; import from here only within escape.
"""

from __future__ import annotations

import numpy as np
from matplotlib.widgets import SpanSelector


class AxesRangeSelector:
    """Owns picking a target ``Line2D`` on an ``Axes`` and an optional
    x-range within it.

    Captures the target line once, at construction, *before* any caller
    creates a ``SpanSelector`` on the same axes -- an interactive
    ``SpanSelector`` adds its own ``Line2D`` edge-handle artists to the
    axes, so re-querying ``ax.get_lines()`` later risks picking up a
    widget-internal line instead of data.
    """

    def __init__(self, ax, line=None):
        self.ax = ax
        self.fig = ax.figure
        self.range = None
        data_lines = list(ax.get_lines())
        self.line = line if line is not None else (data_lines[-1] if data_lines else None)
        self._pickable_lines = data_lines
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def enable_picking(self):
        for ln in self._pickable_lines:
            ln.set_picker(True)
            ln.set_pickradius(5)

    def _on_pick(self, event):
        if event.artist in self._pickable_lines:
            self.line = event.artist
            print(f"[escape] target line set to {self.line.get_label()!r}")

    def set_range(self, xmin, xmax):
        if xmax > xmin:
            self.range = (xmin, xmax)
            print(f"[escape] range set to [{xmin:.6g}, {xmax:.6g}]")

    def target_line(self):
        if self.line is None:
            raise RuntimeError(
                "No line selected -- pass line=..., or click a data line on "
                "the axes to select it."
            )
        return self.line

    def initial_xrange(self):
        """(xmin, xmax) of the target line's data, or (0.0, 1.0) if there is none yet."""
        if self.line is None:
            return 0.0, 1.0
        x = np.asarray(self.line.get_xdata(), dtype=float)
        return float(x.min()), float(x.max())

    def xy_selection(self):
        line = self.target_line()
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if self.range is not None:
            xmin, xmax = self.range
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]
        return x, y

    def make_span_selector(self, on_select, active=False):
        span = SpanSelector(
            self.ax,
            on_select,
            direction="horizontal",
            useblit=False,
            interactive=True,
            props=dict(alpha=0.15, facecolor="tab:orange"),
        )
        span.set_active(active)
        return span
