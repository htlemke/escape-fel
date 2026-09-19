"""Graphical range selection on the histogram of a 1-D escape ``Array``.

Two matplotlib-only front-ends (no ipywidgets/Qt needed, so they work on any
*interactive* matplotlib backend -- ``%matplotlib widget``/``qt``/``tk``):

* :class:`HistogramFilter` -- drag a span on the histogram, get
  ``array.filter(lo, hi)``.
* :class:`HistogramDigitizer` -- drag a span for the region to digitize and
  enter either a number of bins or a bin size, get
  ``array.digitize(edges)``.

Both are reached through :meth:`escape.Array.filter_interactive` /
:meth:`escape.Array.digitize_interactive`. Keep a reference to the returned
object: its ``.result`` is computed from whatever is selected at the time you
ask for it.
"""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, SpanSelector, TextBox

# Digitizing into more bins than this is almost certainly a typo (e.g. a bin
# size entered in the wrong unit); refuse rather than draw/compute it.
MAX_BINS = 10000
# Above this many bin edges the edge overlay is skipped (it would only be a
# solid block of lines).
MAX_DRAWN_EDGES = 500


def _fmt(x):
    return f"{x:.6g}"


class _HistRangeSelector:
    """Histogram + draggable span + editable min/max boxes.

    Subclasses add their own controls in ``_build_controls`` and refresh their
    status text/overlays in ``_update`` (called whenever the selected range
    changes).
    """

    _op_name = None  # "filter" / "digitize", used in messages and the title

    def __init__(self, array, bins="auto", cut_percentage=0, figsize=(8, 6)):
        if not np.prod(np.asarray(array.shape)) == array.shape[array.index_dim]:
            raise NotImplementedError(
                f"Only 1d escape arrays can be {self._op_name}d interactively."
            )
        if "inline" in matplotlib.get_backend().lower():
            warnings.warn(
                "The inline matplotlib backend is not interactive; the span "
                "selector needs e.g. `%matplotlib widget` or `%matplotlib qt`.",
                stacklevel=3,
            )
        self.array = array

        values = array.data.ravel()
        if hasattr(values, "compute"):
            from escape.storage.storage import _eagerly_compute_dask

            values = _eagerly_compute_dask(values, f"{self._op_name}_interactive", array.name)
        values = np.asarray(values, dtype=float)
        self._values = values[np.isfinite(values)]
        if self._values.size == 0:
            raise ValueError(f"{array.name!r} has no finite values to histogram.")

        hmin, hmax = np.percentile(self._values, [cut_percentage, 100 - cut_percentage])
        if not hmin < hmax:
            hmin, hmax = float(self._values.min()) - 0.5, float(self._values.max()) + 0.5
        edges = np.histogram_bin_edges(self._values, bins, range=(hmin, hmax))
        counts, edges = np.histogram(self._values, bins=edges)

        self.fig = plt.figure(figsize=figsize)
        # Keep the widgets alive even if the caller doesn't hold on to us
        # (matplotlib widgets are only weakly referenced by the canvas).
        self.fig._escape_hist_select = self
        self.ax = self.fig.add_axes([0.09, 0.40, 0.88, 0.54])
        self.ax.stairs(counts, edges, color="0.35")
        self.ax.set_xlim(edges[0], edges[-1])
        self.ax.set_xlabel(array._labeled_name() or "value")
        self.ax.set_ylabel("events")
        self.ax.set_title(f"{array.name or 'array'}: drag to select a range ({self._op_name})")

        self._updating = False
        self._range = (float(edges[0]), float(edges[-1]))
        self.span = SpanSelector(
            self.ax,
            self._on_span,
            "horizontal",
            interactive=True,
            drag_from_anywhere=True,
            onmove_callback=self._on_span,
            props=dict(alpha=0.25, facecolor="tab:blue"),
            button=[1],
        )
        self.span.extents = self._range

        self._tb_lo = TextBox(self.fig.add_axes([0.09, 0.26, 0.16, 0.06]), "min ")
        self._tb_hi = TextBox(self.fig.add_axes([0.34, 0.26, 0.16, 0.06]), "max ")
        self._tb_lo.on_submit(self._on_box)
        self._tb_hi.on_submit(self._on_box)
        self._info = self.fig.text(0.09, 0.15, "", va="center", ha="left")
        self._code = self.fig.text(
            0.09, 0.07, "", va="center", ha="left", family="monospace", color="0.3"
        )
        self._build_controls()
        self._set_range(*self._range)

    # -- range handling ---------------------------------------------------

    @property
    def range(self):
        """The currently selected ``(lo, hi)``."""
        return self._range

    @range.setter
    def range(self, lo_hi):
        self._set_range(*lo_hi, move_span=True)

    def _on_span(self, lo, hi):
        self._set_range(lo, hi)

    def _on_box(self, _text):
        if self._updating:
            return
        try:
            lo, hi = float(self._tb_lo.text), float(self._tb_hi.text)
        except ValueError:
            self._set_range(*self._range)  # put the last valid values back
            return
        self._set_range(lo, hi, move_span=True)

    def _set_range(self, lo, hi, move_span=False):
        lo, hi = sorted((float(lo), float(hi)))
        self._range = (lo, hi)
        if move_span:
            self.span.extents = (lo, hi)
        self._updating = True
        try:
            self._tb_lo.set_val(_fmt(lo))
            self._tb_hi.set_val(_fmt(hi))
        finally:
            self._updating = False
        self._update()
        self.fig.canvas.draw_idle()

    # -- subclass hooks -----------------------------------------------------

    def _build_controls(self):
        pass

    def _update(self):
        raise NotImplementedError


class HistogramFilter(_HistRangeSelector):
    """Pick a value range on a histogram and filter the Array to it.

    Created by :meth:`escape.Array.filter_interactive`. Drag on the histogram
    (drag the edges or the whole span to adjust) or type exact ``min``/``max``
    values; the status line shows how many events are kept and the equivalent
    ``.filter(lo, hi)`` call.

    Attributes
    ----------
    range : tuple[float, float]
        Selected ``(lo, hi)``; assignable, moves the span.
    result : escape.Array
        ``array.filter(lo, hi)`` for the current selection (both ends
        inclusive), computed on access.
    """

    _op_name = "filter"

    def _update(self):
        lo, hi = self._range
        kept = np.count_nonzero((self._values >= lo) & (self._values <= hi))
        total = self._values.size
        self._info.set_text(f"keeps {kept} of {total} events ({100 * kept / total:.1f}%)")
        self._code.set_text(f".filter({_fmt(lo)}, {_fmt(hi)})")

    @property
    def result(self):
        lo, hi = self._range
        return self.array.filter(lo, hi)


class HistogramDigitizer(_HistRangeSelector):
    """Pick the region to digitize on a histogram and how to bin it.

    Created by :meth:`escape.Array.digitize_interactive`. The selected span is
    the region to digitize; the bins are specified either as a **number of
    bins** (equal bins spanning exactly the selection) or as a **bin size**
    (equal bins starting at ``min`` and lying fully inside the selection, so a
    remainder smaller than one bin at the top is left out). Pick the mode with
    the radio buttons and enter the value in the ``value`` box; the bin edges
    are drawn over the histogram.

    Parameters
    ----------
    array : escape.Array
        1-D Array to digitize.
    n_bins : int
        Initial number of bins.
    bins, cut_percentage, figsize
        Histogram display settings (``numpy.histogram_bin_edges`` bins rule,
        percentage of outliers cut from each end of the displayed range, and
        figure size).
    **digitize_kwargs
        Forwarded to :func:`escape.storage.storage.digitize` for ``.result``
        (e.g. ``right=True``, ``include_outlier_bins=True``).

    Attributes
    ----------
    range : tuple[float, float]
        Selected ``(lo, hi)``; assignable, moves the span.
    bins : numpy.ndarray or None
        Current bin edges, ``None`` while the bin specification is invalid.
    result : escape.Array
        ``array.digitize(bins, **digitize_kwargs)`` for the current selection.
    """

    _op_name = "digitize"

    def __init__(self, array, n_bins=10, bins="auto", cut_percentage=0, figsize=(8, 6), **digitize_kwargs):
        self.digitize_kwargs = digitize_kwargs
        self._initial_value = str(n_bins)
        self._edge_lines = None
        self._bins = None
        super().__init__(array, bins=bins, cut_percentage=cut_percentage, figsize=figsize)

    def _build_controls(self):
        self._radio = RadioButtons(
            self.fig.add_axes([0.72, 0.03, 0.25, 0.29], frame_on=False),
            ("number of bins", "bin size"),
        )
        self._tb_val = TextBox(
            self.fig.add_axes([0.60, 0.26, 0.10, 0.06]), "value ", initial=self._initial_value
        )
        self._radio.on_clicked(lambda _label: self._refresh())
        self._tb_val.on_submit(lambda _text: self._refresh())

    def _refresh(self):
        self._update()
        self.fig.canvas.draw_idle()

    def _compute_bins(self):
        lo, hi = self._range
        if not hi > lo:
            raise ValueError("selected range is empty")
        try:
            value = float(self._tb_val.text)
        except ValueError:
            raise ValueError(f"{self._tb_val.text!r} is not a number") from None
        if self._radio.value_selected == "number of bins":
            if value < 1 or value != int(value):
                raise ValueError("number of bins must be a positive integer")
            edges = np.linspace(lo, hi, int(value) + 1)
        else:
            if not value > 0:
                raise ValueError("bin size must be positive")
            n = int(np.floor((hi - lo) / value + 1e-9))
            if n < 1:
                raise ValueError("bin size is larger than the selected range")
            if n > MAX_BINS:
                raise ValueError(f"bin size gives {n} bins (max {MAX_BINS})")
            edges = lo + value * np.arange(n + 1)
        if len(edges) - 1 > MAX_BINS:
            raise ValueError(f"{len(edges) - 1} bins requested (max {MAX_BINS})")
        return edges

    def _update(self):
        if self._edge_lines is not None:
            self._edge_lines.remove()
            self._edge_lines = None
        try:
            edges = self._compute_bins()
        except ValueError as exc:
            self._bins = None
            self._info.set_text(f"invalid bins: {exc}")
            self._info.set_color("tab:red")
            self._code.set_text("")
            return
        self._bins = edges
        self._info.set_color("black")
        nb = len(edges) - 1
        text = f"{nb} bins of width {_fmt((edges[-1] - edges[0]) / nb)}, {_fmt(edges[0])} to {_fmt(edges[-1])}"
        if not self.digitize_kwargs.get("include_outlier_bins"):
            inds = np.digitize(self._values, edges, right=self.digitize_kwargs.get("right", False))
            binned = np.count_nonzero((inds > 0) & (inds < len(edges)))
            text += f"; {binned} of {self._values.size} events binned"
        self._info.set_text(text)
        self._code.set_text(f".digitize(np.linspace({_fmt(edges[0])}, {_fmt(edges[-1])}, {nb + 1}))")
        if len(edges) <= MAX_DRAWN_EDGES:
            self._edge_lines = self.ax.vlines(
                edges, 0, 1, transform=self.ax.get_xaxis_transform(), colors="tab:orange", linewidth=0.8
            )

    @property
    def bins(self):
        return self._bins

    @property
    def result(self):
        if self._bins is None:
            raise ValueError("the bin specification is invalid; see the status line of the figure")
        return self.array.digitize(self._bins, **self.digitize_kwargs)
