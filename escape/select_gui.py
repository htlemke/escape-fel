"""Engine + front-ends behind escape's "Select curves" toolbar button
(:func:`escape.plot_utilities.attach_select_button`).

Distilled from a tested prototype in eco's archiver plotting
(``eco.dbase.archiver``, ``_antialiasing`` ... ``_attach_curve_selector``) --
see that module's history for the original. This version follows the same
engine-shared-by-two-front-ends split as :mod:`escape.fit_gui` /
:mod:`escape.freq_gui` (``_FitEngine``/``_FreqEngine`` there,
:class:`_CurveSelectEngine` here): a pure-matplotlib engine that knows how to
show/hide curves and keep the legend/y-range consistent, with a Qt dialog and
an ipywidgets panel both driving it. Only matplotlib is imported at module
level; ``qtpy``/``ipywidgets`` are imported lazily inside the functions that
need them, so importing this module (and so attaching the button in the
first place) costs nothing on a backend/environment without either.
"""

from __future__ import annotations

import re

import matplotlib.colors as mcolors

# A line with no explicit label gets matplotlib's own auto-generated one,
# stable across versions: "_child%d" (the artist's position among *all*
# children added to the axes, not just lines). Shown as "(line N)" instead
# of the raw underscore-prefixed form, which is also what hides it from a
# real ``ax.legend()`` call.
_AUTO_LABEL_RE = re.compile(r"^_child(\d+)$")


def _display_label(line, index):
    """A human-friendly label for ``line`` in the curve list: its own label,
    unless that's one of matplotlib's auto-generated "no legend entry"
    labels (leading underscore) -- then ``"(line N)"``, using the number
    embedded in the auto label itself when there is one (``_child3`` ->
    ``"(line 3)"``), falling back to ``index`` (position among the
    selectable lines) for any other underscore-prefixed label shape."""
    label = line.get_label()
    if not label.startswith("_"):
        return label
    m = _AUTO_LABEL_RE.match(label)
    return f"(line {m.group(1) if m else index})"


def _legend_kwargs(legend):
    """What is worth carrying over from ``legend`` to its replacement after
    some curves are hidden: location, frame, column count, title. Not
    ``bbox_to_anchor`` -- the stored value is a ``TransformedBbox`` that
    can't be passed back as-is (see :class:`_CurveSelectEngine`'s
    docstring)."""
    kwargs = {"frameon": legend.get_frame_on()}
    loc = getattr(legend, "_loc", None)  # private, but the only place it's kept
    if loc is not None:
        kwargs["loc"] = loc
    ncol = getattr(legend, "_ncols", None) or getattr(legend, "_ncol", None)
    if ncol:
        kwargs["ncol"] = ncol
    title = legend.get_title().get_text()
    if title:
        kwargs["title"] = title
    return kwargs


class _CurveSelectEngine:
    """Show/hide a fixed set of ``lines`` on ``ax``, keeping the legend and
    y-range consistent -- the logic shared by :class:`QtCurveSelector` and
    :class:`IpywidgetsCurveSelector`.

    ``lines`` is frozen at construction (same convention as
    :class:`escape.fit_gui._FitEngine`'s target line / escape's other
    axes-attached tools): the curve list a panel shows is whatever was on
    the axes when it was first opened, not re-scanned on every click.

    Not preserved across a legend rebuild: ``bbox_to_anchor`` (see
    :func:`_legend_kwargs`). ``relim()`` only knows lines/patches/images,
    not collections (scatter, ``fill_between``, escape's own
    :func:`escape.plot_utilities.errortube`) -- with any of those present on
    the axes, re-fitting the y-range to the visible lines alone would be
    wrong, so the range is left untouched in that case rather than guessed at.
    """

    def __init__(self, fig, ax, lines):
        self.fig = fig
        self.ax = ax
        self.lines = list(lines)
        # None: the axes never had a legend -- never create one. Captured
        # once here (refreshed from the live legend on every apply()) rather
        # than always read fresh, because with every curve hidden there is
        # no live legend to read from, and "show everything again" would
        # otherwise have nothing to rebuild it with.
        legend = ax.get_legend()
        self._legend_look = None if legend is None else _legend_kwargs(legend)

    def apply(self, states):
        """Show exactly the curves whose entry in ``states`` (parallel to
        :attr:`lines`) is true; hide the rest, rebuild the legend without
        them (other labelled artists this selector doesn't manage keep
        their entries), and re-fit the y-range to what's left."""
        for line, state in zip(self.lines, states):
            line.set_visible(bool(state))
        shown = [l for l in self.lines if l.get_visible()]

        legend = self.ax.get_legend()
        if legend is not None:
            self._legend_look = _legend_kwargs(legend)  # as the user last left it
            legend.remove()
        if shown and self._legend_look is not None:
            entries = [
                (handle, label)
                for handle, label in zip(*self.ax.get_legend_handles_labels())
                if getattr(handle, "get_visible", lambda: True)()
            ]
            if entries:
                self.ax.legend(*zip(*entries), **self._legend_look)

        if shown and not self.ax.collections:
            self.ax.relim(visible_only=True)
            self.ax.autoscale(enable=True, axis="y")  # a prior zoom/pan switches this off
            self.ax.autoscale_view(scalex=False)

        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Qt front-end
# ---------------------------------------------------------------------------


def _qt_antialiasing(QtGui):
    return getattr(QtGui.QPainter, "Antialiasing", None) or QtGui.QPainter.RenderHint.Antialiasing


def _swatch_icon(QtGui, color):
    """A short line with a dot in ``color`` -- the curve's legend handle, as
    a small icon next to its checkbox."""
    pixmap = QtGui.QPixmap(28, 12)
    pixmap.fill(QtGui.QColor(0, 0, 0, 0))
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(_qt_antialiasing(QtGui))
        qcolor = QtGui.QColor(mcolors.to_hex(color))
        painter.setPen(QtGui.QPen(qcolor, 2))
        painter.drawLine(1, 6, 27, 6)
        painter.setBrush(qcolor)
        painter.drawEllipse(10, 3, 6, 6)
    finally:
        painter.end()
    return QtGui.QIcon(pixmap)


def make_qt_select_dialog(fig, ax, lines):
    """Build, show and return the "Select curves" Qt window for ``lines`` on
    ``ax`` -- a non-modal child of ``fig``'s own window.

    Returned already shown (``_get_or_create_axes_gui``'s factory contract:
    it raises an *existing* panel back to the front itself, but doesn't show
    a freshly-built one). The caller
    (:func:`escape.plot_utilities._run_select_button`) caches the returned
    ``QDialog`` on the axes so a second click re-raises the same window
    (filter text and scroll position included) instead of rebuilding it.
    """
    from qtpy import QtCore, QtGui, QtWidgets

    engine = _CurveSelectEngine(fig, ax, lines)

    dialog = QtWidgets.QDialog(fig.canvas.manager.window)
    dialog.setWindowTitle("Select curves")
    dialog._escape_engine = engine  # kept for introspection/tests; not used internally
    layout = QtWidgets.QVBoxLayout(dialog)

    filter_box = QtWidgets.QLineEdit()
    filter_box.setPlaceholderText("filter, e.g. readback")
    filter_box.setClearButtonEnabled(True)
    layout.addWidget(filter_box)

    holder = QtWidgets.QWidget()
    rows = QtWidgets.QVBoxLayout(holder)
    boxes = []
    for i, line in enumerate(lines):
        box = QtWidgets.QCheckBox(_display_label(line, i))
        box.setIcon(_swatch_icon(QtGui, line.get_color()))
        box.setIconSize(QtCore.QSize(28, 12))
        box.setChecked(line.get_visible())
        rows.addWidget(box)
        boxes.append(box)
    rows.addStretch(1)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(holder)
    layout.addWidget(scroll)

    def guarded(fn):
        # PyQt5 calls qFatal()/abort() when an exception escapes a slot
        # unless sys.excepthook was replaced -- swallow and print instead,
        # escape's usual convention (see attach_fit_button's docstring).
        def run(*_args):
            try:
                fn()
            except Exception as e:
                print(f"[escape] curve selection failed: {e}")

        return run

    def sync():
        engine.apply([box.isChecked() for box in boxes])

    def set_shown(state):
        for box in boxes:
            if not box.isHidden():  # i.e. matches the current filter
                box.blockSignals(True)
                box.setChecked(state)
                box.blockSignals(False)
        sync()

    def refilter():
        needle = filter_box.text().strip().lower()
        for box in boxes:
            box.setVisible(needle in box.text().lower())

    button_row = QtWidgets.QHBoxLayout()
    for text, state in (("All", True), ("None", False)):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(guarded(lambda state=state: set_shown(state)))
        button_row.addWidget(button)
    button_row.addStretch(1)
    layout.addLayout(button_row)

    for box in boxes:
        box.toggled.connect(guarded(sync))
    filter_box.textChanged.connect(guarded(refilter))

    metrics = dialog.fontMetrics()
    advance = getattr(metrics, "horizontalAdvance", None) or metrics.width
    widest = max(advance(box.text()) for box in boxes)
    dialog.resize(min(widest + 130, 1000), min(28 * len(boxes) + 140, 760))

    def _on_fig_close(_event=None):
        try:
            dialog.close()
        except RuntimeError:
            pass  # already gone along with the figure window

    fig.canvas.mpl_connect("close_event", _on_fig_close)

    dialog.show()
    return dialog


# ---------------------------------------------------------------------------
# ipywidgets front-end
# ---------------------------------------------------------------------------


def _swatch_html(color):
    hexcolor = mcolors.to_hex(color)
    return (
        f"<svg width='28' height='14' style='vertical-align:middle'>"
        f"<line x1='1' y1='7' x2='27' y2='7' stroke='{hexcolor}' stroke-width='2'/>"
        f"<circle cx='14' cy='7' r='3' fill='{hexcolor}'/></svg>"
    )


def make_ipywidgets_select_panel(fig, ax, lines):
    """The ipympl-backend analogue of :func:`make_qt_select_dialog`: an
    ipywidgets ``VBox`` displayed below the figure, driving the same
    :class:`_CurveSelectEngine`. Shown (``display()``-ed) on construction,
    same contract as :func:`make_qt_select_dialog`."""
    import ipywidgets as widgets
    from IPython.display import display

    from escape.plot_utilities import _make_resizable

    engine = _CurveSelectEngine(fig, ax, lines)

    filter_box = widgets.Text(placeholder="filter, e.g. readback", layout=widgets.Layout(width="97%"))
    checks = []
    rows = []
    for i, line in enumerate(lines):
        swatch = widgets.HTML(_swatch_html(line.get_color()))
        cb = widgets.Checkbox(
            value=line.get_visible(), description=_display_label(line, i), indent=False,
            layout=widgets.Layout(width="auto"),
        )
        checks.append(cb)
        rows.append(widgets.HBox([swatch, cb]))
    list_box = widgets.VBox(rows, layout=widgets.Layout(max_height="400px", overflow_y="auto"))

    def sync(change=None):
        engine.apply([cb.value for cb in checks])

    def set_shown(state):
        for row, cb in zip(rows, checks):
            if row.layout.display != "none":  # i.e. matches the current filter
                cb.unobserve(sync, names="value")
                cb.value = state
                cb.observe(sync, names="value")
        sync()

    def refilter(change):
        needle = (change["new"] or "").strip().lower()
        for row, cb in zip(rows, checks):
            row.layout.display = "" if needle in cb.description.lower() else "none"

    for cb in checks:
        cb.observe(sync, names="value")
    filter_box.observe(refilter, names="value")

    all_btn = widgets.Button(description="All")
    none_btn = widgets.Button(description="None")
    all_btn.on_click(lambda b: set_shown(True))
    none_btn.on_click(lambda b: set_shown(False))

    panel = widgets.VBox(
        [
            widgets.HTML("<b>Select curves</b>"),
            filter_box,
            list_box,
            widgets.HBox([all_btn, none_btn]),
        ],
        layout=widgets.Layout(border="solid 1px #ccc", padding="6px", width="420px"),
    )
    panel._escape_engine = engine
    _make_resizable(panel)
    display(panel)
    return panel


# ---------------------------------------------------------------------------
# Public, backend-agnostic entry point (parity with fit_gui.AxesFitter /
# freq_gui.FreqAnalyzer / plot_utilities.PeakAnalyzer, for anyone who wants
# the panel without going through the toolbar button)
# ---------------------------------------------------------------------------


def _detect_backend():
    """"qt" or "ipywidgets" for the current environment -- same idea as
    :func:`escape.fit_gui.detect_backend`, duplicated rather than shared
    (each of escape's axes-tool modules keeps its own small copy -- see
    that function's module for why)."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "qt" in backend:
        try:
            import qtpy  # noqa: F401

            return "qt"
        except Exception:
            pass
    try:
        import ipywidgets  # noqa: F401

        from IPython import get_ipython

        if get_ipython() is not None:
            return "ipywidgets"
    except Exception:
        pass
    return None


def CurveSelector(ax=None, *, lines=None, backend="auto"):
    """Attach a "Select curves to show" panel to a matplotlib ``Axes``,
    outside of escape's toolbar-button machinery (see
    :func:`escape.plot_utilities.attach_select_button` for the usual,
    toolbar-driven way to get this).

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to attach to. Defaults to the current axes.
    lines : list of matplotlib.lines.Line2D, optional
        Curves to offer. Defaults to every ``Line2D`` on ``ax`` not marked
        as an escape overlay artist.
    backend : {"auto", "qt", "ipywidgets"}
        "auto" (the default) picks a companion Qt window if a Qt
        matplotlib backend and a Qt binding are both available, otherwise
        an inline ``ipywidgets`` panel (Jupyter only).

    Returns
    -------
    The ``QDialog`` from :func:`make_qt_select_dialog`, or the ``VBox``
    from :func:`make_ipywidgets_select_panel`.
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    if lines is None:
        lines = [l for l in ax.get_lines() if not getattr(l, "_escape_overlay", False)]
    if len(lines) < 2:
        raise ValueError("CurveSelector needs at least two curves to choose between.")
    chosen = backend if backend != "auto" else _detect_backend()
    if chosen == "qt":
        return make_qt_select_dialog(ax.figure, ax, lines)
    if chosen == "ipywidgets":
        return make_ipywidgets_select_panel(ax.figure, ax, lines)
    raise RuntimeError(
        "CurveSelector needs either a Qt matplotlib backend with a Qt binding "
        "installed, or a Jupyter kernel with ipywidgets."
    )
