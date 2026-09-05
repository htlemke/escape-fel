"""Interactive lmfit fitting attached to a matplotlib Axes.

Adds a small fit control panel next to an existing ``Axes`` -- outside the
plot area, so it never covers your data -- with a model-expression field
(any of lmfit's built-in models, by name, e.g. ``"gaussian, linear"`` for a
Gaussian peak on a linear background) and Run / Report / Code controls.
Drag across the axes (the x-span selector) to restrict the fit to a range;
without a drag the whole line's data is used.

Three interchangeable front-ends share one fitting engine:

* :class:`QtAxesFitter` -- a companion Qt window (QComboBox dropdown,
  QLineEdit expression field, tabs for report/code with a real clipboard
  Copy button). Used when a Qt matplotlib backend is active and a Qt
  binding is importable.
* :class:`IpywidgetsAxesFitter` -- an ``ipywidgets.VBox`` panel (Dropdown +
  Text + Output) displayed below the figure, Jupyter's native widget UI.
  Used when running inside an IPython/Jupyter kernel with ipywidgets
  installed.
* :class:`MplAxesFitter` -- built from ``matplotlib.widgets`` alone (a
  ``TextBox`` for the expression, since matplotlib has no dropdown/combobox
  widget -- only ``RadioButtons``, which lists every option at once and
  doesn't compose well with a long model catalog). Works under any
  interactive matplotlib backend without extra dependencies; this is the
  fallback when neither of the above applies.

:func:`AxesFitter` picks the right one automatically (``backend="auto"``,
the default); pass ``backend="qt"|"ipywidgets"|"mpl"`` to force a choice, or
instantiate one of the three classes directly.

Quick start::

    import numpy as np, matplotlib.pyplot as plt
    from escape.fit_gui import AxesFitter

    x = np.linspace(0, 10, 300)
    y = 3 * np.exp(-((x - 5) ** 2) / (2 * 0.5**2)) + 0.1 * x + np.random.normal(scale=0.05, size=x.size)

    fig, ax = plt.subplots()
    ax.plot(x, y, ".", label="data")
    fitter = AxesFitter(ax)  # picks Qt / ipywidgets / matplotlib automatically

The fit result lives at ``fitter.engine.result`` (an ``lmfit.model.ModelResult``)
after a run; ``fitter.engine.get_code()`` returns a standalone Python script
reproducing that fit, for pasting into a notebook and tuning further by
hand -- the GUI is meant to get you most of the way to working lmfit code,
not to replace it.
"""

from __future__ import annotations

import html
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector, TextBox
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._axes_selection import snapshot_data_lines

try:
    import lmfit
    import lmfit.models as _lmfit_models
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "escape.fit_gui requires the optional 'lmfit' dependency: "
        "pip install lmfit  (or: pip install 'escape-fel[fit]')"
    ) from exc


# ---------------------------------------------------------------------------
# Model registry / composite-model helpers (backend-agnostic)
# ---------------------------------------------------------------------------


def _build_model_registry():
    """Map short lowercase names (e.g. ``"gaussian"``) to ``(ModelClass, extra_kwargs)``.

    Built from every ``*Model`` class in :mod:`lmfit.models` that can be
    instantiated with just a ``prefix`` -- this excludes the handful (like
    ``ExpressionModel``, ``SplineModel``) that need extra required arguments
    and so aren't a fit for a name-only expression.

    ``StepModel``/``RectangleModel`` additionally take a ``form=`` argument
    selecting the functional shape ('linear' (their default), 'atan', 'erf',
    'logistic') -- e.g. ``form='erf'`` is a scaled, centered error function.
    That's a constructor argument, not something a bare class name exposes,
    so each shape also gets its own registry entry (``"step_erf"``, etc.),
    plus a plain ``"erf"`` alias for the single most commonly wanted one.
    """
    registry = {}
    for name in dir(_lmfit_models):
        obj = getattr(_lmfit_models, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, lmfit.Model)
            and name.endswith("Model")
            and obj is not lmfit.Model
        ):
            try:
                obj(prefix="t_")
            except Exception:
                continue
            registry[name[: -len("Model")].lower()] = (obj, {})

    for base_key, cls in (("step", _lmfit_models.StepModel), ("rectangle", _lmfit_models.RectangleModel)):
        for form in ("erf", "atan", "logistic", "linear"):
            registry[f"{base_key}_{form}"] = (cls, {"form": form})
    registry["erf"] = registry["step_erf"]
    # "constant" is lmfit's name (a flat y=c offset, no slope) -- add the more
    # intuitive "offset" as an alias, since that's the plain-language version
    # of what people usually mean ("a background with no slope").
    registry["offset"] = registry["constant"]
    return registry


MODEL_REGISTRY = _build_model_registry()


def _kwargs_repr(kwargs):
    return "".join(f", {k}={v!r}" for k, v in kwargs.items())


def _param_set_repr(p):
    """``value=...[, min=...][, max=...][, vary=False]`` for an lmfit Parameter,
    omitting bounds that are at their +/-inf default and ``vary`` when it's
    the default ``True`` -- keeps generated code close to what someone would
    type by hand for an unconstrained parameter."""
    parts = [f"value={p.value!r}"]
    if np.isfinite(p.min):
        parts.append(f"min={p.min!r}")
    if np.isfinite(p.max):
        parts.append(f"max={p.max!r}")
    if not p.vary:
        parts.append("vary=False")
    return ", ".join(parts)


def _pre_scroll_html(text, max_height="320px"):
    """Monospace text in a box that scrolls (both axes) instead of wrapping --
    for the ipywidgets Report/Code panes, which would otherwise wrap long
    lines and force the whole panel wider than it should be."""
    return (
        f"<div style='max-height:{max_height}; overflow:auto; border:1px solid #ccc; padding:4px;'>"
        f"<pre style='white-space:pre; margin:0; font-family:monospace; font-size:12px;'>"
        f"{html.escape(text)}</pre></div>"
    )


def parse_model_expression(expr, registry=MODEL_REGISTRY):
    """Parse a comma/plus-separated model expression into a composite Model.

    ``"gaussian, gaussian, linear"`` (or ``"gaussian + gaussian + linear"``)
    builds ``GaussianModel(prefix='gaussian1_') + GaussianModel(prefix='gaussian2_')
    + LinearModel(prefix='linear1_')``. Repeated names get numbered prefixes
    so their parameters don't collide. A scaled error function on top of a
    constant offset -- e.g. an edge/step response -- is
    ``"erf, constant"`` (equivalently ``"step_erf, constant"``).

    Returns
    -------
    model : lmfit.Model
    components : list of (prefix, name, ModelClass, instance, extra_kwargs)
    """
    tokens = [t.strip().lower() for t in re.split(r"[,+]", expr) if t.strip()]
    if not tokens:
        raise ValueError("Empty model expression.")
    counts = {}
    components = []
    model = None
    for tok in tokens:
        if tok not in registry:
            raise ValueError(
                f"Unknown model {tok!r}. Available: {', '.join(sorted(registry))}"
            )
        counts[tok] = counts.get(tok, 0) + 1
        prefix = f"{tok}{counts[tok]}_"
        cls, extra = registry[tok]
        comp = cls(prefix=prefix, **extra)
        components.append((prefix, tok, cls, comp, extra))
        model = comp if model is None else model + comp
    return model, components


def guess_composite_params(model, components, x, y):
    """Initial parameter guess for a composite model.

    Background-like components (no ``center`` parameter) are guessed against
    the full data. Peak-like components (have a ``center`` parameter) are
    each guessed against an equal x-slice of the data, so multiple peaks of
    the same type don't all land on top of each other -- a cheap way to
    spread out the starting guesses without any user interaction.
    """
    params = model.make_params()
    peak_comps = [c for c in components if "center" in c[3].param_names]
    bg_comps = [c for c in components if c not in peak_comps]
    for _, _, _, comp, _ in bg_comps:
        try:
            params.update(comp.guess(y, x=x))
        except Exception:
            pass
    if peak_comps:
        edges = np.linspace(x.min(), x.max(), len(peak_comps) + 1)
        for i, (_, _, _, comp, _) in enumerate(peak_comps):
            mask = (x >= edges[i]) & (x <= edges[i + 1])
            if mask.sum() < 2:
                mask = np.ones_like(x, dtype=bool)
            try:
                params.update(comp.guess(y[mask], x=x[mask]))
            except Exception:
                pass
    return params


def _fmt_array(a):
    with np.printoptions(threshold=np.inf, precision=8, floatmode="maxprec"):
        return "np.array(" + np.array2string(np.asarray(a), separator=", ") + ")"


# ---------------------------------------------------------------------------
# Shared, UI-independent fitting engine
# ---------------------------------------------------------------------------


class _FitEngine:
    """Holds fit state and does the actual work; no widgets of its own.

    All three UI front-ends wrap one of these and call into it. It also owns
    picking the target line (via a shared, axes-cached snapshot -- see
    :func:`escape._axes_selection.snapshot_data_lines` -- taken before any
    widget creates its own artists on the axes, since an interactive
    ``SpanSelector`` adds ``Line2D`` edge-handle artists that a fresh
    ``ax.get_lines()`` call could otherwise pick up as if they were data)
    and drawing the fit overlay onto ``ax``, which is common to every backend.
    """

    def __init__(self, ax, line=None, model_expr="gaussian, linear"):
        self.ax = ax
        self.fig = ax.figure
        self.model_expr = model_expr
        self.model = None
        self.params = None
        self.result = None
        self.components = None
        self.range = None
        self._fit_artists = []
        self._preview_artists = []
        self._x_fit = None
        self._y_fit = None

        data_lines = snapshot_data_lines(ax)
        self.line = line if line is not None else (data_lines[-1] if data_lines else None)
        self._pickable_lines = data_lines
        self._pick_cid = self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def enable_picking(self):
        for ln in self._pickable_lines:
            ln.set_picker(True)
            ln.set_pickradius(5)

    def _on_pick(self, event):
        if event.artist in self._pickable_lines:
            self.line = event.artist
            print(f"[fit_gui] target line set to {self.line.get_label()!r}")

    def set_range(self, xmin, xmax):
        if xmax > xmin:
            self.range = (xmin, xmax)
            print(f"[fit_gui] fit range set to [{xmin:.6g}, {xmax:.6g}]")

    def _target_line(self):
        if self.line is None:
            raise RuntimeError(
                "No line to fit -- pass line=..., or click a data line on the "
                "axes to select it."
            )
        return self.line

    def initial_xrange(self):
        """(xmin, xmax) of the target line's data, or (0.0, 1.0) if there is none yet."""
        if self.line is None:
            return 0.0, 1.0
        x = np.asarray(self.line.get_xdata(), dtype=float)
        return float(x.min()), float(x.max())

    def xy_selection(self):
        line = self._target_line()
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if self.range is not None:
            xmin, xmax = self.range
            mask = (x >= xmin) & (x <= xmax)
            x, y = x[mask], y[mask]
        return x, y

    def build_model(self, model_expr=None):
        """Parse the expression and (re)guess starting parameters.

        Stores the composite model + a fresh :class:`lmfit.Parameters` on
        ``self.model``/``self.params`` -- the basis for :meth:`preview`,
        :meth:`fit_with_current_params` (which a parameter-table UI can edit
        in between), and code generation. Always re-guesses, discarding any
        edits made to the previous ``self.params`` -- call this only when the
        expression changes, not on every preview/fit.
        """
        if model_expr is not None:
            self.model_expr = model_expr
        x, y = self.xy_selection()
        self._x_fit, self._y_fit = x, y
        model, components = parse_model_expression(self.model_expr)
        self.model = model
        self.components = components
        self.params = guess_composite_params(model, components, x, y) if x.size >= 2 else model.make_params()
        return self.params

    def preview(self):
        """Evaluate the current model at ``self.params`` (no fitting) and
        draw it as a dashed "initial guess" curve, so parameter edits can be
        checked against the data before committing to a fit."""
        if self.model is None:
            self.build_model()
        x = self._x_fit
        if x is None or x.size < 2:
            print("[fit_gui] not enough points in the selected range to preview.")
            return
        xf = np.linspace(x.min(), x.max(), max(500, x.size * 2))
        yf = self.model.eval(self.params, x=xf)
        for artist in self._preview_artists:
            artist.remove()
        (line,) = self.ax.plot(xf, yf, ":", color="tab:blue", lw=1.5, label="initial guess")
        self._preview_artists = [line]
        self.ax.legend(fontsize="small")
        self.fig.canvas.draw_idle()

    def run_fit(self, model_expr=None):
        """Rebuild the model fresh (re-guessing), then fit -- the one-shot
        path used when there's no parameter table to edit in between."""
        self.build_model(model_expr)
        return self._fit()

    def fit_with_current_params(self):
        """Fit using whatever is currently in ``self.params`` as-is (values,
        bounds, vary flags) -- for a UI that lets the user edit them first
        via :meth:`build_model`/a parameter table, instead of always
        re-guessing from scratch."""
        if self.model is None:
            self.build_model()
        return self._fit()

    def _fit(self):
        x, y = self._x_fit, self._y_fit
        if x is None or x.size < 2:
            print("[fit_gui] not enough points in the selected range to fit.")
            return None
        result = self.model.fit(y, self.params, x=x)
        self.result = result
        # carry the fitted values back into self.params (keeping whatever
        # bounds/vary the user set) so a follow-up preview/fit/get_code
        # reflects the latest best estimate rather than the pre-fit guess.
        for name, p in result.params.items():
            if name in self.params:
                self.params[name].value = p.value
        self._draw_result(x, result)
        print(result.fit_report())
        return result

    def _draw_result(self, x, result):
        for artist in self._preview_artists:
            artist.remove()
        self._preview_artists = []
        for artist in self._fit_artists:
            artist.remove()
        self._fit_artists = []

        xf = np.linspace(x.min(), x.max(), max(500, x.size * 2))
        yf = result.eval(x=xf)
        (fit_line,) = self.ax.plot(
            xf, yf, "-", color="crimson", lw=1.5,
            label=f"fit: {self.model_expr} (red. $\\chi^2$={result.redchi:.3g})",
        )
        self._fit_artists.append(fit_line)

        comps = result.eval_components(x=xf)
        if len(comps) > 1:
            for prefix, comp_y in comps.items():
                comp_y = np.broadcast_to(comp_y, xf.shape)
                (cl,) = self.ax.plot(
                    xf, comp_y, "--", lw=1, alpha=0.7, label=f"{prefix.rstrip('_')}"
                )
                self._fit_artists.append(cl)

        self.ax.legend(fontsize="small")
        self.fig.canvas.draw_idle()

    def fit_report_text(self):
        if self.result is None:
            return "(no fit run yet)"
        return self.result.fit_report()

    def get_code(self):
        """Return a standalone Python script reproducing the current fit.

        Parameter values/bounds/vary flags are emitted explicitly from
        ``self.params`` (its values were carried forward from the fit
        result, so it reflects the actual fitted starting point, plus
        whatever bounds/fixed flags were set) rather than regenerated via
        ``.guess()`` -- that way any limits/fixed parameters set through a
        parameter-table UI show up in the generated code too.
        """
        if self.result is None:
            raise RuntimeError("Run a fit first.")
        x, y = self._x_fit, self._y_fit
        class_names = sorted({cls.__name__ for _, _, cls, _, _ in self.components})

        lines = []
        lines.append("import numpy as np")
        lines.append("from lmfit.models import " + ", ".join(class_names))
        lines.append("")
        lines.append(f"x = {_fmt_array(x)}")
        lines.append(f"y = {_fmt_array(y)}")
        lines.append("")
        terms = [
            f"{cls.__name__}(prefix={prefix!r}{_kwargs_repr(extra)})"
            for prefix, _, cls, _, extra in self.components
        ]
        lines.append("model = " + " + ".join(terms))
        lines.append("params = model.make_params()")
        for name, p in self.params.items():
            if p.expr:  # derived (e.g. fwhm/height computed from sigma) -- not a free param
                continue
            lines.append(f"params[{name!r}].set({_param_set_repr(p)})")
        lines.append("")
        lines.append("result = model.fit(y, params, x=x)")
        lines.append("print(result.fit_report())")
        lines.append("")
        lines.append("import matplotlib.pyplot as plt")
        lines.append("fig, ax = plt.subplots()")
        lines.append("ax.plot(x, y, '.', label='data')")
        lines.append("xf = np.linspace(x.min(), x.max(), 500)")
        lines.append("ax.plot(xf, result.eval(x=xf), '-', label='fit')")
        lines.append("ax.legend()")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _qt_available():
    try:
        from qtpy import QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def _ipywidgets_available():
    try:
        import ipywidgets  # noqa: F401
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"


def detect_backend():
    """Pick "qt", "ipywidgets", or "mpl" for the current environment."""
    backend = matplotlib.get_backend().lower()
    if "qt" in backend and _qt_available():
        return "qt"
    if _ipywidgets_available():
        return "ipywidgets"
    return "mpl"


# ---------------------------------------------------------------------------
# matplotlib.widgets front-end (no extra dependencies)
# ---------------------------------------------------------------------------


class MplAxesFitter:
    """Fit panel built from ``matplotlib.widgets``, placed outside ``ax``.

    Uses ``mpl_toolkits.axes_grid1.make_axes_locatable`` to append a control
    strip alongside ``ax`` (the same mechanism matplotlib uses for
    colorbars), so the panel never overlaps your data -- at the cost of
    shrinking ``ax`` slightly to make room, same trade-off as a colorbar.

    matplotlib has no dropdown/combobox widget (only ``RadioButtons``, which
    renders every option at once), so the model is entered as free text
    here; :class:`IpywidgetsAxesFitter` and :class:`QtAxesFitter` offer a
    real dropdown for adding components.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to attach to. Defaults to the current axes.
    line : matplotlib.lines.Line2D, optional
        Line to fit. Defaults to the most recently added line on ``ax``; can
        also be picked by clicking directly on a line.
    model_expr : str
        Initial model expression, e.g. ``"gaussian, linear"``. Any
        comma/plus-separated list of names from :data:`MODEL_REGISTRY`.
    side : {"right", "left"}
        Which side of ``ax`` the panel is appended to.
    """

    def __init__(self, ax=None, line=None, model_expr="gaussian, linear", side="right"):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.ax = ax or plt.gca()
        self.fig = self.ax.figure
        self.engine = _FitEngine(self.ax, line=line, model_expr=model_expr)
        self.engine.enable_picking()

        divider = make_axes_locatable(self.ax)
        self._panel = divider.append_axes(side, size="38%", pad=0.45)
        self._panel.axis("off")

        rows = np.linspace(0.95, 0.08, 6)
        col = dict(x=0.04, w=0.92)

        self._panel.text(0.5, 1.0, "lmfit", ha="center", va="top", fontsize=9, fontweight="bold")

        self._ax_text = self._panel.inset_axes([col["x"], rows[1], col["w"], 0.09])
        self.textbox = TextBox(self._ax_text, "", initial=model_expr)
        self.textbox.on_submit(self._on_model_text)

        self._ax_range = self._panel.inset_axes([col["x"], rows[2], col["w"], 0.09])
        self.button_range = Button(self._ax_range, "Select range")
        self.button_range.on_clicked(self._toggle_span)

        self._ax_run = self._panel.inset_axes([col["x"], rows[3], col["w"], 0.09])
        self.button_run = Button(self._ax_run, "Run fit")
        self.button_run.on_clicked(lambda evt: self.run_fit())

        self._ax_report = self._panel.inset_axes([col["x"], rows[4], col["w"], 0.09])
        self.button_report = Button(self._ax_report, "Report")
        self.button_report.on_clicked(lambda evt: self.show_report())

        self._ax_code = self._panel.inset_axes([col["x"], rows[5], col["w"], 0.09])
        self.button_code = Button(self._ax_code, "Code")
        self.button_code.on_clicked(lambda evt: self.show_code())

        self.span = SpanSelector(
            self.ax,
            lambda xmin, xmax: self.engine.set_range(xmin, xmax),
            direction="horizontal",
            useblit=False,
            interactive=True,
            props=dict(alpha=0.15, facecolor="tab:orange"),
        )
        self.span.set_active(False)

    def _toggle_span(self, event):
        active = not self.span.active
        self.span.set_active(active)
        self.button_range.label.set_text("Range: on" if active else "Select range")
        self.fig.canvas.draw_idle()

    def _on_model_text(self, text):
        self.engine.model_expr = text

    def run_fit(self):
        return self.engine.run_fit()

    def show_report(self):
        fig = plt.figure(figsize=(7, 6))
        fig.text(0.02, 0.98, self.engine.fit_report_text(), family="monospace", fontsize=8, va="top", ha="left")
        plt.axis("off")
        fig.show()
        return fig

    def show_code(self):
        code = self.engine.get_code()
        print(code)
        fig = plt.figure(figsize=(8, 6))
        fig.text(0.02, 0.98, code, family="monospace", fontsize=8, va="top", ha="left")
        plt.axis("off")
        fig.show()
        return fig


# ---------------------------------------------------------------------------
# ipywidgets front-end
# ---------------------------------------------------------------------------


def _make_ipywidgets_fitter_class():
    import ipywidgets as widgets
    from IPython.display import display

    class IpywidgetsAxesFitter(widgets.VBox):
        """Fit panel built from ``ipywidgets``, displayed below the figure.

        A real dropdown lists every model in :data:`MODEL_REGISTRY`; "+ Add"
        appends the selected one to the (still freely editable) expression
        field, so composing e.g. ``"gaussian, gaussian, linear"`` doesn't
        require typing model names by hand.

        A "Parameters" tab lists each free parameter with editable value /
        min / max / vary(fix) fields, live-synced into the fit engine as you
        type -- "Preview" evaluates the model at those values (no fit) and
        draws it dashed, so a starting guess can be checked/tuned against
        the data before running the actual fit; unchecking "vary" fixes a
        parameter at its current value. "Run fit" uses whatever's currently
        in the table as the fit's starting point/constraints rather than
        re-guessing -- unless the model expression changed since the table
        was last built, in which case it (re-)guesses fresh first, so a
        single click with no table editing still behaves as a one-shot fit.

        Parameters
        ----------
        detach : bool
            If ``True``, show the panel in a JupyterLab Sidecar tab split to
            the right of the notebook instead of inline below the figure
            (same mechanism as ``detached=True`` elsewhere in
            :mod:`escape.plot_utilities`). Needs the ``sidecar`` package and
            its JupyterLab extension -- and, once detached, the Sidecar tab
            itself has a draggable splitter, so it's user-resizable there.
        title : str, optional
            Sidecar panel title when ``detach=True``. Defaults to "lmfit".
        width : str
            CSS width of the panel when shown inline (``detach=False``).
            Default ``"600px"``; widen this if the model expression or
            parameter names are long enough to feel cramped.
        """

        def __init__(
            self, ax=None, line=None, model_expr="gaussian, linear", detach=False, title=None, width="600px"
        ):
            self.ax = ax or plt.gca()
            self.engine = _FitEngine(self.ax, line=line, model_expr=model_expr)
            self.engine.enable_picking()
            self._built_expr = None  # expression self.engine.params was last built for

            self._component_dd = widgets.Dropdown(
                options=sorted(MODEL_REGISTRY), value="gaussian", description="component:",
                layout=widgets.Layout(width="220px"),
            )
            self._add_btn = widgets.Button(description="+ Add", layout=widgets.Layout(width="70px"))
            self._clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="70px"))
            self._add_btn.on_click(self._on_add)
            self._clear_btn.on_click(self._on_clear)

            self._expr_text = widgets.Text(
                value=model_expr, description="model:", layout=widgets.Layout(width="97%"),
                continuous_update=False,  # only fire on Enter/blur, not every keystroke
            )
            self._expr_text.observe(self._on_expr_change, names="value")

            xmin0, xmax0 = self.engine.initial_xrange()
            self._xmin_box = widgets.FloatText(value=xmin0, description="xmin", layout=widgets.Layout(width="150px"))
            self._xmax_box = widgets.FloatText(value=xmax0, description="xmax", layout=widgets.Layout(width="150px"))
            self._xmin_box.observe(self._on_range_box, names="value")
            self._xmax_box.observe(self._on_range_box, names="value")

            self._range_toggle = widgets.ToggleButton(
                value=False, description="Select range on plot", layout=widgets.Layout(width="180px")
            )
            self._range_toggle.observe(self._on_range_toggle, names="value")

            self._guess_btn = widgets.Button(description="Guess params")
            self._guess_btn.on_click(lambda b: self._on_guess())
            self._preview_btn = widgets.Button(description="Preview")
            self._preview_btn.on_click(lambda b: self._on_preview())
            self._run_btn = widgets.Button(description="Run fit", button_style="success")
            self._run_btn.on_click(lambda b: self.run_fit())

            self._param_box = widgets.VBox(layout=widgets.Layout(max_height="320px", overflow="auto"))
            self._report_html = widgets.HTML(_pre_scroll_html("(no fit run yet)"))
            self._code_html = widgets.HTML(_pre_scroll_html(""))
            tabs = widgets.Tab(children=[self._param_box, self._report_html, self._code_html])
            tabs.set_title(0, "Parameters")
            tabs.set_title(1, "Report")
            tabs.set_title(2, "Code")

            self.span = SpanSelector(
                self.ax,
                self._on_span,
                direction="horizontal",
                useblit=False,
                interactive=True,
                props=dict(alpha=0.15, facecolor="tab:orange"),
            )
            self.span.set_active(False)

            super().__init__(
                [
                    widgets.HTML("<b>lmfit</b>"),
                    widgets.HBox([self._component_dd, self._add_btn, self._clear_btn]),
                    self._expr_text,
                    widgets.HBox([self._xmin_box, self._xmax_box, self._range_toggle]),
                    widgets.HBox([self._guess_btn, self._preview_btn, self._run_btn]),
                    tabs,
                ],
                layout=widgets.Layout(border="solid 1px #ccc", padding="6px", width=width),
            )

            from escape.plot_utilities import _close_sidecar, _open_sidecar, _suppress_inline_redisplay

            key = f"fit_gui-{id(self.ax)}"
            _close_sidecar(key)
            if detach:
                _open_sidecar(key, title or "lmfit", lambda: display(self))
                _suppress_inline_redisplay(self)
            else:
                display(self)

        def _on_add(self, b):
            name = self._component_dd.value
            self._expr_text.value = (
                name if not self._expr_text.value.strip() else self._expr_text.value.rstrip(", ") + ", " + name
            )

        def _on_clear(self, b):
            self._expr_text.value = ""

        def _on_expr_change(self, change):
            self.engine.model_expr = change["new"]
            self._try_rebuild()

        def _try_rebuild(self):
            """(Re)build+guess and refresh the parameter table, tolerating an
            incomplete/invalid expression (e.g. mid-edit) by leaving the old
            table in place instead of raising into a widget callback."""
            try:
                self._on_guess()
            except ValueError as e:
                print(f"[fit_gui] {e}")

        def _on_range_box(self, change):
            self.engine.set_range(self._xmin_box.value, self._xmax_box.value)

        def _on_range_toggle(self, change):
            self.span.set_active(change["new"])

        def _on_span(self, xmin, xmax):
            self.engine.set_range(xmin, xmax)
            self._xmin_box.value, self._xmax_box.value = xmin, xmax

        def _ensure_built(self):
            """(Re)build+guess only if there's no model yet or the expression
            changed since the last build -- otherwise keep whatever's in the
            table (including the user's edits)."""
            if self.engine.model is None or self._built_expr != self._expr_text.value:
                self._on_guess()

        def _on_guess(self):
            self.engine.build_model(self._expr_text.value)
            self._built_expr = self._expr_text.value
            self._rebuild_param_table()

        def _on_preview(self):
            self._ensure_built()
            self.engine.preview()

        def _rebuild_param_table(self):
            params = self.engine.params
            if params is None:
                self._param_box.children = []
                return
            name_col_width = "190px"
            header = widgets.HBox(
                [
                    widgets.Label("parameter", layout=widgets.Layout(width=name_col_width)),
                    widgets.Label("value", layout=widgets.Layout(width="110px")),
                    widgets.Label("min", layout=widgets.Layout(width="80px")),
                    widgets.Label("max", layout=widgets.Layout(width="80px")),
                    widgets.Label("vary", layout=widgets.Layout(width="50px")),
                ]
            )
            rows = [header]
            for name, p in params.items():
                if p.expr:  # derived (e.g. fwhm/height) -- not directly editable
                    continue
                # a Label truncates instead of wrapping; an HTML <div> with
                # word-break lets long names (e.g. "dampedharmonicoscillator1_amplitude")
                # wrap onto a second line within the column instead of being cut off.
                name_cell = widgets.HTML(
                    f"<div style='word-break:break-all; font-family:monospace; font-size:12px;'>{html.escape(name)}</div>",
                    layout=widgets.Layout(width=name_col_width),
                )
                value_box = widgets.FloatText(value=p.value, layout=widgets.Layout(width="110px"))
                min_box = widgets.Text(
                    value="" if not np.isfinite(p.min) else repr(p.min),
                    placeholder="-inf", layout=widgets.Layout(width="80px"),
                )
                max_box = widgets.Text(
                    value="" if not np.isfinite(p.max) else repr(p.max),
                    placeholder="inf", layout=widgets.Layout(width="80px"),
                )
                vary_cb = widgets.Checkbox(value=p.vary, indent=False, layout=widgets.Layout(width="50px"))
                value_box.observe(lambda ch, n=name: self._on_param_edit(n, "value", ch["new"]), names="value")
                min_box.observe(lambda ch, n=name: self._on_param_edit(n, "min", ch["new"]), names="value")
                max_box.observe(lambda ch, n=name: self._on_param_edit(n, "max", ch["new"]), names="value")
                vary_cb.observe(lambda ch, n=name: self._on_param_edit(n, "vary", ch["new"]), names="value")
                rows.append(
                    widgets.HBox(
                        [name_cell, value_box, min_box, max_box, vary_cb],
                        layout=widgets.Layout(align_items="center"),
                    )
                )
            self._param_box.children = rows

        def _on_param_edit(self, name, field, value):
            params = self.engine.params
            if params is None or name not in params:
                return
            if field == "vary":
                params[name].set(vary=value)
            elif field == "value":
                params[name].set(value=value)
            else:
                try:
                    v = float(value) if str(value).strip() else (-np.inf if field == "min" else np.inf)
                except ValueError:
                    return
                params[name].set(**{field: v})

        def run_fit(self):
            self._ensure_built()
            result = self.engine.fit_with_current_params()
            self._rebuild_param_table()
            if result is not None:
                self._report_html.value = _pre_scroll_html(self.engine.fit_report_text())
                self._code_html.value = _pre_scroll_html(self.engine.get_code())
            return result

    return IpywidgetsAxesFitter


# ---------------------------------------------------------------------------
# Qt front-end
# ---------------------------------------------------------------------------


def _make_qt_axes_fitter_class():
    from qtpy import QtCore, QtGui, QtWidgets

    class QtAxesFitter(QtWidgets.QWidget):
        """Fit panel as a companion Qt window next to the figure.

        A real ``QComboBox`` dropdown lists every model in
        :data:`MODEL_REGISTRY`; "+ Add" appends it to the (still freely
        editable) expression field. Report/Code are read-only, non-wrapping
        text panes (so long lines scroll instead of wrapping) with a
        Copy-to-clipboard button (via ``QApplication.clipboard``).

        A "Parameters" tab is a table (parameter / value / min / max / vary)
        editable in place -- edits sync live into the fit engine. "Preview"
        evaluates the model at the table's current values (no fit) and draws
        it dashed, so a starting guess can be checked/tuned against the data
        first; unchecking "vary" fixes a parameter at its current value.
        "Run fit" fits from whatever's currently in the table -- unless the
        model expression changed since it was last built, in which case it
        (re-)guesses fresh first, so a single click with no table editing
        still behaves as a one-shot fit.

        Parameters
        ----------
        detach : bool
            If ``True``, position this window immediately to the right of
            ``ax``'s figure window (the ipywidgets-backend analogue of
            Sidecar's "split-right"). If ``False`` (the default), the window
            manager places it as it would any new top-level window.
        """

        def __init__(self, ax=None, line=None, model_expr="gaussian, linear", detach=False):
            self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            super().__init__()
            self.ax = ax or plt.gca()
            self.engine = _FitEngine(self.ax, line=line, model_expr=model_expr)
            self.engine.enable_picking()
            self._built_expr = None  # expression self.engine.params was last built for
            self._param_row_names = []

            self.setWindowTitle("lmfit fit panel")
            layout = QtWidgets.QVBoxLayout(self)

            add_row = QtWidgets.QHBoxLayout()
            self._component_cb = QtWidgets.QComboBox()
            self._component_cb.addItems(sorted(MODEL_REGISTRY))
            self._component_cb.setCurrentText("gaussian")
            add_btn = QtWidgets.QPushButton("+ Add")
            clear_btn = QtWidgets.QPushButton("Clear")
            add_btn.clicked.connect(self._on_add)
            clear_btn.clicked.connect(self._on_clear)
            add_row.addWidget(self._component_cb)
            add_row.addWidget(add_btn)
            add_row.addWidget(clear_btn)
            layout.addLayout(add_row)

            self._expr_edit = QtWidgets.QLineEdit(model_expr)
            self._expr_edit.textChanged.connect(self._on_expr_change)
            self._expr_edit.editingFinished.connect(self._try_rebuild)  # Enter/blur, not every keystroke
            layout.addWidget(self._expr_edit)

            range_row = QtWidgets.QHBoxLayout()
            xmin0, xmax0 = self.engine.initial_xrange()
            self._xmin_spin = QtWidgets.QDoubleSpinBox()
            self._xmax_spin = QtWidgets.QDoubleSpinBox()
            for spin, val in ((self._xmin_spin, xmin0), (self._xmax_spin, xmax0)):
                spin.setRange(-1e30, 1e30)
                spin.setDecimals(6)
                spin.setValue(val)
            self._xmin_spin.valueChanged.connect(self._on_range_spin)
            self._xmax_spin.valueChanged.connect(self._on_range_spin)
            self._range_btn = QtWidgets.QPushButton("Select range on plot")
            self._range_btn.setCheckable(True)
            self._range_btn.toggled.connect(self._on_range_toggle)
            range_row.addWidget(QtWidgets.QLabel("xmin"))
            range_row.addWidget(self._xmin_spin)
            range_row.addWidget(QtWidgets.QLabel("xmax"))
            range_row.addWidget(self._xmax_spin)
            range_row.addWidget(self._range_btn)
            layout.addLayout(range_row)

            run_row = QtWidgets.QHBoxLayout()
            guess_btn = QtWidgets.QPushButton("Guess params")
            guess_btn.clicked.connect(self._on_guess)
            preview_btn = QtWidgets.QPushButton("Preview")
            preview_btn.clicked.connect(self._on_preview)
            run_btn = QtWidgets.QPushButton("Run fit")
            run_btn.clicked.connect(self.run_fit)
            run_row.addWidget(guess_btn)
            run_row.addWidget(preview_btn)
            run_row.addWidget(run_btn)
            layout.addLayout(run_row)

            tabs = QtWidgets.QTabWidget()

            self._param_table = QtWidgets.QTableWidget(0, 5)
            self._param_table.setHorizontalHeaderLabels(["parameter", "value", "min", "max", "vary"])
            self._param_table.horizontalHeader().setStretchLastSection(False)
            self._param_table.verticalHeader().setVisible(False)
            self._param_table.setWordWrap(True)  # wrap long names instead of truncating (see _rebuild_param_table)
            self._param_table.cellChanged.connect(self._on_param_cell_changed)
            tabs.addTab(self._param_table, "Parameters")

            self._report_text = QtWidgets.QPlainTextEdit(readOnly=True)
            self._code_text = QtWidgets.QPlainTextEdit(readOnly=True)
            mono_font = QtGui.QFont("Monospace", 9)
            mono_font.setStyleHint(QtGui.QFont.TypeWriter)
            for w in (self._report_text, self._code_text):
                w.setFont(mono_font)
                w.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)  # scroll long lines, don't wrap
            tabs.addTab(self._report_text, "Report")
            tabs.addTab(self._code_text, "Code")
            layout.addWidget(tabs)

            copy_btn = QtWidgets.QPushButton("Copy code to clipboard")
            copy_btn.clicked.connect(self._copy_code)
            layout.addWidget(copy_btn)

            self.span = SpanSelector(
                self.ax,
                self._on_span,
                direction="horizontal",
                useblit=False,
                interactive=True,
                props=dict(alpha=0.15, facecolor="tab:orange"),
            )
            self.span.set_active(False)

            self.resize(480, 560)
            if detach:
                self._move_beside_figure()
            self.show()

        def _move_beside_figure(self):
            try:
                fig_window = self.ax.figure.canvas.window()
            except Exception:
                fig_window = None
            if fig_window is not None:
                g = fig_window.frameGeometry()
                self.move(g.x() + g.width(), g.y())

        def _on_add(self):
            name = self._component_cb.currentText()
            cur = self._expr_edit.text().rstrip(", ")
            self._expr_edit.setText(name if not cur else cur + ", " + name)
            self._try_rebuild()

        def _on_clear(self):
            self._expr_edit.setText("")
            self._try_rebuild()

        def _on_expr_change(self, text):
            self.engine.model_expr = text

        def _try_rebuild(self):
            """(Re)build+guess and refresh the parameter table, tolerating an
            incomplete/invalid expression (e.g. mid-edit) by leaving the old
            table in place instead of raising into a Qt signal handler."""
            try:
                self._on_guess()
            except ValueError as e:
                print(f"[fit_gui] {e}")

        def _on_range_spin(self, _val):
            self.engine.set_range(self._xmin_spin.value(), self._xmax_spin.value())

        def _on_range_toggle(self, checked):
            self.span.set_active(checked)
            self._range_btn.setText("Range: on" if checked else "Select range on plot")

        def _on_span(self, xmin, xmax):
            self.engine.set_range(xmin, xmax)
            self._xmin_spin.setValue(xmin)
            self._xmax_spin.setValue(xmax)

        def _ensure_built(self):
            """(Re)build+guess only if there's no model yet or the expression
            changed since the last build -- otherwise keep whatever's in the
            table (including the user's edits)."""
            if self.engine.model is None or self._built_expr != self._expr_edit.text():
                self._on_guess()

        def _on_guess(self):
            self.engine.build_model(self._expr_edit.text())
            self._built_expr = self._expr_edit.text()
            self._rebuild_param_table()

        def _on_preview(self):
            self._ensure_built()
            self.engine.preview()

        def _rebuild_param_table(self):
            self._param_table.blockSignals(True)
            self._param_table.setRowCount(0)
            self._param_row_names = []
            params = self.engine.params
            if params is not None:
                for name, p in params.items():
                    if p.expr:  # derived (e.g. fwhm/height) -- not directly editable
                        continue
                    row = self._param_table.rowCount()
                    self._param_table.insertRow(row)
                    self._param_row_names.append(name)
                    name_item = QtWidgets.QTableWidgetItem(name)
                    name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
                    self._param_table.setItem(row, 0, name_item)
                    self._param_table.setItem(row, 1, QtWidgets.QTableWidgetItem(repr(p.value)))
                    self._param_table.setItem(
                        row, 2, QtWidgets.QTableWidgetItem("" if not np.isfinite(p.min) else repr(p.min))
                    )
                    self._param_table.setItem(
                        row, 3, QtWidgets.QTableWidgetItem("" if not np.isfinite(p.max) else repr(p.max))
                    )
                    vary_item = QtWidgets.QTableWidgetItem()
                    vary_item.setFlags(vary_item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    vary_item.setCheckState(QtCore.Qt.Checked if p.vary else QtCore.Qt.Unchecked)
                    self._param_table.setItem(row, 4, vary_item)
            self._param_table.resizeColumnsToContents()
            self._param_table.setColumnWidth(0, 190)  # long names (e.g. dampedharmonicoscillator*) wrap, not truncate
            self._param_table.resizeRowsToContents()
            self._param_table.blockSignals(False)

        def _on_param_cell_changed(self, row, col):
            if row >= len(self._param_row_names):
                return
            name = self._param_row_names[row]
            params = self.engine.params
            if params is None or name not in params:
                return
            item = self._param_table.item(row, col)
            if col == 1:
                try:
                    params[name].set(value=float(item.text()))
                except ValueError:
                    pass
            elif col in (2, 3):
                field = "min" if col == 2 else "max"
                text = item.text().strip()
                try:
                    v = float(text) if text else (-np.inf if field == "min" else np.inf)
                    params[name].set(**{field: v})
                except ValueError:
                    pass
            elif col == 4:
                params[name].set(vary=item.checkState() == QtCore.Qt.Checked)

        def run_fit(self):
            self._ensure_built()
            result = self.engine.fit_with_current_params()
            self._rebuild_param_table()
            if result is not None:
                self._report_text.setPlainText(self.engine.fit_report_text())
                self._code_text.setPlainText(self.engine.get_code())
            return result

        def _copy_code(self):
            QtWidgets.QApplication.clipboard().setText(self._code_text.toPlainText())

    return QtAxesFitter


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# QtAxesFitter / IpywidgetsAxesFitter are built lazily (via module __getattr__,
# below) so importing escape.fit_gui itself never requires Qt or ipywidgets --
# only actually using that backend does. Cached so repeated access returns the
# same class object (isinstance checks stay meaningful).
_BACKEND_CLASS_BUILDERS = {"qt": _make_qt_axes_fitter_class, "ipywidgets": _make_ipywidgets_fitter_class}
_backend_class_cache = {}


def _get_backend_class(name):
    if name not in _backend_class_cache:
        _backend_class_cache[name] = _BACKEND_CLASS_BUILDERS[name]()
    return _backend_class_cache[name]


def __getattr__(name):
    key = {"QtAxesFitter": "qt", "IpywidgetsAxesFitter": "ipywidgets"}.get(name)
    if key is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get_backend_class(key)


def AxesFitter(ax=None, *, backend="auto", line=None, model_expr="gaussian, linear", **kwargs):
    """Attach an interactive lmfit fitting UI to a matplotlib ``Axes``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to attach to. Defaults to the current axes.
    backend : {"auto", "qt", "ipywidgets", "mpl"}
        "auto" (the default) picks :class:`QtAxesFitter` if a Qt matplotlib
        backend is active and a Qt binding is importable; otherwise
        :class:`IpywidgetsAxesFitter` if running in a Jupyter/IPython kernel
        with ipywidgets installed; otherwise :class:`MplAxesFitter`, which
        needs nothing beyond matplotlib itself.
    line : matplotlib.lines.Line2D, optional
        Line to fit. Defaults to the most recently added line on ``ax``.
    model_expr : str
        Initial model expression, e.g. ``"gaussian, linear"``.
    detach : bool
        Qt/ipywidgets only: split the panel out beside the figure instead of
        showing it inline/wherever the window manager puts it -- a Sidecar
        tab (``anchor="split-right"``) for ipywidgets, a window positioned
        immediately to the right of the figure's window for Qt. Not
        meaningful for the ``mpl`` backend (its panel is already outside the
        axes, in the same figure) -- passing it there raises.

    Returns
    -------
    One of :class:`QtAxesFitter`, :class:`IpywidgetsAxesFitter`, or
    :class:`MplAxesFitter` -- all expose ``.engine`` (a :class:`_FitEngine`
    with ``.result``, ``.get_code()``, ``.run_fit()``) and a ``.run_fit()``
    convenience method of their own.
    """
    ax = ax or plt.gca()
    chosen = backend if backend != "auto" else detect_backend()
    if chosen == "mpl":
        if "detach" in kwargs:
            raise ValueError("detach is not meaningful for the 'mpl' backend (its panel is already outside the axes).")
        return MplAxesFitter(ax, line=line, model_expr=model_expr, **kwargs)
    if chosen not in _BACKEND_CLASS_BUILDERS:
        raise ValueError(f"Unknown backend {chosen!r}; expected 'qt', 'ipywidgets', or 'mpl'.")
    cls = _get_backend_class(chosen)
    return cls(ax, line=line, model_expr=model_expr, **kwargs)
