"""Interactive frequency-domain analysis attached to a matplotlib Axes.

Adds a small control panel next to an existing ``Axes`` offering three kinds
of frequency analysis on the line plotted there:

* **Power spectrum** -- periodogram, Welch's averaged periodogram, or a
  Lomb-Scargle periodogram (for unevenly-sampled x, e.g. a scan over an
  uneven parameter grid) -- via :mod:`scipy.signal`.
* **Spectrogram** -- ``scipy.signal.spectrogram``, a sliding-window FFT.
* **Wavelet scalogram** -- a continuous wavelet transform via the optional
  ``PyWavelets`` dependency (``pip install escape-fel[freq]``); scipy
  dropped its own ``cwt`` a few releases back.

The result is drawn into a separate figure (reused across repeated runs,
keyed to the source axes, rather than piling up windows), so it doesn't
disturb the layout of the plot you're analyzing. Drag across the axes (the
x-span selector) to restrict the analysis to a range; without a drag the
whole line's data is used.

Same three interchangeable front-ends as :mod:`escape.fit_gui`, sharing one
analysis engine, auto-selected by :func:`FreqAnalyzer`:

* :class:`QtFreqAnalyzer` -- a companion Qt window.
* :class:`IpywidgetsFreqAnalyzer` -- an ``ipywidgets.VBox`` panel, with
  ``detach=True`` splitting it into a JupyterLab Sidecar tab.
* :class:`MplFreqAnalyzer` -- built from ``matplotlib.widgets`` alone (a
  fixed-size ``RadioButtons`` list fits the five analysis methods well,
  unlike :mod:`escape.fit_gui`'s open-ended model catalog).

Quick start::

    import numpy as np, matplotlib.pyplot as plt
    from escape.freq_gui import FreqAnalyzer

    t = np.linspace(0, 10, 2000)
    y = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 11 * t) + 0.1 * np.random.randn(t.size)

    fig, ax = plt.subplots()
    ax.plot(t, y, label="signal")
    analyzer = FreqAnalyzer(ax)  # picks Qt / ipywidgets / matplotlib automatically

``analyzer.engine.get_code()`` returns a standalone Python script
reproducing the current analysis, for pasting into a notebook and tuning
further by hand.
"""

from __future__ import annotations

import html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal as _sp_signal

from ._axes_selection import AxesRangeSelector

METHOD_LABELS = {
    "periodogram": "Power spectrum (periodogram)",
    "welch": "Power spectrum (Welch)",
    "lombscargle": "Power spectrum (Lomb-Scargle)",
    "spectrogram": "Spectrogram",
    "scalogram": "Wavelet scalogram",
}
METHODS = list(METHOD_LABELS)

CONTINUOUS_WAVELETS = ["morl", "mexh", "cmor1.5-1.0", "gaus4", "gaus8", "cgau4", "shan1.5-1.0"]


# ---------------------------------------------------------------------------
# Pure computation (backend-agnostic)
# ---------------------------------------------------------------------------


def _uniform_dt(x, warn=True):
    """Median sample spacing, with a one-line heads-up if ``x`` looks
    unevenly sampled (methods other than Lomb-Scargle assume uniform ``dt``)."""
    dx = np.diff(x)
    dt = float(np.median(dx))
    if dt <= 0:
        raise ValueError("x must be strictly increasing for this analysis.")
    if warn:
        spread = float(np.std(dx)) / dt
        if spread > 0.05:
            print(
                f"[freq_gui] x spacing is non-uniform (relative spread {spread:.1%}); "
                "treating sampling as uniform with dt=median(diff(x)) -- consider "
                "the Lomb-Scargle method instead for irregular sampling."
            )
    return dt


def compute_periodogram(x, y):
    dt = _uniform_dt(x)
    f, P = _sp_signal.periodogram(y, fs=1.0 / dt, detrend="constant")
    return f, P


def compute_welch(x, y, nperseg=None, noverlap=None):
    dt = _uniform_dt(x)
    f, P = _sp_signal.welch(y, fs=1.0 / dt, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return f, P


def compute_lombscargle(x, y, n_freqs=1000):
    """Lomb-Scargle periodogram -- for unevenly-spaced ``x``.

    Frequency range spans ``1/(x.max()-x.min())`` up to a Nyquist-like
    ``n/(2*(x.max()-x.min()))`` set by the *average* sampling density --
    deliberately not the smallest single gap in ``x``, which for
    irregularly-spaced data can be pathologically tiny by chance and blow
    the frequency grid up to where it no longer resolves anything real.
    """
    y0 = np.asarray(y, dtype=float) - np.mean(y)
    n = len(x)
    span = float(x.max() - x.min())
    if span <= 0:
        raise ValueError("x must span a nonzero range.")
    freqs = np.linspace(1.0 / span, n / (2.0 * span), n_freqs)
    power = _sp_signal.lombscargle(x, y0, 2 * np.pi * freqs, normalize=True)
    return freqs, power


def compute_spectrogram(x, y, nperseg=None, noverlap=None):
    dt = _uniform_dt(x)
    f, t, Sxx = _sp_signal.spectrogram(y, fs=1.0 / dt, nperseg=nperseg, noverlap=noverlap)
    return f, t + x[0], Sxx


def compute_scalogram(x, y, n_scales=64, wavelet="morl"):
    """Continuous wavelet transform via the optional ``PyWavelets`` dependency."""
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "Wavelet analysis needs the optional 'PyWavelets' dependency: "
            "pip install PyWavelets  (or: pip install 'escape-fel[freq]')"
        ) from exc
    dt = _uniform_dt(x)
    scales = np.geomspace(2.0, max(4.0, len(y) / 4.0), n_scales)
    coeffs, freqs = pywt.cwt(y, scales, wavelet, sampling_period=dt)
    power = np.abs(coeffs) ** 2
    return freqs, x, power


def _fmt_array(a):
    with np.printoptions(threshold=np.inf, precision=8, floatmode="maxprec"):
        return "np.array(" + np.array2string(np.asarray(a), separator=", ") + ")"


def _pre_scroll_html(text, max_height="320px"):
    """Monospace text in a box that scrolls (both axes) instead of wrapping."""
    return (
        f"<div style='max-height:{max_height}; overflow:auto; border:1px solid #ccc; padding:4px;'>"
        f"<pre style='white-space:pre; margin:0; font-family:monospace; font-size:12px;'>"
        f"{html.escape(text)}</pre></div>"
    )


# ---------------------------------------------------------------------------
# Shared, UI-independent analysis engine
# ---------------------------------------------------------------------------


class _FreqEngine:
    """Holds analysis state and does the actual work; no widgets of its own.

    Draws into a separate output figure (created lazily, reused across runs
    by figure number so repeated clicks don't pile up windows) rather than
    onto ``ax`` itself.
    """

    def __init__(self, ax, line=None, method="welch"):
        self.sel = AxesRangeSelector(ax, line=line)
        self.method = method
        self.nperseg = None
        self.noverlap = None
        self.n_scales = 64
        self.wavelet = "morl"
        self.result = None
        self._x = None
        self._y = None
        self.out_fig = None
        self.out_ax = None

    def enable_picking(self):
        self.sel.enable_picking()

    def set_range(self, xmin, xmax):
        self.sel.set_range(xmin, xmax)

    def initial_xrange(self):
        return self.sel.initial_xrange()

    def _output_axes(self):
        key = f"escape-freq-{id(self.sel.ax)}"
        if self.out_fig is None or not plt.fignum_exists(self.out_fig.number):
            self.out_fig = plt.figure(num=key)
        else:
            self.out_fig.clf()
        self.out_ax = self.out_fig.add_subplot(111)
        return self.out_ax

    def run(self, method=None):
        if method is not None:
            self.method = method
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}. Available: {', '.join(METHODS)}")
        x, y = self.sel.xy_selection()
        if x.size < 8:
            print("[freq_gui] not enough points in the selected range to analyze.")
            return None
        self._x, self._y = x, y
        title = METHOD_LABELS[self.method]

        try:
            if self.method == "periodogram":
                f, P = compute_periodogram(x, y)
            elif self.method == "welch":
                f, P = compute_welch(x, y, nperseg=self.nperseg, noverlap=self.noverlap)
            elif self.method == "lombscargle":
                f, P = compute_lombscargle(x, y)
            elif self.method == "spectrogram":
                f, t, Sxx = compute_spectrogram(x, y, nperseg=self.nperseg, noverlap=self.noverlap)
            elif self.method == "scalogram":
                f, t, power = compute_scalogram(x, y, n_scales=self.n_scales, wavelet=self.wavelet)
        except ImportError as e:
            print(f"[freq_gui] {e}")
            return None

        ax2 = self._output_axes()
        if self.method in ("periodogram", "welch", "lombscargle"):
            ax2.semilogy(f, P)
            ax2.set_xlabel("frequency")
            ax2.set_ylabel("power")
            self.result = {"f": f, "P": P}
        elif self.method == "spectrogram":
            pcm = ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-300), shading="auto")
            self.out_fig.colorbar(pcm, ax=ax2, label="power [dB]")
            ax2.set_xlabel("time (x)")
            ax2.set_ylabel("frequency")
            self.result = {"f": f, "t": t, "Sxx": Sxx}
        else:  # scalogram
            pcm = ax2.pcolormesh(t, f, power, shading="auto")
            self.out_fig.colorbar(pcm, ax=ax2, label=r"$|CWT|^2$")
            ax2.set_yscale("log")
            ax2.set_xlabel("time (x)")
            ax2.set_ylabel("frequency")
            self.result = {"f": f, "t": t, "power": power}

        ax2.set_title(title)
        self.out_fig.tight_layout()
        self.out_fig.canvas.draw_idle()
        self.out_fig.show()
        return self.result

    def get_code(self):
        """Return a standalone Python script reproducing the current analysis."""
        if self.result is None:
            raise RuntimeError("Run an analysis first.")
        x, y = self._x, self._y
        lines = ["import numpy as np", "from scipy import signal", "", f"x = {_fmt_array(x)}", f"y = {_fmt_array(y)}", ""]

        if self.method == "periodogram":
            lines += [
                "dt = np.median(np.diff(x))",
                "f, P = signal.periodogram(y, fs=1.0 / dt, detrend='constant')",
            ]
        elif self.method == "welch":
            lines += [
                "dt = np.median(np.diff(x))",
                f"f, P = signal.welch(y, fs=1.0 / dt, nperseg={self.nperseg!r}, "
                f"noverlap={self.noverlap!r}, detrend='constant')",
            ]
        elif self.method == "lombscargle":
            lines += [
                "y0 = y - y.mean()",
                "span = x.max() - x.min()",
                "f = np.linspace(1.0 / span, len(x) / (2.0 * span), 1000)",
                "P = signal.lombscargle(x, y0, 2 * np.pi * f, normalize=True)",
            ]
        elif self.method == "spectrogram":
            lines += [
                "dt = np.median(np.diff(x))",
                f"f, t, Sxx = signal.spectrogram(y, fs=1.0 / dt, nperseg={self.nperseg!r}, "
                f"noverlap={self.noverlap!r})",
                "t = t + x[0]",
            ]
        else:  # scalogram
            lines += [
                "import pywt",
                "dt = np.median(np.diff(x))",
                f"scales = np.geomspace(2.0, max(4.0, len(y) / 4.0), {self.n_scales})",
                f"coeffs, f = pywt.cwt(y, scales, {self.wavelet!r}, sampling_period=dt)",
                "power = np.abs(coeffs) ** 2",
                "t = x",
            ]

        lines += ["", "import matplotlib.pyplot as plt", "fig, ax = plt.subplots()"]
        if self.method in ("periodogram", "welch", "lombscargle"):
            lines += [
                "ax.semilogy(f, P)",
                "ax.set_xlabel('frequency'); ax.set_ylabel('power')",
            ]
        elif self.method == "spectrogram":
            lines += [
                "pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-300), shading='auto')",
                "fig.colorbar(pcm, ax=ax, label='power [dB]')",
                "ax.set_xlabel('time'); ax.set_ylabel('frequency')",
            ]
        else:  # scalogram
            lines += [
                "pcm = ax.pcolormesh(t, f, power, shading='auto')",
                "fig.colorbar(pcm, ax=ax, label='|CWT|^2')",
                "ax.set_yscale('log')",
                "ax.set_xlabel('time'); ax.set_ylabel('frequency')",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend detection (same logic as escape.fit_gui, duplicated to avoid a
# freq_gui -> fit_gui import, which would drag in fit_gui's hard lmfit
# requirement for a feature that doesn't need lmfit at all)
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
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "qt" in backend and _qt_available():
        return "qt"
    if _ipywidgets_available():
        return "ipywidgets"
    return "mpl"


# ---------------------------------------------------------------------------
# matplotlib.widgets front-end (no extra dependencies beyond scipy)
# ---------------------------------------------------------------------------


class MplFreqAnalyzer:
    """Frequency-analysis panel built from ``matplotlib.widgets``, placed
    outside ``ax`` (via ``axes_grid1``, same mechanism as
    :class:`escape.fit_gui.MplAxesFitter`).

    The five methods are a small, fixed set, so unlike the fit panel's
    open-ended model catalog this uses ``RadioButtons`` rather than free
    text. ``nperseg``/``noverlap``/``n_scales``/``wavelet`` (whichever apply
    to the selected method) are entered as ``key=value`` pairs in a single
    options text box, e.g. ``"nperseg=256, noverlap=128"``.
    """

    def __init__(self, ax=None, line=None, method="welch", side="right"):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.ax = ax or plt.gca()
        self.fig = self.ax.figure
        self.engine = _FreqEngine(self.ax, line=line, method=method)
        self.engine.enable_picking()

        divider = make_axes_locatable(self.ax)
        self._panel = divider.append_axes(side, size="42%", pad=0.5)
        self._panel.axis("off")
        self._panel.text(0.5, 1.0, "frequency analysis", ha="center", va="top", fontsize=9, fontweight="bold")

        col_x, col_w = 0.04, 0.92
        self._ax_radio = self._panel.inset_axes([col_x, 0.55, col_w, 0.40])
        self.radio = RadioButtons(self._ax_radio, [METHOD_LABELS[m] for m in METHODS], active=METHODS.index(method))
        self.radio.on_clicked(self._on_method)

        self._ax_options = self._panel.inset_axes([col_x, 0.45, col_w, 0.08])
        self.options_box = TextBox(self._ax_options, "", initial="")
        self.options_box.on_submit(self._on_options)

        self._ax_range = self._panel.inset_axes([col_x, 0.33, col_w, 0.08])
        self.button_range = Button(self._ax_range, "Select range")
        self.button_range.on_clicked(self._toggle_span)

        self._ax_run = self._panel.inset_axes([col_x, 0.21, col_w, 0.08])
        self.button_run = Button(self._ax_run, "Run")
        self.button_run.on_clicked(lambda evt: self.run())

        self._ax_code = self._panel.inset_axes([col_x, 0.09, col_w, 0.08])
        self.button_code = Button(self._ax_code, "Code")
        self.button_code.on_clicked(lambda evt: self.show_code())

        self.span = self.engine.sel.make_span_selector(lambda a, b: self.engine.set_range(a, b))

    def _on_method(self, label):
        self.engine.method = METHODS[[METHOD_LABELS[m] for m in METHODS].index(label)]

    def _on_options(self, text):
        opts = _parse_options(text)
        for key in ("nperseg", "noverlap", "n_scales"):
            if key in opts:
                setattr(self.engine, key, opts[key])
        if "wavelet" in opts:
            self.engine.wavelet = str(opts["wavelet"])

    def _toggle_span(self, event):
        active = not self.span.active
        self.span.set_active(active)
        self.button_range.label.set_text("Range: on" if active else "Select range")
        self.fig.canvas.draw_idle()

    def run(self):
        return self.engine.run()

    def show_code(self):
        try:
            code = self.engine.get_code()
        except RuntimeError as e:
            print(f"[freq_gui] {e}")
            return None
        print(code)
        fig = plt.figure(figsize=(8, 6))
        fig.text(0.02, 0.98, code, family="monospace", fontsize=8, va="top", ha="left")
        plt.axis("off")
        fig.show()
        return fig


def _parse_options(text):
    """``"nperseg=256, noverlap=128, wavelet=mexh"`` -> a dict with numeric
    values coerced to int/float where possible."""
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, val = (s.strip() for s in item.split("=", 1))
        try:
            out[key] = int(val)
        except ValueError:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val.strip("'\"")
    return out


# ---------------------------------------------------------------------------
# ipywidgets front-end
# ---------------------------------------------------------------------------


def _make_ipywidgets_freq_analyzer_class():
    import ipywidgets as widgets
    from IPython.display import display

    class IpywidgetsFreqAnalyzer(widgets.VBox):
        """Frequency-analysis panel built from ``ipywidgets``, displayed
        below the figure (or in a Sidecar tab with ``detach=True``)."""

        def __init__(
            self, ax=None, line=None, method="welch", detach=False, title=None, width="600px"
        ):
            self.ax = ax or plt.gca()
            self.engine = _FreqEngine(self.ax, line=line, method=method)
            self.engine.enable_picking()

            self._method_dd = widgets.Dropdown(
                options=[(METHOD_LABELS[m], m) for m in METHODS], value=method, description="analysis:",
                layout=widgets.Layout(width="97%"),
            )
            self._method_dd.observe(self._on_method, names="value")

            self._nperseg_box = widgets.IntText(
                value=0, description="nperseg (0=auto)", style={"description_width": "initial"},
                layout=widgets.Layout(width="180px"),
            )
            self._noverlap_box = widgets.IntText(
                value=0, description="noverlap (0=auto)", style={"description_width": "initial"},
                layout=widgets.Layout(width="180px"),
            )
            self._nscales_box = widgets.IntText(value=64, description="n_scales", layout=widgets.Layout(width="150px"))
            self._wavelet_dd = widgets.Dropdown(
                options=CONTINUOUS_WAVELETS, value="morl", description="wavelet", layout=widgets.Layout(width="220px")
            )
            for box, attr in ((self._nperseg_box, "nperseg"), (self._noverlap_box, "noverlap")):
                box.observe(lambda ch, a=attr: setattr(self.engine, a, ch["new"] or None), names="value")
            self._nscales_box.observe(lambda ch: setattr(self.engine, "n_scales", ch["new"]), names="value")
            self._wavelet_dd.observe(lambda ch: setattr(self.engine, "wavelet", ch["new"]), names="value")

            xmin0, xmax0 = self.engine.initial_xrange()
            self._xmin_box = widgets.FloatText(value=xmin0, description="xmin", layout=widgets.Layout(width="150px"))
            self._xmax_box = widgets.FloatText(value=xmax0, description="xmax", layout=widgets.Layout(width="150px"))
            self._xmin_box.observe(self._on_range_box, names="value")
            self._xmax_box.observe(self._on_range_box, names="value")
            self._range_toggle = widgets.ToggleButton(value=False, description="Select range on plot",
                                                       layout=widgets.Layout(width="180px"))
            self._range_toggle.observe(self._on_range_toggle, names="value")

            self._run_btn = widgets.Button(description="Run", button_style="success")
            self._run_btn.on_click(lambda b: self._on_run_clicked())

            self._code_html = widgets.HTML(_pre_scroll_html(""))

            self.span = self.engine.sel.make_span_selector(self._on_span)

            super().__init__(
                [
                    widgets.HTML("<b>frequency analysis</b>"),
                    self._method_dd,
                    widgets.HBox([self._nperseg_box, self._noverlap_box, self._nscales_box, self._wavelet_dd], layout=widgets.Layout(flex_flow="row wrap")),
                    widgets.HBox([self._xmin_box, self._xmax_box, self._range_toggle], layout=widgets.Layout(flex_flow="row wrap")),
                    self._run_btn,
                    self._code_html,
                ],
                layout=widgets.Layout(border="solid 1px #ccc", padding="6px", width=width),
            )

            from escape.plot_utilities import _close_sidecar, _make_resizable, _open_sidecar, _suppress_inline_redisplay

            _make_resizable(self)
            key = f"freq_gui-{id(self.ax)}"
            _close_sidecar(key)
            if detach:
                _open_sidecar(key, title or "frequency analysis", lambda: display(self))
                _suppress_inline_redisplay(self)
            else:
                display(self)

        def _on_method(self, change):
            self.engine.method = change["new"]

        def _on_range_box(self, change):
            self.engine.set_range(self._xmin_box.value, self._xmax_box.value)

        def _on_range_toggle(self, change):
            self.span.set_active(change["new"])

        def _on_span(self, xmin, xmax):
            self.engine.set_range(xmin, xmax)
            self._xmin_box.value, self._xmax_box.value = xmin, xmax

        def _on_run_clicked(self):
            # A button callback arrives as a comm message, so the output
            # figure ``engine.run`` creates would be displayed with no cell
            # to attach to (invisible under ipympl) -- route it into the
            # figure's output host instead. See plot_utilities._run_in_output_host.
            from escape.plot_utilities import _run_in_output_host

            _run_in_output_host(self.ax.figure, self.run)

        def run(self):
            result = self.engine.run()
            if result is not None:
                self._code_html.value = _pre_scroll_html(self.engine.get_code())
            return result

    return IpywidgetsFreqAnalyzer


# ---------------------------------------------------------------------------
# Qt front-end
# ---------------------------------------------------------------------------


def _make_qt_freq_analyzer_class():
    from qtpy import QtGui, QtWidgets

    class QtFreqAnalyzer(QtWidgets.QWidget):
        """Frequency-analysis panel as a companion Qt window next to the figure."""

        def __init__(self, ax=None, line=None, method="welch", detach=False):
            self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            super().__init__()
            self.ax = ax or plt.gca()
            self.engine = _FreqEngine(self.ax, line=line, method=method)
            self.engine.enable_picking()

            self.setWindowTitle("frequency analysis panel")
            layout = QtWidgets.QVBoxLayout(self)

            self._method_cb = QtWidgets.QComboBox()
            for m in METHODS:
                self._method_cb.addItem(METHOD_LABELS[m], m)
            self._method_cb.setCurrentIndex(METHODS.index(method))
            self._method_cb.currentIndexChanged.connect(self._on_method)
            layout.addWidget(self._method_cb)

            opt_row = QtWidgets.QHBoxLayout()
            self._nperseg_spin = QtWidgets.QSpinBox()
            self._noverlap_spin = QtWidgets.QSpinBox()
            self._nscales_spin = QtWidgets.QSpinBox()
            for spin, val, attr in (
                (self._nperseg_spin, 0, "nperseg"),
                (self._noverlap_spin, 0, "noverlap"),
                (self._nscales_spin, 64, "n_scales"),
            ):
                spin.setRange(0, 100000)
                spin.setValue(val)
                spin.setSpecialValueText("auto")
                spin.valueChanged.connect(lambda v, a=attr: setattr(self.engine, a, v or None))
            self._wavelet_cb = QtWidgets.QComboBox()
            self._wavelet_cb.addItems(CONTINUOUS_WAVELETS)
            self._wavelet_cb.currentTextChanged.connect(lambda t: setattr(self.engine, "wavelet", t))
            opt_row.addWidget(QtWidgets.QLabel("nperseg"))
            opt_row.addWidget(self._nperseg_spin)
            opt_row.addWidget(QtWidgets.QLabel("noverlap"))
            opt_row.addWidget(self._noverlap_spin)
            opt_row.addWidget(QtWidgets.QLabel("n_scales"))
            opt_row.addWidget(self._nscales_spin)
            opt_row.addWidget(QtWidgets.QLabel("wavelet"))
            opt_row.addWidget(self._wavelet_cb)
            layout.addLayout(opt_row)

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

            run_btn = QtWidgets.QPushButton("Run")
            run_btn.clicked.connect(self.run)
            layout.addWidget(run_btn)

            self._code_text = QtWidgets.QPlainTextEdit(readOnly=True)
            mono_font = QtGui.QFont("Monospace", 9)
            mono_font.setStyleHint(QtGui.QFont.TypeWriter)
            self._code_text.setFont(mono_font)
            self._code_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            layout.addWidget(self._code_text)

            copy_btn = QtWidgets.QPushButton("Copy code to clipboard")
            copy_btn.clicked.connect(self._copy_code)
            layout.addWidget(copy_btn)

            self.span = self.engine.sel.make_span_selector(self._on_span)

            self.resize(460, 480)
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

        def _on_method(self, index):
            self.engine.method = self._method_cb.itemData(index)

        def _on_range_spin(self, _val):
            self.engine.set_range(self._xmin_spin.value(), self._xmax_spin.value())

        def _on_range_toggle(self, checked):
            self.span.set_active(checked)
            self._range_btn.setText("Range: on" if checked else "Select range on plot")

        def _on_span(self, xmin, xmax):
            self.engine.set_range(xmin, xmax)
            self._xmin_spin.setValue(xmin)
            self._xmax_spin.setValue(xmax)

        def run(self):
            result = self.engine.run()
            if result is not None:
                self._code_text.setPlainText(self.engine.get_code())
            return result

        def _copy_code(self):
            QtWidgets.QApplication.clipboard().setText(self._code_text.toPlainText())

    return QtFreqAnalyzer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_BACKEND_CLASS_BUILDERS = {"qt": _make_qt_freq_analyzer_class, "ipywidgets": _make_ipywidgets_freq_analyzer_class}
_backend_class_cache = {}


def _get_backend_class(name):
    if name not in _backend_class_cache:
        _backend_class_cache[name] = _BACKEND_CLASS_BUILDERS[name]()
    return _backend_class_cache[name]


def __getattr__(name):
    key = {"QtFreqAnalyzer": "qt", "IpywidgetsFreqAnalyzer": "ipywidgets"}.get(name)
    if key is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _get_backend_class(key)


def FreqAnalyzer(ax=None, *, backend="auto", line=None, method="welch", **kwargs):
    """Attach an interactive frequency-analysis UI to a matplotlib ``Axes``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to attach to. Defaults to the current axes.
    backend : {"auto", "qt", "ipywidgets", "mpl"}
        "auto" (the default) picks :class:`QtFreqAnalyzer`, then
        :class:`IpywidgetsFreqAnalyzer`, then :class:`MplFreqAnalyzer` -- see
        :func:`escape.fit_gui.AxesFitter` for the exact detection rule (this
        mirrors it).
    line : matplotlib.lines.Line2D, optional
        Line to analyze. Defaults to the most recently added line on ``ax``.
    method : {"periodogram", "welch", "lombscargle", "spectrogram", "scalogram"}
        Initial analysis method.
    detach : bool
        Qt/ipywidgets only: split the panel out beside the figure (a Sidecar
        tab for ipywidgets, a window positioned beside the figure for Qt).
        Not meaningful for the ``mpl`` backend.

    Returns
    -------
    One of :class:`QtFreqAnalyzer`, :class:`IpywidgetsFreqAnalyzer`, or
    :class:`MplFreqAnalyzer` -- all expose ``.engine`` (a :class:`_FreqEngine`
    with ``.result``, ``.get_code()``, ``.run()``) and a ``.run()``
    convenience method of their own.
    """
    ax = ax or plt.gca()
    chosen = backend if backend != "auto" else detect_backend()
    if chosen == "mpl":
        if "detach" in kwargs:
            raise ValueError("detach is not meaningful for the 'mpl' backend (its panel is already outside the axes).")
        return MplFreqAnalyzer(ax, line=line, method=method, **kwargs)
    if chosen not in _BACKEND_CLASS_BUILDERS:
        raise ValueError(f"Unknown backend {chosen!r}; expected 'qt', 'ipywidgets', or 'mpl'.")
    cls = _get_backend_class(chosen)
    return cls(ax, line=line, method=method, **kwargs)
