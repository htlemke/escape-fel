"""Peak/step-finding and its plot overlay, shared by :mod:`escape.plot_utilities`
and :mod:`escape.stream.plots`.

Split out into its own leaf module (numpy + matplotlib only, no
ipywidgets/dask/IPython) so both can use the same implementation without
either depending on the other -- this used to be two hand-maintained
copies of the same code, which is exactly the kind of thing that drifts
(and did: a bug fix here previously had to be applied twice).
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def _skew(a):
    """Sample skewness (3rd standardized moment) -- avoids a hard
    ``scipy.stats`` dependency for the one number :func:`find_peak` needs
    from it."""
    a = np.asarray(a, dtype=float)
    s = a.std()
    if s == 0:
        return 0.0
    return float(np.mean((a - a.mean()) ** 3) / s**3)


# The level (fraction of the way from one asymptote to the other) at which
# an erf-shaped step -- the integral of a Gaussian -- crosses the
# half-maximum points of that underlying Gaussian: analytically
# 0.5 * (1 +/- erf(sqrt(ln 2))), i.e. ~11.95%/88.05% -- see find_peak's
# step case.
_STEP_LEVEL_HI = 0.5 * (1 + math.erf(math.sqrt(math.log(2))))
_STEP_LEVEL_LO = 1 - _STEP_LEVEL_HI


def find_peak(x, y, n_bg=3, bg_model="linear", fixed_offset=None, mode="auto", plot=False):
    """Locate a peak (or step) in a 1-D scan trace.

    A numpy/Python-3 port of the lab's old ``PeakAnalysis`` tool (originally
    ``eco.utilities.PeakAnalysis``, Python 2 only and unusable as-is): fits a
    background through the first/last ``n_bg`` points and subtracts it,
    classifies the trace as peak-shaped or step-shaped by comparing the
    skewness of the background-subtracted signal to that of its derivative
    (a step's derivative is peak-shaped, which is what the comparison
    distinguishes) unless *mode* forces one, then finds the center and width.
    Peaks may be positive or negative (a dip) -- the feature is oriented by
    its signed integral before locating it, so either works the same way.

    - Peak case: center/FWHM from linear interpolation of the half-maximum
      crossings on either side of the maximum -- the standard FWHM
      definition.
    - Step case: ``center`` is the 50%-level ("half-rise") crossing between
      the two asymptotic levels (mean of the first/last ``n_bg`` points) on
      the actual curve. ``fwhm`` comes from that same curve's crossings, near
      *that* center point, at the two levels symmetric around the 50% point
      where an erf-shaped step (the integral of a Gaussian) reaches the
      half-maximum points of that Gaussian -- analytically
      ``0.5 * (1 +/- erf(sqrt(ln 2)))``, i.e. ~11.95%/88.05% of the way from
      one asymptote to the other. This *is* (up to rounding) the original
      ``PeakAnalysis``'s empirical ``0.1195`` constant -- it just wasn't
      derived there. "Near that center point" specifically: found by walking
      outward from the center's index in each direction and linearly
      interpolating at the first level crossing found, not a global
      nearest-level match -- a noisy or non-monotonic baseline could
      otherwise satisfy the target level somewhere far from the actual
      transition. Reading the width straight off the curve at these levels,
      rather than off its numerical derivative, also avoids differentiation
      noise on noisy/live data. ``center`` and the
      half-width of ``fwhm`` generally don't coincide exactly (they come
      from different crossings), unlike the peak case where they do by
      construction.

    Parameters
    ----------
    x, y : array-like
        The scan trace (need not be pre-sorted). Non-finite points are
        dropped before analysis.
    n_bg : int
        Number of points at each end used to estimate the background /
        asymptotic levels.
    bg_model : "linear" or "offset"
        How the background is estimated from the ``n_bg`` edge points
        (ignored if *fixed_offset* is given). ``"linear"`` (default) fits a
        line through them (handles a sloped background/baseline drift);
        ``"offset"`` uses their plain mean instead (a flat background --
        use this if a sloped fit is overreacting to edge noise on a
        genuinely flat baseline).
    fixed_offset : float, optional
        Skip background estimation entirely and subtract this constant
        instead -- for a known/fixed baseline (e.g. a detector's dark
        level) rather than one estimated from this particular trace.
    mode : "auto", "peak", or "step"
        ``"auto"`` (default) classifies via the skewness comparison
        described above. ``"peak"``/``"step"`` forces that branch instead
        -- use this when the automatic classification picks the wrong one
        (e.g. a noisy or asymmetric trace).
    plot : bool or Axes/Line2D/Figure, optional
        If given, draw the same overlay :class:`escape.plot_utilities.PeakAnalyzer`
        draws (center line, FWHM/crossing markers, background curve or
        asymptotic levels, and a text readout) on top of an existing plot --
        see :func:`_update_peak_overlay`. ``True`` uses the current axes
        (``plt.gca()``); an explicit ``Axes``, ``Line2D``, or ``Figure``
        draws on that axes (a line's or figure's own axes) instead. A
        second call on the same axes replaces the previous overlay rather
        than stacking a new one on top of it. Default ``False`` draws
        nothing. Ignored (nothing drawn) if the trace can't be analyzed --
        see Returns below.

    Returns
    -------
    dict, or ``None`` if the trace has too few finite points
    (< ``2 * n_bg + 3``) or no variation (flat) to analyze -- callers meant
    to run on live, possibly-incomplete data should treat ``None`` as
    "nothing to show yet", not an error. Keys:

    - ``center``, ``fwhm``, ``is_peak``: as above.
    - ``peak_x``, ``peak_y`` : the peak/dip extremum's coordinates for a
      peak; for a step, just ``center``'s coordinates again (there's no
      single extremum point to mark there -- see
      :func:`_update_peak_overlay`, which doesn't draw a point marker for
      the step case).
    - ``crossing_1``, ``crossing_2`` : ``(x, y)`` tuples -- the two points
      used to determine the width (the half-max crossings on the curve for
      a peak, or the ~11.95%/88.05% level crossings on the curve for a step
      -- see above), in the original data's coordinates, for plotting
      directly on top of the raw trace.
    - ``background`` : ``(x, y)`` arrays of the fitted/constant background
      that was subtracted before locating the peak, spanning the data --
      or ``None`` for a step (see ``levels`` instead).
    - ``levels`` : ``(level_before, level_after)`` -- the two asymptotic
      levels (mean of the first/last ``n_bg`` points) a step was measured
      between -- or ``None`` for a peak.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 * n_bg + 3 or np.ptp(y) == 0:
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]

    xb = np.concatenate([x[:n_bg], x[-n_bg:]])
    yb = np.concatenate([y[:n_bg], y[-n_bg:]])
    if fixed_offset is not None:
        b = np.full_like(x, float(fixed_offset))
    elif bg_model == "offset":
        b = np.full_like(x, yb.mean())
    else:
        a = np.polyfit(xb, yb, 1)
        b = np.polyval(a, x)
    yf = y - b
    yd = np.diff(yf)

    if mode == "peak":
        is_peak = True
    elif mode == "step":
        is_peak = False
    else:
        is_peak = abs(_skew(yf)) > abs(_skew(yd))

    if is_peak:
        xw, yw = x, yf
        # Orient the feature to point "up" (positive-going), signed area
        # under yw. `sign` undoes this for anything that needs to go back
        # to the real (unflipped) signal afterward -- see crossing_1/2
        # below.
        sign = 1.0
        if np.sum((xw[1:] - xw[:-1]) * (yw[1:] + yw[:-1]) / 2) < 0:
            yw = -yw
            sign = -1.0

        mm = int(np.argmax(yw))
        half = yw[mm] / 2

        def _crossing(indices):
            for i in indices:
                if yw[i] < half:
                    j = i + 1 if i < mm else i - 1
                    if j < 0 or j >= len(xw) or yw[j] == yw[i]:
                        return xw[i]
                    frac = (half - yw[i]) / (yw[j] - yw[i])
                    return xw[i] + frac * (xw[j] - xw[i])
            return None

        xhm1 = _crossing(range(mm, -1, -1))
        xhm2 = _crossing(range(mm, len(xw)))
        if xhm1 is None or xhm2 is None:
            return None
        center, fwhm = (xhm1 + xhm2) / 2, abs(xhm2 - xhm1)
        peak_x = xw[mm]
        peak_y = y[int(np.argmin(np.abs(x - peak_x)))]

        # xhm1/xhm2 live in the background-subtracted domain (half of the
        # subtracted peak's height, in the possibly sign-flipped working
        # array) -- add the background back at each crossing's x to place
        # the marker on the original, visible curve. For a negative peak
        # (a dip), `half` is positive in the flipped array but the real
        # crossing sits *below* the background, not above it -- `sign`
        # (from the orientation flip above) corrects for that; without it
        # the crossing markers land mirrored above the background instead
        # of between it and the dip.
        y_hm = sign * half + np.interp([xhm1, xhm2], x, b)
        crossing_1, crossing_2 = (float(xhm1), float(y_hm[0])), (float(xhm2), float(y_hm[1]))
        background = (x.copy(), b.copy())
        levels = None
    else:
        lev0, lev1 = y[:n_bg].mean(), y[-n_bg:].mean()

        def _level_crossing(level):
            gg = int(np.argmin(np.abs(y - level)))
            j = gg + 1 if gg + 1 < len(x) else gg - 1
            if j < 0 or y[j] == y[gg]:
                return x[gg]
            frac = (level - y[gg]) / (y[j] - y[gg])
            return x[gg] + frac * (x[j] - x[gg])

        # The half-rise point: where the actual curve crosses halfway
        # between the two baselines. A global nearest-level match is fine
        # here -- on a well-behaved step trace it's expected to occur once,
        # near the middle.
        center = _level_crossing(lev0 + 0.5 * (lev1 - lev0))

        # The 12%/88% crossings, by contrast, are found by walking outward
        # from the *index* nearest that center point, separately in each
        # direction, and stopping at the first level crossing -- rather
        # than a global nearest-level match, which a noisy or non-monotonic
        # baseline could otherwise satisfy somewhere far from the actual
        # transition (same "walk outward from a known-good anchor" idea as
        # the peak case's crossing search above, anchored on the center
        # index instead of the extremum index).
        i0 = int(np.argmin(np.abs(x - center)))
        # Normalized so the low-x asymptote is 0 and the high-x asymptote is
        # 1 regardless of whether the step rises or falls (dividing by a
        # negative lev1 - lev0 flips sign consistently) -- so "below"/
        # "above" below always means "toward lev0"/"toward lev1".
        yn = (y - lev0) / (lev1 - lev0)

        def _walk_crossing(indices, level, below):
            prev = i0
            for i in indices:
                crossed = yn[i] <= level if below else yn[i] >= level
                if crossed:
                    if yn[i] == yn[prev]:
                        return x[i]
                    frac = (level - yn[prev]) / (yn[i] - yn[prev])
                    return x[prev] + frac * (x[i] - x[prev])
                prev = i
            return None

        # Width: the curve's own *nearest* crossings at the ~11.95%/88.05%
        # levels (_STEP_LEVEL_LO/_STEP_LEVEL_HI) rather than an arbitrary
        # 10-90% level crossing, or the derivative's half-max crossings --
        # see the docstring above for the derivation.
        x_lo = _walk_crossing(range(i0, -1, -1), _STEP_LEVEL_LO, below=True)
        x_hi = _walk_crossing(range(i0, len(x)), _STEP_LEVEL_HI, below=False)
        if x_lo is None or x_hi is None:
            return None
        fwhm = abs(x_hi - x_lo)
        crossing_1 = (float(x_lo), float(np.interp(x_lo, x, y)))
        crossing_2 = (float(x_hi), float(np.interp(x_hi, x, y)))
        background = None
        levels = (float(lev0), float(lev1))
        peak_x, peak_y = float(center), float(np.interp(center, x, y))

    result = dict(
        center=float(center), fwhm=float(fwhm),
        peak_x=float(peak_x), peak_y=float(peak_y), is_peak=bool(is_peak),
        crossing_1=crossing_1, crossing_2=crossing_2,
        background=background, levels=levels,
    )

    if plot:
        ax = _resolve_plot_target(plot)
        drawn = _update_peak_overlay(
            ax, getattr(ax, "_escape_peak_overlay", None), x, y,
            n_bg=n_bg, bg_model=bg_model, fixed_offset=fixed_offset, mode=mode,
        )
        ax._escape_peak_overlay = drawn
        _draw_safe(ax.figure)

    return result


# Artists a peak overlay draws, so _update_peak_overlay can remove and
# redraw them together on every live-plot update / button re-click.
_PEAK_OVERLAY_COLOR = "crimson"


def _draw_safe(fig):
    """Schedule a redraw. Safe to call from a backend timer callback (GUI thread)."""
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


def _resolve_plot_target(plot):
    """Resolve a :func:`find_peak` ``plot=`` argument to the ``Axes`` to draw
    the overlay on: ``True`` for the current axes (``plt.gca()``), or an
    explicit ``Axes``/``Line2D``/``Figure`` handle. An ``Axes`` is
    self-referential through its own ``.axes`` (``ax.axes is ax``), so the
    same ``plot.axes`` lookup resolves both it and a ``Line2D`` (to its
    parent axes) without needing to type-check which one it is."""
    if plot is True:
        return plt.gca()
    if hasattr(plot, "gca"):  # Figure
        return plot.gca()
    return plot.axes  # Axes or Line2D


def _update_peak_overlay(ax, drawn, x, y, n_bg=3, bg_model="linear", fixed_offset=None, mode="auto"):
    """(Re)draw the peak-analysis overlay on ``ax`` from the current
    ``x``/``y`` trace, removing whatever ``drawn`` (a previous call's
    return value) left behind first.

    Draws, from :func:`find_peak`'s result: a center reference line and a
    text readout, plus, depending on ``is_peak`` -- a visual sanity check
    of every quantity the analysis used, not just its answer:

    - Peak case: the two FWHM crossing lines (at ``center +/- fwhm/2``, the
      same points as ``crossing_1``/``crossing_2``), the subtracted
      background curve, the crossing points themselves marked on the
      curve, and the peak/dip point labeled with its (x, y) coordinates.
    - Step case: the two asymptotic baseline levels, and the two
      width-determining points (the curve's own ~11.95%/88.05% level
      crossings, see :func:`find_peak`) as vertical lines rather than
      points on the curve -- there's no single "step point" to mark the
      way there's an unambiguous extremum for a peak.

    ``n_bg``/``bg_model``/``fixed_offset``/``mode`` are the defaults used
    when nothing else overrides them; if a
    :class:`escape.plot_utilities.PeakAnalyzer` panel is attached to ``ax``,
    its current settings (``ax._escape_peak_params``, including whether to
    show the overlay at all) take precedence -- so a live-streaming plot
    calling this with its own defaults on every redraw still respects the
    panel once one is attached.

    Returns the new ``drawn`` dict (pass it back in next time), or ``None``
    if the overlay is switched off or :func:`find_peak` couldn't analyze
    this trace (too few points, flat) -- callers should treat that the
    same as "no overlay drawn".
    """
    live = getattr(ax, "_escape_peak_params", None)
    if live is not None:
        n_bg = live.get("n_bg", n_bg)
        bg_model = live.get("bg_model", bg_model)
        fixed_offset = live.get("fixed_offset", fixed_offset)
        mode = live.get("mode", mode)
        show = live.get("show", True)
    else:
        show = True

    if drawn is not None:
        for artist in drawn["artists"]:
            try:
                artist.remove()
            except Exception:
                pass
    if not show:
        return None

    result = find_peak(x, y, n_bg=n_bg, bg_model=bg_model, fixed_offset=fixed_offset, mode=mode)
    if result is None:
        return None
    center, fwhm, is_peak = result["center"], result["fwhm"], result["is_peak"]
    kind = "peak" if is_peak else "step"
    artists = [
        ax.axvline(center, color=_PEAK_OVERLAY_COLOR, ls="--", lw=1, alpha=0.8),
        ax.text(
            0.02, 0.98, f"{kind}: center={center:.4g}\nFWHM={fwhm:.4g}",
            transform=ax.transAxes, va="top", ha="left", color=_PEAK_OVERLAY_COLOR, fontsize=9,
        ),
    ]
    if is_peak:
        # center +/- fwhm/2 == crossing_1/2's x exactly (by construction --
        # both come straight from the same half-max crossings), so either
        # pair of vertical lines marks the same two points.
        artists.append(ax.axvline(center - fwhm / 2, color=_PEAK_OVERLAY_COLOR, ls=":", lw=1, alpha=0.5))
        artists.append(ax.axvline(center + fwhm / 2, color=_PEAK_OVERLAY_COLOR, ls=":", lw=1, alpha=0.5))
        bx, by = result["background"]
        artists.append(ax.plot(bx, by, "-.", color=_PEAK_OVERLAY_COLOR, lw=1, alpha=0.5)[0])
        for cx, cy in (result["crossing_1"], result["crossing_2"]):
            artists.append(ax.plot([cx], [cy], "x", color=_PEAK_OVERLAY_COLOR, ms=8, mew=1.5)[0])
        artists.append(
            ax.plot(
                [result["peak_x"]], [result["peak_y"]], "o",
                color=_PEAK_OVERLAY_COLOR, mfc="none", mec=_PEAK_OVERLAY_COLOR, ms=9, mew=1.5,
            )[0]
        )
        artists.append(
            ax.annotate(
                f"({result['peak_x']:.4g}, {result['peak_y']:.4g})",
                xy=(result["peak_x"], result["peak_y"]), xytext=(6, 6), textcoords="offset points",
                color=_PEAK_OVERLAY_COLOR, fontsize=8,
            )
        )
    else:
        # Step case: no single "step point" marker (there isn't one, the
        # way there's an unambiguous peak/dip extremum for the peak case)
        # -- just the two baselines and, as vertical lines rather than
        # points on the curve, the half-rise center (above) and the two
        # width points (crossing_1/2, the curve's own ~11.95%/88.05% level
        # crossings -- see find_peak).
        for lev in result["levels"]:
            artists.append(ax.axhline(lev, color=_PEAK_OVERLAY_COLOR, ls="-.", lw=1, alpha=0.5))
        for cx, cy in (result["crossing_1"], result["crossing_2"]):
            artists.append(ax.axvline(cx, color=_PEAK_OVERLAY_COLOR, ls=":", lw=1, alpha=0.5))
    # Marked (rather than relying on color alone, which one missing
    # `color=` kwarg silently breaks -- see the peak-point marker's history)
    # so every "find the data line" scan elsewhere in escape (this module's
    # own Peak/Peak-params buttons, and escape.fit_gui's/escape.freq_gui's
    # shared escape._axes_selection.snapshot_data_lines) can reliably skip
    # every artist this function draws, in either attachment order.
    for artist in artists:
        artist._escape_overlay = True
    return {"artists": artists, "result": result}
