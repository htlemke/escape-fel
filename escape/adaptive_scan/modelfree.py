"""Model-free adaptive sampling: refine where the signal changes significantly.

:class:`GradientScan` assumes nothing about the shape of the signal.  It
looks at the measured curve as a polyline through the (averaged) points and
refines the intervals that are *long* in normalised coordinates::

    loss_i = sqrt( (dx_i / x_span)^2 + (dy_i / y_scale)^2 )

so that flat regions get coarse sampling and steep regions (edges, fast
oscillations) dense sampling -- the length-based loss of the ``adaptive``
package's ``Learner1D`` (Nijholt et al. 2019).  To make this robust against
noise, only the *significant* part of a change counts::

    dy_i = max(|y_{i+1} - y_i| - kappa * sqrt(err_i^2 + err_{i+1}^2), 0)

so pure noise does not attract points.  An optional curvature term
(``curvature > 0``) adds the significant second difference, which also
refines smooth extrema that a slope-only loss under-samples.

Each interval that still has a grid position inside is a candidate for a new
point (the grid position nearest to its middle).  Its value is the interval
loss, times the probability-like reliability ``s^2 / (s^2 + noise(e)^2)`` of
a point measured with exposure ``e`` (``s`` = half the change over the
interval), so very short exposures are not worth much where the change is
small.  Intervals that can no longer be split (grid resolution reached) can
instead get more statistics: re-measuring an end point is worth the change
over the interval, times that point's share of the interval's noise, times
the fractional noise reduction.

Like every strategy it is planned per unit time (moves, backlash, exposure)
by :class:`~escape.adaptive_scan.strategy.ScanStrategy`.
"""

import numpy as np

from .strategy import ScanStrategy

__all__ = ["ModelFreeScan", "GradientScan"]


class ModelFreeScan(ScanStrategy):
    """Model-free, noise-aware refinement of a 1-D scan (curvature or gradient).

    Parameters
    ----------
    t_range : (float, float)
        Scan range.
    loss : {"gradient", "interpolation"}
        ``"gradient"``: polyline length (density follows the slope; best for
        edges and other localised changes).  ``"interpolation"``: estimated
        linear-interpolation error ``|y''| dx^2 / 8`` from noise-shrunk second
        differences, plus ``uniform_weight * dx`` (density follows the
        curvature; better for extrema and oscillations).
    kappa : float
        Number of standard errors of a change that are treated as noise.
    curvature : float
        ``loss="gradient"`` only: weight of an added curvature (triangle
        area) term.
    uniform_weight : float
        ``loss="interpolation"`` only: weight of the interval length, which
        keeps refining flat or noise-dominated regions slowly.
    y_scale : float, optional
        Signal scale for normalising changes; default: robust range of the
        averaged data (5-95 percentile), at least three typical errors.
    dx_min : float, optional
        Do not split intervals shorter than this (besides the grid limit).
    grid_step, candidates, n_candidates, motion, exposure, explore,
    n_initial, reorder
        See :class:`~escape.adaptive_scan.strategy.ScanStrategy`.

    Examples
    --------
    >>> scan = ModelFreeScan((-1, 5), grid_step=0.02)
    >>> t = scan.initial_points()
    >>> scan.add(t, *measure(t))
    >>> while scan.max_loss() > 0.02:
    ...     plan = scan.plan(5)
    ...     scan.add(plan.positions, *measure(plan.positions), exposure=plan.exposures)
    """

    def __init__(
        self,
        t_range,
        loss="interpolation",
        kappa=1.0,
        curvature=0.0,
        uniform_weight=0.3,
        y_scale=None,
        dx_min=0.0,
        explore=0.0,
        **kwargs,
    ):
        super().__init__([t_range], explore=explore, **kwargs)
        self.t_range = tuple(float(v) for v in self.bounds[0])
        if loss not in ("gradient", "interpolation"):
            raise ValueError("loss must be 'gradient' or 'interpolation'")
        self.loss = loss
        self.kappa = kappa
        self.curvature = curvature
        self.uniform_weight = uniform_weight
        self._y_scale = y_scale
        self.dx_min = dx_min

    # ------------------------------------------------------------- averages
    def averaged(self):
        """Measured positions combined per grid position.

        Returns ``(idx, x, y, err)``: candidate index, position,
        inverse-variance weighted mean and its error, sorted by position.
        """
        if not self.n_points:
            z = np.zeros(0)
            return z.astype(int), z, z, z
        idx = self._nearest(self._X)
        w = 1.0 / self.yerr**2
        uidx, inv = np.unique(idx, return_inverse=True)
        sw = np.bincount(inv, w)
        y = np.bincount(inv, w * self.y) / sw
        return uidx, self._C[uidx, 0], y, 1.0 / np.sqrt(sw)

    def y_scale(self, y=None, err=None):
        if self._y_scale is not None:
            return self._y_scale
        if y is None:
            _i, _x, y, err = self.averaged()
        if len(y) < 2:
            return 1.0
        rng = np.percentile(y, 95) - np.percentile(y, 5)
        return float(max(rng, 3 * np.median(err), 1e-12))

    def interval_losses(self, x=None, y=None, err=None):
        """Loss of each interval between neighbouring averaged points."""
        if x is None:
            _i, x, y, err = self.averaged()
        if len(x) < 2:
            return np.zeros(0)
        span = self.t_range[1] - self.t_range[0]
        ys = self.y_scale(y, err)
        dx = np.diff(x) / span
        if self.loss == "interpolation":
            return self._interpolation_losses(x / span, y / ys, err / ys, dx)
        d = np.diff(y)
        sd = np.sqrt(err[1:] ** 2 + err[:-1] ** 2)
        sig = np.maximum(np.abs(d) - self.kappa * sd, 0.0) / ys
        loss = np.sqrt(dx**2 + sig**2)
        if self.curvature > 0 and len(x) > 2:
            d2 = y[:-2] - 2 * y[1:-1] + y[2:]
            s2 = np.sqrt(err[:-2] ** 2 + 4 * err[1:-1] ** 2 + err[2:] ** 2)
            c = np.maximum(np.abs(d2) - self.kappa * s2, 0.0) / ys
            area = 0.5 * c * (dx[:-1] + dx[1:]) / 2
            loss[:-1] += self.curvature * area
            loss[1:] += self.curvature * area
        return loss

    def _interpolation_losses(self, x, y, err, h):
        # second derivative at interior nodes (non-uniform spacing) and its noise
        c = np.zeros(len(x))
        if len(x) > 2:
            hl, hr = h[:-1], h[1:]
            f = 2.0 / (hl + hr)
            d2 = f * ((y[2:] - y[1:-1]) / hr - (y[1:-1] - y[:-2]) / hl)
            s2 = f * np.sqrt(
                (err[2:] / hr) ** 2 + (err[1:-1] * (1 / hr + 1 / hl)) ** 2
                + (err[:-2] / hl) ** 2
            )
            c[1:-1] = np.maximum(np.abs(d2) - self.kappa * s2, 0.0)
            c[0], c[-1] = c[1], c[-2]
        ci = np.maximum(c[:-1], c[1:])
        # max. linear-interpolation error ~ |y''| h^2 / 8, plus a uniform floor
        return np.sqrt((self.uniform_weight * h) ** 2 + (ci * h**2 / 8) ** 2)

    def max_loss(self):
        """Largest loss among intervals that can still be split (convergence measure)."""
        idx, x, y, err = self.averaged()
        loss = self.interval_losses(x, y, err)
        split = (np.diff(idx) > 1) & (np.diff(x) > self.dx_min)
        return float(loss[split].max()) if split.any() else 0.0

    # ------------------------------------------------------------- planning
    def _begin_plan(self, noise2):
        self._pending = {}  # candidate index -> (y, err) pseudo-measurement

    def _state(self):
        idx, x, y, err = self.averaged()
        if self._pending:
            idx, x, y, err = (list(a) for a in (idx, x, y, err))
            for k, (_y, e) in self._pending.items():
                if k in idx:
                    j = idx.index(k)
                    err[j] = 1.0 / np.sqrt(1 / err[j] ** 2 + 1 / e**2)
                else:
                    idx.append(k)
                    x.append(self._C[k, 0])
                    y.append(_y)  # interpolated when committed
                    err.append(e)
            o = np.argsort(x)
            idx, x, y, err = (np.asarray(a)[o] for a in (idx, x, y, err))
        return np.asarray(idx, int), np.asarray(x), np.asarray(y), np.asarray(err)

    def _utility(self, noise2, picks):
        idx, x, y, err = self._state()
        U = np.zeros((len(self._C), len(noise2)))
        if len(x) < 2:
            return U
        loss = self.interval_losses(x, y, err)
        ys = self.y_scale(y, err)
        noise = np.sqrt(noise2)
        d = np.abs(np.diff(y))
        sd2 = err[1:] ** 2 + err[:-1] ** 2
        for i in range(len(loss)):
            a, b = idx[i], idx[i + 1]
            if b - a > 1 and x[i + 1] - x[i] > self.dx_min:
                k = a + int(np.argmin(np.abs(self._C[a + 1 : b, 0] - 0.5 * (x[i] + x[i + 1])))) + 1
                s = max(d[i], 0.05 * ys) / 2
                U[k] = np.maximum(U[k], loss[i] * s**2 / (s**2 + noise2))
            else:
                # resolution reached: more statistics where the curve changes
                for p, j in ((a, i), (b, i + 1)):
                    e_old = err[j]
                    e_new = 1.0 / np.sqrt(1 / e_old**2 + 1 / noise2)
                    share = e_old**2 / sd2[i]
                    U[p] += d[i] / ys * share * (1 - e_new / e_old)
        return U

    def _commit(self, idx, noise2):
        _i, x, y, err = self._state()
        e = float(np.sqrt(noise2))
        if idx in self._pending:
            y0, e0 = self._pending[idx]
            self._pending[idx] = (y0, 1.0 / np.sqrt(1 / e0**2 + 1 / e**2))
        else:
            self._pending[idx] = (float(np.interp(self._C[idx, 0], x, y)), e)

    def _end_plan(self):
        self._pending = {}

    # ------------------------------------------------------------------ plot
    def plot(self, axes=None, truth=None):
        """Averaged data and the loss of each interval."""
        import matplotlib.pyplot as plt

        if axes is None:
            _fig, axes = plt.subplots(
                2, 1, sharex=True, figsize=(8, 6), height_ratios=(3, 1)
            )
        ax, ax2 = axes
        _i, x, y, err = self.averaged()
        if truth is not None:
            tt = self.candidates
            ax.plot(tt, truth(tt), color="C2", lw=1, label="truth")
        ax.errorbar(x, y, err, fmt=".-", color="k", ms=3, lw=0.6, label="averaged")
        loss = self.interval_losses(x, y, err)
        if len(loss):
            ax2.step(x, np.r_[loss, loss[-1]], where="post")
        ax.legend(fontsize="small")
        ax.set_ylabel("signal")
        ax2.set_ylabel("interval loss")
        ax2.set_xlabel("position")
        return axes


class GradientScan(ModelFreeScan):
    """:class:`ModelFreeScan` with the gradient (polyline-length) loss.

    Point density follows the significant slope: ideal for finding and
    resolving edges and other localised changes; it under-samples extrema
    (use :class:`ModelFreeScan`'s default interpolation loss there).
    """

    def __init__(self, t_range, loss="gradient", **kwargs):
        super().__init__(t_range, loss=loss, **kwargs)
