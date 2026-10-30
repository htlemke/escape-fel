"""2-D peak (beam profile / rocking map) model and adaptive 2-D scan.

:class:`Peak2D` is a tilted, optionally asymmetric 2-D peak::

    u =  cos(a) (x - x0) + sin(a) (y - y0)
    v = -sin(a) (x - x0) + cos(a) (y - y0)
    f = offset + amplitude * exp(-u^2 / (2 su^2) - v^2 / (2 sv^2))

with ``su = su_minus`` for ``u < 0`` and ``su_plus`` for ``u > 0`` (same for
``v``): a *split Gaussian* along each principal axis, which describes a
skewed profile while staying smooth (value and slope continuous) at the
centre.  ``a`` is the tilt of the ``u`` axis against ``x``.

:func:`fit_peak2d` fits it (amplitude and offset linear, the rest by
least squares from moment-based and brightest-point starts) and returns a
:class:`Peak2DFit` in canonical form (``u`` = major axis, angle in
(-90, 90] degrees).  :class:`PeakScan2D` plans 2-D scans that learn centre,
size, tilt and asymmetry fastest, with both scan axes' motion costs.
"""

from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from .strategy import ModelScan

__all__ = ["Peak2D", "Peak2DSpec", "Peak2DFit", "fit_peak2d", "PeakScan2D"]

FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _pair(s):
    s = np.atleast_1d(np.asarray(s, float))
    return (float(s[0]), float(s[-1]))


@dataclass
class Peak2D:
    """Tilted, optionally asymmetric 2-D peak (callable on ``(n, 2)`` positions).

    Parameters
    ----------
    x0, y0 : float
        Centre (position of the maximum).
    sigma_u, sigma_v : float or (minus, plus)
        Gaussian widths along the tilted axes; a pair gives different widths
        on the negative / positive side (asymmetry).
    angle : float
        Tilt of the ``u`` axis against ``x``, in radians.
    amplitude, offset : float
    """

    x0: float = 0.0
    y0: float = 0.0
    sigma_u: Union[float, Tuple[float, float]] = 1.0
    sigma_v: Union[float, Tuple[float, float]] = 1.0
    angle: float = 0.0
    amplitude: float = 1.0
    offset: float = 0.0

    def __call__(self, X):
        X = np.asarray(X, float).reshape(-1, 2)
        sm, sp = _pair(self.sigma_u)
        vm, vp = _pair(self.sigma_v)
        return self.offset + self.amplitude * _shape(
            X, self.x0, self.y0, self.angle, sm, sp, vm, vp
        )

    @property
    def fwhm(self):
        """Full widths at half maximum along ``u`` and ``v``."""
        return (
            FWHM * np.mean(_pair(self.sigma_u)),
            FWHM * np.mean(_pair(self.sigma_v)),
        )


def _shape(X, x0, y0, a, sm, sp, vm, vp):
    dx, dy = X[:, 0] - x0, X[:, 1] - y0
    c, s = np.cos(a), np.sin(a)
    u = c * dx + s * dy
    v = -s * dx + c * dy
    su = np.where(u < 0, sm, sp)
    sv = np.where(v < 0, vm, vp)
    return np.exp(-0.5 * (u / su) ** 2 - 0.5 * (v / sv) ** 2)


@dataclass(frozen=True)
class Peak2DSpec:
    """Model structure: symmetric (4 widths tied pairwise) or asymmetric."""

    asymmetric: bool = False

    def nonlinear_names(self):
        if self.asymmetric:
            return ["x0", "y0", "angle", "su_minus", "su_plus", "sv_minus", "sv_plus"]
        return ["x0", "y0", "angle", "su", "sv"]

    def param_names(self):
        return self.nonlinear_names() + ["offset", "amplitude"]

    def widths(self, theta):
        if self.asymmetric:
            return tuple(theta[3:7])
        return theta[3], theta[3], theta[4], theta[4]

    def design(self, X, theta):
        sm, sp, vm, vp = (max(w, 1e-12) for w in self.widths(theta))
        g = _shape(X, theta[0], theta[1], theta[2], sm, sp, vm, vp)
        return np.column_stack([np.ones(len(X)), g])


def _canonical(spec, theta):
    """Equivalent parameters with u = major axis and angle in [-pi/2, pi/2)."""
    th = np.array(theta, float)
    x0, y0, a = th[:3]
    sm, sp, vm, vp = spec.widths(th)
    if (sm + sp) < (vm + vp):
        # rotating the axes by +90 deg maps u' = v and v' = -u
        a = a + np.pi / 2
        sm, sp, vm, vp = vm, vp, sp, sm
    k = np.floor((a + np.pi / 2) / np.pi)
    a = a - k * np.pi
    if int(k) % 2:  # a rotation by pi flips u -> -u and v -> -v
        sm, sp, vm, vp = sp, sm, vp, vm
    if spec.asymmetric:
        return np.array([x0, y0, a, sm, sp, vm, vp])
    return np.array([x0, y0, a, sm, vm])


@dataclass
class Peak2DFit:
    """Fitted 2-D peak; the interface the scan design needs."""

    spec: Peak2DSpec
    theta: np.ndarray
    coef: np.ndarray
    cov: np.ndarray
    chi2: float
    n_points: int
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    yerr: np.ndarray = field(repr=False)

    @property
    def n_params(self):
        return len(self.theta) + len(self.coef)

    @property
    def chi2_red(self):
        return self.chi2 / max(self.n_points - self.n_params, 1)

    @property
    def bic(self):
        return self.chi2 + self.n_params * np.log(max(self.n_points, 2))

    @property
    def params(self):
        return np.concatenate([self.theta, self.coef])

    @property
    def param_names(self):
        return self.spec.param_names()

    def predict(self, X, params=None):
        params = self.params if params is None else params
        k = len(self.theta)
        return self.spec.design(np.asarray(X, float).reshape(-1, 2), params[:k]) @ params[k:]

    def jacobian(self, X, rel_step=1e-6):
        X = np.asarray(X, float).reshape(-1, 2)
        k = len(self.theta)
        J = np.empty((len(X), self.n_params))
        scale = np.ptp(self.X, axis=0).max() if len(self.X) else 1.0
        for i in range(k):
            h = rel_step * max(abs(self.theta[i]), 1e-3 * scale)
            tp, tm = self.theta.copy(), self.theta.copy()
            tp[i] += h
            tm[i] -= h
            J[:, i] = (
                self.spec.design(X, tp) @ self.coef - self.spec.design(X, tm) @ self.coef
            ) / (2 * h)
        J[:, k:] = self.spec.design(X, self.theta)
        return J

    def predict_std(self, X):
        J = self.jacobian(X)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, self.cov, J), 0.0))

    @property
    def peak(self):
        """Fitted model as a :class:`Peak2D`."""
        sm, sp, vm, vp = self.spec.widths(self.theta)
        return Peak2D(
            self.theta[0], self.theta[1], (sm, sp), (vm, vp), self.theta[2],
            self.coef[1], self.coef[0],
        )

    def errors(self):
        """Values and 1-sigma errors, incl. FWHM, angle in degrees and asymmetry."""
        p, cov = self.params, self.cov
        names = self.param_names
        out = {n: (v, float(np.sqrt(max(cov[i, i], 0.0))))
               for i, (n, v) in enumerate(zip(names, p))}

        def lin(name, grad):
            g = np.zeros(len(p))
            for k, c in grad.items():
                g[names.index(k)] = c
            out[name] = (float(g @ p), float(np.sqrt(max(g @ cov @ g, 0.0))))

        lin("angle_deg", {"angle": 180 / np.pi})
        if self.spec.asymmetric:
            lin("fwhm_u", {"su_minus": FWHM / 2, "su_plus": FWHM / 2})
            lin("fwhm_v", {"sv_minus": FWHM / 2, "sv_plus": FWHM / 2})
            for ax in ("u", "v"):
                m, pl = p[names.index(f"s{ax}_minus")], p[names.index(f"s{ax}_plus")]
                tot = m + pl
                g = {f"s{ax}_minus": -2 * pl / tot**2, f"s{ax}_plus": 2 * m / tot**2}
                gv = np.zeros(len(p))
                for k, c in g.items():
                    gv[names.index(k)] = c
                out[f"asym_{ax}"] = (
                    float((pl - m) / tot), float(np.sqrt(max(gv @ cov @ gv, 0.0)))
                )
        else:
            lin("fwhm_u", {"su": FWHM})
            lin("fwhm_v", {"sv": FWHM})
        return out


def _finalize(spec, theta, X, y, yerr):
    w = 1.0 / yerr
    D = spec.design(X, theta)
    coef, *_ = np.linalg.lstsq(D * w[:, None], y * w, rcond=None)
    r = (y - D @ coef) * w
    fit = Peak2DFit(spec, np.asarray(theta, float), coef, np.zeros((0, 0)),
                    float(r @ r), len(y), X, y, yerr)
    J = fit.jacobian(X) * w[:, None]
    F = J.T @ J
    F += np.diag(1e-8 * np.maximum(np.diag(F), 1e-12) + 1e-12)
    fit.cov = np.linalg.pinv(F, hermitian=True) * max(1.0, fit.chi2_red)
    return fit


def _moment_start(X, y, span):
    base = np.median(y)
    w = np.maximum(y - base, 0)
    if w.sum() <= 0:
        w = np.ones_like(y)
    c = (w[:, None] * X).sum(0) / w.sum()
    d = X - c
    C = (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / w.sum()
    ev, evec = np.linalg.eigh(C)
    a = np.arctan2(evec[1, 1], evec[0, 1])
    s = np.sqrt(np.clip(ev[::-1], (0.01 * span) ** 2, None))
    return [c[0], c[1], a, s[0], s[1]]


def fit_peak2d(X, y, yerr, asymmetric=False, bounds=None, start=None):
    """Fit a :class:`Peak2D` to scattered 2-D data.

    Parameters
    ----------
    X : array-like, shape (n, 2)
        Positions.
    y, yerr : array-like
        Values and 1-sigma errors.
    asymmetric : bool
        Fit split (asymmetric) widths.
    bounds : ((xlo, xhi), (ylo, yhi)), optional
        Scan area; default: data extent.
    start : list of Peak2DFit, optional
        Earlier fits used as additional starting points.

    Returns
    -------
    Peak2DFit
    """
    X = np.asarray(X, float).reshape(-1, 2)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    if bounds is None:
        bounds = np.column_stack([X.min(0), X.max(0)])
    bounds = np.asarray(bounds, float)
    span = float(np.max(bounds[:, 1] - bounds[:, 0]))
    spec = Peak2DSpec(asymmetric)
    w = 1.0 / yerr
    starts = [_moment_start(X, y, span)]
    ib = int(np.argmax(y))
    starts += [
        [X[ib, 0], X[ib, 1], 0.0, 0.1 * span, 0.1 * span],
        [X[ib, 0], X[ib, 1], np.pi / 4, 0.05 * span, 0.03 * span],
    ]
    if asymmetric:
        starts = [s[:3] + [s[3], s[3], s[4], s[4]] for s in starts]
    for r in start or []:
        th = list(r.theta)
        if r.spec.asymmetric == asymmetric:
            starts.append(th)
        elif asymmetric:
            starts.append(th[:3] + [th[3], th[3], th[4], th[4]])
    wmin = 1e-3 * span
    nw = 4 if asymmetric else 2
    lo = [bounds[0, 0] - 0.5 * span, bounds[1, 0] - 0.5 * span, -np.inf] + [wmin] * nw
    hi = [bounds[0, 1] + 0.5 * span, bounds[1, 1] + 0.5 * span, np.inf] + [2 * span] * nw

    def resid(theta):
        D = spec.design(X, theta)
        coef, *_ = np.linalg.lstsq(D * w[:, None], y * w, rcond=None)
        return (y - D @ coef) * w

    best = None
    for s in starts:
        s = np.clip(np.asarray(s, float), np.array(lo) + 1e-12, np.array(hi) - 1e-12)
        try:
            res = least_squares(resid, s, bounds=(lo, hi), x_scale="jac")
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or res.cost < best.cost:
            best = res
    theta = _canonical(spec, best.x if best is not None else np.asarray(starts[0]))
    return _finalize(spec, theta, X, y, yerr)


class PeakScan2D(ModelScan):
    """Adaptive 2-D scan for a peak's centre, size, tilt and asymmetry.

    Parameters
    ----------
    bounds : ((xlo, xhi), (ylo, yhi))
        Scan area.
    asymmetric : bool or "auto"
        Model with split widths; ``"auto"`` keeps both a symmetric and an
        asymmetric hypothesis, weighted by BIC (and measures where they differ).
    targets : str or list of str
        ``"shape"`` (default: centre, tilt and widths), ``"size"`` (widths
        only), ``"all"``, or parameter names.
    grid_step : float or (float, float), optional
        Scan grid per axis (multiples of the step, :func:`~escape.utilities.roundto`).
    n_candidates : int
        Default grid points per axis.
    motion : list of two Axis or Motion
        Positioning times of the two axes.
    explore, n_initial, exposure, reorder
        See :class:`~escape.adaptive_scan.strategy.ScanStrategy`.
    """

    def __init__(self, bounds, asymmetric="auto", targets="shape", n_candidates=41,
                 n_initial=25, explore=0.3, **kwargs):
        super().__init__(bounds, targets=targets, n_candidates=n_candidates,
                         n_initial=n_initial, explore=explore, **kwargs)
        self.asymmetric = asymmetric

    def fit(self):
        if self.n_points < 8:
            self.results = []
            return self.results
        modes = [False, True] if self.asymmetric == "auto" else [bool(self.asymmetric)]
        res = []
        for asym in modes:
            if self.n_points <= len(Peak2DSpec(asym).param_names()) + 1:
                continue
            res.append(fit_peak2d(self._X, self.y, self.yerr, asymmetric=asym,
                                  bounds=self.bounds, start=self.results))
        self.results = sorted(res, key=lambda r: r.bic)
        return self.results

    def _target_index(self, result):
        names = result.param_names
        if self.targets == "shape":
            return np.arange(len(result.theta))
        if self.targets == "size":
            return np.array([i for i, n in enumerate(names) if n.startswith("s")])
        return super()._target_index(result)

    def _describe(self, spec):
        return "asymmetric 2-D peak" if spec.asymmetric else "2-D peak"

    def plot(self, ax=None, truth=None, show="fit"):
        """Measured points (colour = value) over the fit or the acquisition map.

        ``show="fit"`` draws the fitted peak, ``"gain"`` the current
        acquisition function; ``truth`` (a callable) adds its contours.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _fig, ax = plt.subplots(figsize=(6, 5))
        xs = np.unique(self._C[:, 0])
        ys = np.unique(self._C[:, 1])
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        G = np.column_stack([XX.ravel(), YY.ravel()])
        if self.best is not None:
            Z = self.best.predict(G) if show == "fit" else self.acquisition()[0]
            if show != "fit":
                Z = Z[self._nearest(G)]
            im = ax.pcolormesh(xs, ys, Z.reshape(XX.shape).T, shading="auto",
                               cmap="viridis")
            plt.colorbar(im, ax=ax, label="fit" if show == "fit" else "information")
        if truth is not None:
            ax.contour(xs, ys, truth(G).reshape(XX.shape).T, colors="w",
                       linewidths=0.7)
        ax.scatter(self._X[:, 0], self._X[:, 1], c=self.y, s=10, cmap="magma",
                   edgecolors="k", linewidths=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax
