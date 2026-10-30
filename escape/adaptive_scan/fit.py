"""Weighted least-squares fit of an :class:`~escape.adaptive_scan.model.ImpulsiveResponse`.

The fit uses *variable projection*: the amplitudes/phases (and the baseline
offset) enter the model linearly and are solved for exactly at every step, so
the nonlinear optimiser only sees the time-structure parameters -- time zero
``t0``, the Gaussian smearing ``sigma``, and per component the decay rate,
frequency and (optionally) rise time.  Starting values for oscillations come
from a Lomb-Scargle periodogram of the residuals of the previous model order,
so :func:`fit_response` can grow the model one oscillation at a time and rank
the orders by BIC.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lombscargle

from .model import ImpulsiveResponse, Oscillation, _component_terms

__all__ = ["ModelSpec", "FitResult", "fit_model", "fit_response"]


@dataclass(frozen=True)
class ModelSpec:
    """Structure of the fitted model.

    Parameters
    ----------
    n_osc : int
        Number of oscillating components.
    n_exp : int
        Number of non-oscillating (step / exponential) components.
    sigma : float or None
        Fixed Gaussian smearing (timing jitter / IRF).  ``None`` fits it.
    fit_rise : bool
        Whether oscillation grow-in times are fitted (otherwise instantaneous).
    fit_exp_rate : bool
        Whether the non-oscillating components decay/recover (rate fitted) or
        are pure steps (rate fixed at 0).
    """

    n_osc: int = 1
    n_exp: int = 1
    sigma: Optional[float] = None
    fit_rise: bool = False
    fit_exp_rate: bool = True

    def nonlinear_names(self):
        names = ["t0"]
        if self.sigma is None:
            names.append("sigma")
        if self.fit_exp_rate:
            names += [f"exp{k}_rate" for k in range(self.n_exp)]
        for k in range(self.n_osc):
            names += [f"osc{k}_freq", f"osc{k}_rate"]
            if self.fit_rise:
                names.append(f"osc{k}_rise")
        return names

    def linear_names(self):
        names = ["offset"] + [f"exp{k}_amp" for k in range(self.n_exp)]
        for k in range(self.n_osc):
            names += [f"osc{k}_a", f"osc{k}_b"]
        return names

    def param_names(self):
        return self.nonlinear_names() + self.linear_names()

    def unpack(self, theta):
        """Split the nonlinear vector into ``(t0, sigma, exp_rates, osc)``."""
        theta = list(theta)
        t0 = theta.pop(0)
        sigma = theta.pop(0) if self.sigma is None else self.sigma
        if self.fit_exp_rate:
            exp_rates = [theta.pop(0) for _ in range(self.n_exp)]
        else:
            exp_rates = [0.0] * self.n_exp
        osc = []
        for _ in range(self.n_osc):
            f, r = theta.pop(0), theta.pop(0)
            rise = theta.pop(0) if self.fit_rise else 0.0
            osc.append((f, r, rise))
        return t0, sigma, exp_rates, osc

    def design(self, t, theta):
        """Design matrix ``X(t)`` whose columns multiply the linear parameters."""
        t = np.asarray(t, dtype=float)
        t0, sigma, exp_rates, osc = self.unpack(theta)
        sigma = max(float(sigma), 0.0)
        u = t - t0
        cols = [np.ones_like(u)]
        for r in exp_rates:
            cols.append(_component_terms(u, r, 0.0, 0.0, 0.0, sigma).real)
        for f, r, rise in osc:
            c = _component_terms(u, r, f, rise, 0.0, sigma)
            cols += [c.real, -c.imag]
        return np.column_stack(cols)


def _solve_linear(X, y, w):
    Xw = X * w[:, None]
    coef, *_ = np.linalg.lstsq(Xw, y * w, rcond=None)
    return coef


@dataclass
class FitResult:
    """Outcome of :func:`fit_model`; also carries what the scan design needs."""

    spec: ModelSpec
    theta: np.ndarray
    coef: np.ndarray
    cov: np.ndarray
    chi2: float
    n_points: int
    t: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    yerr: np.ndarray = field(repr=False)
    success: bool = True

    @property
    def n_params(self):
        return len(self.theta) + len(self.coef)

    @property
    def dof(self):
        return max(self.n_points - self.n_params, 1)

    @property
    def chi2_red(self):
        return self.chi2 / self.dof

    @property
    def bic(self):
        return self.chi2 + self.n_params * np.log(max(self.n_points, 2))

    @property
    def params(self):
        return np.concatenate([self.theta, self.coef])

    @property
    def param_names(self):
        return self.spec.param_names()

    @property
    def nonlinear_slice(self):
        return slice(0, len(self.theta))

    def predict(self, t, params=None):
        params = self.params if params is None else params
        k = len(self.theta)
        return self.spec.design(t, params[:k]) @ params[k:]

    def jacobian(self, t, rel_step=1e-6):
        """d model / d params at ``t`` (rows: delays, columns: :attr:`param_names`)."""
        t = np.asarray(t, dtype=float)
        k = len(self.theta)
        X = self.spec.design(t, self.theta)
        J = np.empty((t.size, self.n_params))
        scale = max(np.ptp(self.t), 1e-12) if self.t.size else 1.0
        for i in range(k):
            h = rel_step * max(abs(self.theta[i]), 1e-3 * scale)
            tp, tm = self.theta.copy(), self.theta.copy()
            tp[i] += h
            tm[i] -= h
            if tm[i] < 0 and self.param_names[i] != "t0":
                # one-sided difference at a non-negativity bound
                J[:, i] = (self.spec.design(t, tp) @ self.coef - X @ self.coef) / h
            else:
                J[:, i] = (
                    self.spec.design(t, tp) @ self.coef
                    - self.spec.design(t, tm) @ self.coef
                ) / (2 * h)
        J[:, k:] = X
        return J

    def predict_std(self, t):
        """1-sigma uncertainty of the fitted curve at ``t`` (linearised)."""
        J = self.jacobian(t)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, self.cov, J), 0.0))

    @property
    def response(self):
        """Fitted model as an :class:`ImpulsiveResponse` (without smearing)."""
        t0, sigma, exp_rates, osc = self.spec.unpack(self.theta)
        comps = []
        for i, r in enumerate(exp_rates):
            comps.append(
                Oscillation(self.coef[1 + i], decay=np.inf if r <= 0 else 1.0 / r)
            )
        j0 = 1 + len(exp_rates)
        for i, (f, r, rise) in enumerate(osc):
            a, b = self.coef[j0 + 2 * i], self.coef[j0 + 2 * i + 1]
            comps.append(
                Oscillation(
                    float(np.hypot(a, b)),
                    period=np.inf if f <= 0 else 1.0 / f,
                    decay=np.inf if r <= 0 else 1.0 / r,
                    rise=rise,
                    phase=float(np.arctan2(b, a)),
                )
            )
        return ImpulsiveResponse(comps, t0=t0, offset=self.coef[0])

    @property
    def sigma(self):
        return self.spec.unpack(self.theta)[1]

    def errors(self):
        """Parameter values and 1-sigma errors, incl. derived periods/decay times."""
        p, cov = self.params, self.cov
        err = np.sqrt(np.maximum(np.diag(cov), 0.0))
        out = {n: (v, e) for n, v, e in zip(self.param_names, p, err)}
        for n in list(out):
            v, e = out[n]
            if n.endswith("_freq") and v > 0:
                out[n.replace("_freq", "_period")] = (1 / v, e / v**2)
            if n.endswith("_rate") and v > 0:
                out[n.replace("_rate", "_decay")] = (1 / v, e / v**2)
        return out

    def summary(self):
        lines = [
            f"{self.spec}  chi2_red={self.chi2_red:.3g}  BIC={self.bic:.1f}  N={self.n_points}"
        ]
        for n, (v, e) in self.errors().items():
            lines.append(f"  {n:>14s} = {v: .5g} +/- {e:.2g}")
        return "\n".join(lines)


def _bounds(spec, t_range, f_max, sigma_max):
    span = t_range[1] - t_range[0]
    lo, hi = [t_range[0]], [t_range[1]]
    if spec.sigma is None:
        lo.append(0.0)
        hi.append(sigma_max)
    rate_max = 50.0 / span
    for _ in range(spec.n_exp if spec.fit_exp_rate else 0):
        lo.append(0.0)
        hi.append(rate_max)
    for _ in range(spec.n_osc):
        lo += [0.25 / span, 0.0]
        hi += [f_max, rate_max]
        if spec.fit_rise:
            lo.append(0.0)
            hi.append(span)
    return np.array(lo, float), np.array(hi, float)


def fit_model(t, y, yerr, spec, theta0, t_range=None, f_max=None, sigma_max=None):
    """Fit one model structure from one starting point.

    Parameters
    ----------
    t, y, yerr : array-like
        Delays, measured values, and 1-sigma errors of the values.
    spec : ModelSpec
        Model structure.
    theta0 : array-like
        Starting nonlinear parameters (see :meth:`ModelSpec.nonlinear_names`).
    t_range : (float, float), optional
        Range for ``t0`` and scale for the rate bounds; defaults to data range.
    f_max : float, optional
        Upper frequency bound; defaults to the Nyquist-like limit of the data.
    sigma_max : float, optional
        Upper bound for a fitted ``sigma``; defaults to a quarter of the range.

    Returns
    -------
    FitResult
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    floor = np.nanmedian(yerr[yerr > 0]) * 1e-3 if np.any(yerr > 0) else 1e-12
    w = 1.0 / np.maximum(yerr, floor)
    if t_range is None:
        t_range = (t.min(), t.max())
    span = t_range[1] - t_range[0]
    if f_max is None:
        f_max = default_f_max(t)
    if sigma_max is None:
        sigma_max = span / 4
    lo, hi = _bounds(spec, t_range, f_max, sigma_max)
    theta0 = np.clip(np.asarray(theta0, float), lo, hi)

    def resid(theta):
        X = spec.design(t, theta)
        coef = _solve_linear(X, y, w)
        return (y - X @ coef) * w

    try:
        res = least_squares(resid, theta0, bounds=(lo, hi), x_scale="jac")
        theta, success = res.x, res.success
    except (ValueError, np.linalg.LinAlgError):
        theta, success = theta0, False
    X = spec.design(t, theta)
    coef = _solve_linear(X, y, w)
    r = (y - X @ coef) * w
    chi2 = float(r @ r)
    result = FitResult(
        spec, theta, coef, np.zeros((0, 0)), chi2, t.size, t, y, yerr, success
    )
    J = result.jacobian(t) * w[:, None]
    fisher = J.T @ J
    # weak prior keeps degenerate directions (e.g. phase of a vanishing mode) finite
    fisher += np.diag(1e-8 * np.maximum(np.diag(fisher), 1e-12) + 1e-12)
    cov = np.linalg.pinv(fisher, hermitian=True)
    result.cov = cov * max(1.0, result.chi2_red)
    return result


def default_f_max(t):
    """Highest frequency the delay sampling can plausibly resolve."""
    ut = np.unique(np.asarray(t, float))
    if ut.size < 3:
        return 1.0 / max(np.ptp(ut), 1e-12)
    return 0.5 / np.median(np.diff(ut)) * 1.5


def _periodogram_peaks(t, r, w, f_min, f_max, n_peaks=3):
    if t.size < 4 or f_max <= f_min:
        return []
    freqs = np.linspace(f_min, f_max, 2000)
    rw = (r - np.average(r, weights=w**2)) * w
    # weighting by w approximates a generalised (noise-weighted) periodogram
    p = lombscargle(t, rw - rw.mean(), 2 * np.pi * freqs)
    peaks = []
    order = np.argsort(p)[::-1]
    df = 1.0 / max(np.ptp(t), 1e-12)
    for i in order:
        if all(abs(freqs[i] - q) > df for q in peaks):
            peaks.append(freqs[i])
        if len(peaks) >= n_peaks:
            break
    return peaks


def fit_response(
    t,
    y,
    yerr,
    max_osc=2,
    n_exp=1,
    sigma=None,
    fit_rise=False,
    t_range=None,
    f_max=None,
    start=None,
    n_peaks=5,
    max_delta_chi2=25.0,
    max_modes=3,
    t0_modes=False,
    fit_exp_rate=True,
):
    """Fit models with 0..``max_osc`` oscillations, best (lowest BIC) first.

    Each order is started from the best fit of the previous order plus a new
    oscillation at a peak of the residual periodogram; ``start`` (a list of
    earlier :class:`FitResult`) adds warm starts, which keeps successive fits
    in an adaptive scan stable.

    For orders >= 1, distinct local optima (different frequencies, e.g.
    aliases) within ``max_delta_chi2`` of the best are returned as competing
    hypotheses (at most ``max_modes`` per order); ``n_peaks`` periodogram
    peaks seed the search.  With ``t0_modes``, fits restarted from time zero
    shifted by fractions of the dominant period find competing ``t0``
    solutions as well.  For order 0 (step / exponential only) the distinct
    best time-zero solutions of a grid search are kept, which turns the fit
    into a search for the step position.  ``fit_exp_rate=False`` fits pure
    steps (no recovery).

    Returns
    -------
    list of FitResult
        Fitted models (orders and competing frequency solutions), sorted by BIC.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)
    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]
    w = 1.0 / np.maximum(yerr, 1e-12)
    if t_range is None:
        t_range = (t.min(), t.max())
    span = t_range[1] - t_range[0]
    if f_max is None:
        f_max = default_f_max(t)
    kw = dict(t_range=t_range, f_max=f_max)
    warm = {}
    for r in start or []:
        if r.spec == ModelSpec(r.spec.n_osc, n_exp, sigma, fit_rise, fit_exp_rate):
            warm.setdefault(r.spec.n_osc, []).append(r)

    def spec_for(n):
        return ModelSpec(
            n_osc=n, n_exp=n_exp, sigma=sigma, fit_rise=fit_rise,
            fit_exp_rate=fit_exp_rate,
        )

    # order 0: brute-force t0 grid, a couple of recovery rates
    spec = spec_for(0)
    sig0 = [0.02 * span] if sigma is None else []
    candidates = []
    t0_grid = np.linspace(t_range[0], t_range[1], 25)[1:-1]
    rates = (0.0, 2.0 / span) if fit_exp_rate else (None,)
    for t0 in t0_grid:
        for rate in rates:
            candidates.append([t0] + sig0 + ([rate] * n_exp if fit_exp_rate else []))
    # cheap pre-screen by chi2 at the start values, then refine the best few
    def chi2_at(theta):
        X = spec.design(t, theta)
        c = _solve_linear(X, y, w)
        r = (y - X @ c) * w
        return r @ r

    if len(spec.param_names()) >= t.size:
        return []
    candidates.sort(key=chi2_at)
    starts, seen = [], set()
    for c in candidates:  # best few with distinct time zero
        if c[0] not in seen:
            seen.add(c[0])
            starts.append(c)
        if len(starts) >= 4:
            break
    starts += [list(r.theta) for r in warm.get(0, [])]
    fits = sorted(
        (fit_model(t, y, yerr, spec, c, **kw) for c in starts), key=lambda r: r.chi2
    )
    best_of_order = fits[0]
    results = _distinct_modes(fits, span, max_delta_chi2)[:max_modes]

    for n in range(1, max_osc + 1):
        spec = spec_for(n)
        if len(spec.param_names()) >= t.size - 1:
            break
        prev = best_of_order
        resid = y - prev.predict(t)
        t0 = prev.theta[0]
        after = t > t0
        peaks = _periodogram_peaks(
            t[after], resid[after], w[after], 0.5 / span, f_max, n_peaks=n_peaks
        )
        starts = []
        for f in peaks:
            for rate in (0.5 / span, 3.0 / span):
                # a grow-in of a quarter period: rise = 0 sits on the bound,
                # where the optimiser cannot see the gradient
                th = list(prev.theta) + [f, rate] + ([0.25 / f] if fit_rise else [])
                starts.append(th)
        starts += [list(r.theta) for r in warm.get(n, [])]
        if not starts:
            break
        fits = [fit_model(t, y, yerr, spec, s, **kw) for s in starts]
        if t0_modes:
            fits += _t0_shifted_fits(min(fits, key=lambda r: r.chi2), t, y, yerr, kw)
        fits.sort(key=lambda r: r.chi2)
        best_of_order = fits[0]
        results += _distinct_modes(fits, span, max_delta_chi2)[:max_modes]

    return sorted(results, key=lambda r: r.bic)


def _t0_shifted_fits(best, t, y, yerr, kw, shifts=(-0.4, -0.2, 0.2, 0.4)):
    """Refits with time zero moved by fractions of the dominant period.

    With a free oscillation phase, moving ``t0`` can be absorbed by the
    oscillation; only the smeared step edge pins it down, so the likelihood
    often has several ``t0`` optima a fraction of a period apart.
    """
    t0, _sigma, _rates, osc = best.spec.unpack(best.theta)
    if not osc:
        return []
    amps = best.response.oscillations
    f = osc[int(np.argmax([abs(c.amplitude) for c in amps]))][0]
    out = []
    for sh in shifts:
        th = best.theta.copy()
        th[0] = t0 + sh / f
        out.append(fit_model(t, y, yerr, best.spec, th, **kw))
    return out


def _distinct_modes(fits, span, max_delta_chi2):
    """Local optima that differ clearly in frequency or time zero, none yet excluded.

    The posterior of irregularly sampled, noisy data is often multimodal --
    frequency aliases, or time zero traded against oscillation phase.  Keeping
    these competing solutions lets the scan design place points that
    discriminate between them instead of trusting one of them, and gives
    honest combined uncertainties (:meth:`AdaptiveScan.estimates`).
    """
    df = 0.5 / span
    kept = []
    for r in fits:
        if r.chi2 - fits[0].chi2 > max_delta_chi2:
            break
        f = np.sort([o[0] for o in r.spec.unpack(r.theta)[3]])
        t0, var0 = r.theta[0], r.cov[0, 0]
        distinct = True
        for _k, g, s0, v0 in kept:
            same_f = np.max(np.abs(f - g)) <= df if f.size else True
            same_t0 = abs(t0 - s0) <= 2 * np.sqrt(var0 + v0) + 1e-3 * span
            if same_f and same_t0:
                distinct = False
                break
        if distinct:
            kept.append((r, f, t0, var0))
    return [k[0] for k in kept]
