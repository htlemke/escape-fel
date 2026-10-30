"""Simulated measurements for trying out / benchmarking scan strategies.

* :class:`PumpProbeSimulator` runs the full shot-by-shot generator of
  :mod:`escape.storage.test_data` (I0 fluctuations, timing jitter, pump
  drop-outs, drift, shot noise) at the requested delays and reduces it to one
  pump-on/pump-off ratio per delay with :func:`pump_probe_points`.
* :class:`GaussianSimulator` is a fast stand-in: any signal function plus
  Gaussian noise ``sqrt(noise^2 / exposure + floor^2)``.
* :class:`Instrument` adds a clock and a position: it charges motion time
  (:class:`~escape.adaptive_scan.costs.Motion`, incl. backlash), exposure and
  acquisition overhead for every measured point.
* :func:`run_plan` drives any strategy against an :class:`Instrument` until a
  time budget is spent; :func:`run_scan` is the simpler loop without costs.
"""

import numpy as np

from .costs import Motion
from .model import ImpulsiveResponse

__all__ = [
    "pump_probe_points",
    "PumpProbeSimulator",
    "GaussianSimulator",
    "Instrument",
    "run_scan",
    "run_plan",
]


def _ratio_of_sums(i, i0):
    """``sum(i) / sum(i0)`` and its (linearised) 1-sigma error."""
    r = i.sum() / i0.sum()
    err = np.sqrt(np.sum((i - r * i0) ** 2)) / i0.sum()
    return r, err


def pump_probe_points(i, i0, pump_on):
    """Per-step relative pump-probe signal from shot-resolved escape Arrays.

    Uses the I0-weighted ratio ``sum(i) / sum(i0)`` per pump state, which,
    unlike a mean of shot-wise ratios, stays well behaved for weak shots.

    Parameters
    ----------
    i, i0 : escape.Array
        Signal and incoming intensity, with scan-step structure.
    pump_on : escape.Array
        Pump flag (truthy = pumped).

    Returns
    -------
    t, y, yerr : numpy.ndarray
        Scan-parameter value (first parameter) of each step, the relative
        change ``R_on / R_off - 1`` with ``R = sum(i) / sum(i0)``, and its
        1-sigma error.
    """
    t = np.asarray(i.scan.par_steps.iloc[:, 0], float)
    y = np.full(t.size, np.nan)
    yerr = np.full(t.size, np.nan)
    for k in range(t.size):
        si, si0, sp = i.scan[k], i0.scan[k], pump_on.scan[k]
        common, a, b = np.intersect1d(si.index, si0.index, return_indices=True)
        _c, a2, c = np.intersect1d(common, sp.index, return_indices=True)
        a, b = a[a2], b[a2]
        vi = np.asarray(si.data, float)[a]
        vi0 = np.asarray(si0.data, float)[b]
        m = np.asarray(sp.data)[c].astype(bool)
        if m.sum() < 2 or (~m).sum() < 2:
            continue
        ron, son = _ratio_of_sums(vi[m], vi0[m])
        roff, soff = _ratio_of_sums(vi[~m], vi0[~m])
        y[k] = ron / roff - 1.0
        yerr[k] = ron / roff * np.hypot(son / ron, soff / roff)
    return t, y, yerr


class PumpProbeSimulator:
    """Measure delays with the shot-by-shot pump-probe generator.

    Parameters
    ----------
    response : ImpulsiveResponse, optional
        True pump-induced response; default single damped oscillation.
    shots_per_point : int
        Shots recorded per requested delay.
    seed : int, optional
        Seed; successive calls draw fresh but reproducible data.
    **kwargs
        Passed to :class:`escape.storage.test_data.TestData` (e.g. ``tjitter``).

    Call it with an array of delays (and optionally exposures, in units of
    ``shots_per_point``); returns ``(y, yerr)``.  ``shots`` counts the total
    shots spent so far.
    """

    def __init__(self, response=None, shots_per_point=200, seed=None, **kwargs):
        self.response = ImpulsiveResponse.default() if response is None else response
        self.shots_per_point = shots_per_point
        self.rng = np.random.default_rng(seed)
        self.kwargs = kwargs
        self.shots = 0

    def truth(self, t, sigma=None):
        """Expected signal, i.e. ``pump_frac * response`` smeared by the jitter."""
        from ..storage.test_data import TestData

        td = TestData(**self.kwargs)
        sigma = td.tjitter if sigma is None else sigma
        return td.pump_frac * self.response(t, sigma=sigma)

    def __call__(self, delays, exposures=None):
        from ..storage.test_data import get_test_data

        delays = np.atleast_1d(np.asarray(delays, float))
        if exposures is None:
            exposures = np.ones(delays.size)
        exposures = np.broadcast_to(np.asarray(exposures, float), delays.shape)
        y = np.full(delays.size, np.nan)
        yerr = np.full(delays.size, np.nan)
        for e in np.unique(exposures):
            sel = exposures == e
            n_shots = max(int(round(e * self.shots_per_point)), 4)
            d = get_test_data(
                delays=delays[sel],
                step_length=n_shots,
                response=self.response,
                seed=int(self.rng.integers(2**63)),
                as_da=False,
                **self.kwargs,
            )
            self.shots += sel.sum() * n_shots
            _t, y[sel], yerr[sel] = pump_probe_points(d["i"], d["i0"], d["pump_on"])
        return y, yerr


class GaussianSimulator:
    """``signal(x) + N(0, noise^2 / exposure + floor^2)``: quick synthetic measurements.

    ``response`` may be an :class:`ImpulsiveResponse` (smeared by ``sigma``)
    or any callable ``f(x)`` (1-D positions, or ``(n, d)`` for 2-D signals).
    """

    def __init__(self, response=None, noise=0.05, sigma=0.0, seed=None, floor=0.0):
        self.response = ImpulsiveResponse.default() if response is None else response
        self.noise = noise
        self.floor = floor
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.shots = 0

    def truth(self, t, sigma=None):
        if isinstance(self.response, ImpulsiveResponse):
            return self.response(t, sigma=self.sigma if sigma is None else sigma)
        return self.response(t)

    def __call__(self, delays, exposures=None):
        delays = np.asarray(delays, float)
        n = len(delays) if delays.ndim else 1
        e = np.ones(n) if exposures is None else np.broadcast_to(
            np.asarray(exposures, float), (n,)
        )
        self.shots += float(np.sum(e))
        err = np.sqrt(self.noise**2 / e + self.floor**2)
        y = self.truth(delays) + err * self.rng.standard_normal(n)
        return y, err


class Instrument:
    """A measurement function with a position and a clock.

    Parameters
    ----------
    measure : callable
        ``measure(positions, exposures) -> (y, yerr)``.
    motion : Axis, list of Axis or Motion, optional
        Positioning-time model (default: free).
    acquisition_overhead : float
        Dead time per measured point.
    position : float or array-like, optional
        Starting position (default: first point measured, no move charged).

    Calling it measures the points *in the given order*, moving from point to
    point, and adds move + exposure + overhead to :attr:`clock`.
    """

    def __init__(self, measure, motion=None, acquisition_overhead=0.0, position=None,
                 ndim=1):
        self.measure = measure
        self.motion = Motion.coerce(motion, ndim)
        self.acquisition_overhead = acquisition_overhead
        self.position = None if position is None else np.atleast_1d(
            np.asarray(position, float)
        )
        self.clock = 0.0
        self.move_time = 0.0
        self.exposure_time = 0.0
        self.n_points = 0
        self.n_reversals = 0

    def __call__(self, positions, exposures=None):
        P = np.asarray(positions, float).reshape(-1, self.motion.ndim)
        e = np.ones(len(P)) if exposures is None else np.broadcast_to(
            np.asarray(exposures, float), (len(P),)
        )
        for p, ei in zip(P, e):
            if self.position is not None:
                mt = float(self.motion.move_time(self.position, p[None, :])[0])
                d = p - self.position
                for ax, di in zip(self.motion.axes, d):
                    if ax.direction and np.sign(di) == -np.sign(ax.direction):
                        self.n_reversals += 1
            else:
                mt = 0.0
            self.move_time += mt
            self.exposure_time += ei
            self.clock += mt + ei + self.acquisition_overhead
            self.position = p
            self.n_points += 1
        pos = P[:, 0] if self.motion.ndim == 1 else P
        return self.measure(pos, e)


def run_scan(scan, measure, n_initial=15, n_rounds=10, points_per_round=5,
             callback=None):
    """Survey, then alternate fit -> suggest -> measure.

    Parameters
    ----------
    scan : AdaptiveScan
        Planner (may already contain data, then the survey is skipped).
    measure : callable
        ``measure(delays) -> (y, yerr)``.
    n_initial : int
        Points of the coarse initial survey.
    n_rounds, points_per_round : int
        Number of adaptive batches and delays per batch.
    callback : callable, optional
        ``callback(scan, round_index)`` after every batch (e.g. live plotting).

    Returns
    -------
    AdaptiveScan
    """
    if scan.n_points == 0:
        t = scan.initial_points(n_initial)
        scan.add(t, *measure(t))
        if callback is not None:
            callback(scan, 0)
    for k in range(1, n_rounds + 1):
        t = scan.suggest(points_per_round)
        scan.add(t, *measure(t))
        if callback is not None:
            callback(scan, k)
    return scan


def run_plan(strategy, instrument, budget=None, n_batches=None, batch=5,
             n_initial=None, callback=None):
    """Survey, then plan -> measure -> add, until the time budget is spent.

    Parameters
    ----------
    strategy : ScanStrategy
        Any strategy (it may already contain data; then the survey is skipped).
    instrument : Instrument
        Measurement with clock and position.
    budget : float, optional
        Stop when ``instrument.clock`` reaches this (time units).
    n_batches : int, optional
        Stop after this many batches (one of ``budget`` / ``n_batches`` needed).
    batch : int
        Points per planned batch.
    n_initial : int, optional
        Survey size (default: the strategy's ``n_initial``).
    callback : callable, optional
        ``callback(strategy, instrument, batch_index)`` after every batch.

    Returns
    -------
    strategy
    """
    if budget is None and n_batches is None:
        raise ValueError("give a time budget or a number of batches")
    if strategy.n_points == 0:
        pts = strategy.initial_points(n_initial)
        e = strategy.exposure_model.base
        y, yerr = instrument(pts, np.full(len(pts), e))
        strategy.add(pts, y, yerr, exposure=e)
        if callback is not None:
            callback(strategy, instrument, 0)
    k = 0
    while True:
        if budget is not None and instrument.clock >= budget:
            break
        if n_batches is not None and k >= n_batches:
            break
        k += 1
        plan = strategy.plan(batch, position=instrument.position)
        y, yerr = instrument(plan.positions, plan.exposures)
        strategy.add(plan.positions, y, yerr, exposure=plan.exposures)
        if callback is not None:
            callback(strategy, instrument, k)
    return strategy
