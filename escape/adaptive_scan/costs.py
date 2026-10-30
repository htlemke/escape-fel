"""What a scan point costs: positioning, exposure, and the measurement grid.

* :class:`Axis` -- time to move one scan axis (motor, delay stage) from one
  value to another: per-move overhead (command latency, settling), a
  speed/acceleration travel profile (so small moves are relatively cheap),
  and direction dependence (backlash correction: positions approached
  against the preferred direction are overshot and approached again).
  :meth:`Axis.calibrate` fits the model to logged move durations.
* :class:`Motion` -- several axes, moved in parallel or one after another.
* :class:`ExposureModel` -- the exposure times (statistics) that may be
  chosen per point and the noise each gives, ``noise(e) = sqrt(c^2/e + floor^2)``;
  ``c`` and ``floor`` are estimated from the measured points automatically.
* :func:`order_route` -- the fastest order to visit a set of points
  (asymmetric travelling-salesman path; exact for small batches).
* :func:`grid_points` -- the scan grid as produced by
  :func:`escape.utilities.roundto` (multiples of a step size).
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

__all__ = [
    "Axis",
    "Motion",
    "ExposureModel",
    "order_route",
    "route_time",
    "grid_points",
    "snap_to_grid",
]


def _roundto(v, interval):
    from ..utilities import roundto

    return roundto(v, interval)


def grid_points(lo, hi, step):
    """All multiples of ``step`` within ``[lo, hi]`` (the :func:`~escape.utilities.roundto` grid)."""
    k0 = np.ceil(lo / step - 1e-9)
    k1 = np.floor(hi / step + 1e-9)
    return _roundto(np.arange(k0, k1 + 1) * step, step)


def snap_to_grid(x, grid):
    """Nearest grid value(s) for ``x`` (1-D grid)."""
    grid = np.asarray(grid, float)
    idx = np.abs(grid[None, :] - np.atleast_1d(x)[:, None]).argmin(1)
    return grid[idx]


@dataclass
class Axis:
    """Positioning-time model of one scan axis.

    Parameters
    ----------
    speed : float
        Maximum speed (units per second).  ``inf`` = instantaneous travel.
    acceleration : float
        Acceleration (units per second^2) of a trapezoidal velocity profile.
        ``inf`` = constant speed.  With a finite acceleration, short moves
        take ``2 sqrt(d / acceleration)``, i.e. relatively less per unit.
    overhead : float
        Fixed time per move (command latency, settling, encoder check).
    backlash : float
        Overshoot distance for positions approached against ``direction``:
        the axis moves past the target by ``backlash`` and comes back, which
        costs a second move (and a second ``overhead``).
    direction : {+1, -1, 0}
        Preferred final approach direction; ``0`` = no direction dependence.
    reverse_penalty : float
        Extra time for moves against ``direction`` (an alternative or addition
        to ``backlash``, e.g. from :meth:`calibrate`).
    """

    speed: float = np.inf
    acceleration: float = np.inf
    overhead: float = 0.0
    backlash: float = 0.0
    direction: int = 1
    reverse_penalty: float = 0.0

    def travel(self, d):
        """Travel time for distance(s) ``d`` >= 0 (without overhead)."""
        d = np.abs(np.asarray(d, float))
        v, a = self.speed, self.acceleration
        if np.isinf(a):
            return d / v if np.isfinite(v) else np.zeros_like(d)
        if np.isinf(v):
            return 2.0 * np.sqrt(d / a)
        d_ramp = v**2 / a
        return np.where(d < d_ramp, 2.0 * np.sqrt(d / a), d / v + v / a)

    def move_time(self, a, b):
        """Time to move from ``a`` to each of ``b`` (scalar ``a``, array ``b``)."""
        b = np.asarray(b, float)
        d = b - float(a)
        t = np.where(d == 0, 0.0, self.overhead + self.travel(d))
        if self.direction:
            against = np.sign(d) == -np.sign(self.direction)
            if self.backlash > 0:
                t = np.where(
                    against,
                    2 * self.overhead
                    + self.travel(np.abs(d) + self.backlash)
                    + self.travel(self.backlash),
                    t,
                )
            t = np.where(against, t + self.reverse_penalty, t)
        return t

    @classmethod
    def calibrate(cls, start, end, duration, direction=1):
        """Fit overhead, speed and reverse penalty to logged moves.

        Least squares on ``duration = overhead + |end-start| / speed
        + reverse_penalty * [move against direction]`` (constant-speed model;
        use the result as a starting point if the axis accelerates slowly).
        """
        start, end, duration = (np.asarray(a, float) for a in (start, end, duration))
        d = end - start
        moved = d != 0
        cols = [np.ones(moved.sum()), np.abs(d[moved])]
        if direction:
            cols.append((np.sign(d[moved]) == -np.sign(direction)).astype(float))
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, duration[moved], rcond=None)
        coef = np.maximum(coef, 0.0)
        speed = 1.0 / coef[1] if coef[1] > 0 else np.inf
        return cls(
            speed=speed,
            overhead=float(coef[0]),
            direction=direction,
            reverse_penalty=float(coef[2]) if direction else 0.0,
        )


class Motion:
    """Several :class:`Axis` objects driven together.

    Parameters
    ----------
    axes : Axis or list of Axis
        One axis per scan dimension.
    parallel : bool
        Axes move simultaneously (time = slowest axis) or one after another
        (time = sum).
    overhead : float
        Extra time per point that moves at all (e.g. a common settle wait).
    """

    def __init__(self, axes=None, parallel=True, overhead=0.0):
        if axes is None:
            axes = [Axis()]
        if isinstance(axes, Axis):
            axes = [axes]
        self.axes = list(axes)
        self.parallel = parallel
        self.overhead = overhead

    @property
    def ndim(self):
        return len(self.axes)

    def move_time(self, a, B):
        """Time from position ``a`` (shape ``(d,)``) to each row of ``B`` (``(n, d)``)."""
        a = np.atleast_1d(np.asarray(a, float))
        B = np.asarray(B, float).reshape(-1, len(self.axes))
        T = np.column_stack(
            [ax.move_time(a[i], B[:, i]) for i, ax in enumerate(self.axes)]
        )
        t = T.max(1) if self.parallel else T.sum(1)
        moved = np.any(B != a[None, :], axis=1)
        return t + self.overhead * moved

    @classmethod
    def coerce(cls, motion, ndim):
        """Accept ``None`` (free motion), an :class:`Axis`, a list of axes or a Motion."""
        if motion is None:
            return cls([Axis(direction=0) for _ in range(ndim)])
        if isinstance(motion, Motion):
            return motion
        return cls(motion)


class ExposureModel:
    """Exposure-time choices and the noise each produces.

    Parameters
    ----------
    options : None, "auto" or array-like
        Exposure times (or shots, any unit consistent with ``overhead`` and
        the motion times) the planner may choose per point.  ``None`` =
        always ``base``; ``"auto"`` = ``base * [0.5, 1, 2, 4, 8]``.
    base : float
        Default exposure, also used for space-filling and survey points.
    overhead : float
        Dead time per acquisition (readout, trigger).
    coeff, floor : float, optional
        Noise model ``noise(e) = sqrt(coeff^2 / e + floor^2)``.  Left at
        ``None`` they are estimated from the measured errors: ``floor`` only
        when points with at least two different exposures exist (else 0).
        ``floor`` is the part of the noise that more statistics does not
        reduce (drifts, systematic jitter).
    """

    def __init__(self, options=None, base=1.0, overhead=0.0, coeff=None, floor=None):
        self.base = float(base)
        if options is None:
            options = [self.base]
        elif isinstance(options, str):
            if options != "auto":
                raise ValueError("options must be None, 'auto' or a list")
            options = self.base * np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        self.options = np.unique(np.asarray(options, float))
        self.overhead = float(overhead)
        self._coeff_fixed, self._floor_fixed = coeff, floor
        self.coeff = 1.0 if coeff is None else float(coeff)
        self.floor = 0.0 if floor is None else float(floor)

    def noise(self, e):
        e = np.asarray(e, float)
        return np.sqrt(self.coeff**2 / e + self.floor**2)

    def calibrate(self, yerr, exposure):
        """Estimate ``coeff`` (and ``floor``) from measured errors."""
        yerr = np.asarray(yerr, float)
        exposure = np.asarray(exposure, float)
        ok = np.isfinite(yerr) & (yerr > 0) & (exposure > 0)
        if not ok.any():
            return self
        yerr, exposure = yerr[ok], exposure[ok]
        floor = self._floor_fixed
        if floor is None:
            floor = 0.0
            if np.unique(exposure).size >= 2:
                # yerr^2 = coeff^2 / e + floor^2, robustly via per-exposure medians
                ue = np.unique(exposure)
                v = np.array([np.median(yerr[exposure == e] ** 2) for e in ue])
                A = np.column_stack([1 / ue, np.ones_like(ue)])
                (c2, f2), *_ = np.linalg.lstsq(A, v, rcond=None)
                floor = np.sqrt(max(f2, 0.0))
        self.floor = float(floor)
        if self._coeff_fixed is None:
            c = np.sqrt(np.maximum(yerr**2 - self.floor**2, 1e-6 * yerr**2)) * np.sqrt(
                exposure
            )
            self.coeff = float(np.median(c))
        return self

    def set_noise(self, noise):
        """Set the model so that one point at ``base`` exposure has error ``noise``."""
        self.coeff = float(np.sqrt(max(noise**2 - self.floor**2, 1e-30) * self.base))
        return self


def route_time(points, start, motion):
    """Total motion time to visit ``points`` (rows) in the given order from ``start``."""
    pts = np.asarray(points, float).reshape(len(points), -1)
    pos = np.asarray(start, float).reshape(-1)
    total = 0.0
    for p in pts:
        total += float(motion.move_time(pos, p[None, :])[0])
        pos = p
    return total


def order_route(points, start, motion, exact_max=10):
    """Order in which to visit ``points`` so the total motion time is minimal.

    Moves are asymmetric (backlash, preferred direction), so this is an
    asymmetric open-path travelling-salesman problem.  Up to ``exact_max``
    points it is solved exactly (dynamic programming); larger batches use a
    nearest-neighbour start improved by 2-opt / or-opt moves.

    Returns
    -------
    order : numpy.ndarray of int
    total : float
        Motion time of that route.
    """
    pts = np.asarray(points, float).reshape(len(points), -1)
    n = len(pts)
    if n <= 1:
        return np.arange(n), route_time(pts, start, motion) if n else 0.0
    start = np.asarray(start, float).reshape(-1)
    C = np.array([motion.move_time(p, pts) for p in pts])  # C[i, j]: i -> j
    c0 = motion.move_time(start, pts)
    if n <= exact_max:
        order = _held_karp(C, c0)
    else:
        order = _heuristic_route(C, c0, pts, start)
    total = c0[order[0]] + sum(C[a, b] for a, b in zip(order[:-1], order[1:]))
    return np.array(order), float(total)


def _held_karp(C, c0):
    n = len(c0)
    if n <= 3:
        best = min(
            permutations(range(n)),
            key=lambda p: c0[p[0]] + sum(C[a, b] for a, b in zip(p[:-1], p[1:])),
        )
        return list(best)
    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=int)
    for j in range(n):
        dp[1 << j, j] = c0[j]
    for mask in range(full):
        for j in range(n):
            if not (mask >> j) & 1 or not np.isfinite(dp[mask, j]):
                continue
            for k in range(n):
                if (mask >> k) & 1:
                    continue
                m2 = mask | (1 << k)
                v = dp[mask, j] + C[j, k]
                if v < dp[m2, k]:
                    dp[m2, k] = v
                    parent[m2, k] = j
    mask = full - 1
    j = int(np.argmin(dp[mask]))
    order = []
    while j >= 0:
        order.append(j)
        pj = parent[mask, j]
        mask ^= 1 << j
        j = pj
    return order[::-1]


def _heuristic_route(C, c0, pts, start):
    n = len(c0)

    def cost(o):
        return c0[o[0]] + sum(C[a, b] for a, b in zip(o[:-1], o[1:]))

    seeds = []
    left = set(range(n))
    cur = int(np.argmin(c0))
    nn = [cur]
    left.remove(cur)
    while left:
        nxt = min(left, key=lambda k: C[cur, k])
        nn.append(nxt)
        left.remove(nxt)
        cur = nxt
    seeds.append(nn)
    for d in range(pts.shape[1]):
        asc = sorted(range(n), key=lambda k: (pts[k, d], tuple(pts[k])))
        for o in (asc, asc[::-1]):
            seeds.append(o)
            ahead = [k for k in o if (pts[k, d] - start[d]) * (1 if o is asc else -1) >= 0]
            behind = [k for k in o if k not in ahead]
            seeds.append(ahead + behind)
    order = min(seeds, key=cost)
    best = cost(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = order[:i] + order[i : k + 1][::-1] + order[k + 1 :]
                c = cost(cand)
                if c < best - 1e-12:
                    order, best, improved = cand, c, True
            for k in range(n):
                if k == i:
                    continue
                o = order[:]
                p = o.pop(i)
                o.insert(k, p)
                c = cost(o)
                if c < best - 1e-12:
                    order, best, improved = o, c, True
    return order
