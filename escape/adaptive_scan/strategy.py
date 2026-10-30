"""Common machinery of all scan strategies: data, grid, cost-aware planning.

Every strategy answers one question: *given what has been measured, where
(and with which exposure) should the next points be measured, and in which
order?*  The strategies differ only in how they value a candidate point
(:meth:`ScanStrategy._utility`); planning is shared:

1. For each candidate grid position ``x`` and each allowed exposure ``e``,
   the strategy's expected value ``U(x, e)`` (information, or reduction of a
   model-free loss) is divided by the time it costs::

       rate(x, e) = U(x, e) / (move_time(current -> x) + e + acquisition_overhead)

   so the plan maximises information *per unit time*: it prefers nearby
   points, points in the preferred approach direction (no backlash
   correction), and the exposure that buys most per second.
2. The best point is committed (the strategy updates its expectations as if
   it had been measured) and the next one is chosen from there, up to the
   requested batch size.  A fixed share ``explore`` of the points goes to
   space-filling positions instead (farthest from everything measured), which
   protects every strategy against a wrong picture of the signal.
3. The batch is put into the fastest visiting order (:func:`order_route`).

Positions come from a candidate grid -- by default a fine linear grid, or the
multiples of ``grid_step`` (the :func:`escape.utilities.roundto` grid).
Measuring a grid position again is allowed and adds statistics.
"""

from dataclasses import dataclass

import numpy as np

from .costs import ExposureModel, Motion, grid_points, order_route

__all__ = ["Plan", "ScanStrategy", "ModelScan", "UniformScan", "information_gain"]


@dataclass
class Plan:
    """An ordered batch of points to measure.

    Iterating yields ``(position, exposure)`` pairs in measurement order.
    """

    positions: np.ndarray
    exposures: np.ndarray
    move_times: np.ndarray
    utilities: np.ndarray
    acquisition_overhead: float = 0.0

    def __len__(self):
        return len(self.exposures)

    def __iter__(self):
        return iter(zip(self.positions, self.exposures))

    @property
    def total_time(self):
        """Motion + exposure + acquisition overhead of the whole batch."""
        return float(
            self.move_times.sum()
            + self.exposures.sum()
            + self.acquisition_overhead * len(self)
        )


def information_gain(J, cov, noise, target_idx):
    """Expected information (nats) on the target parameters per candidate row of ``J``.

    ``noise`` is the 1-sigma error of the new measurement (scalar, or one
    value per row).  For several noise levels at once use
    :func:`_gain_terms` + :func:`_gain`.
    """
    s, qn = _gain_terms(J, cov, target_idx)
    noise = np.broadcast_to(np.asarray(noise, float), s.shape)
    return _gain(s, qn, noise**2)


def _gain_terms(J, cov, target_idx):
    G = J @ cov
    s = np.einsum("ij,ij->i", G, J)
    V = G[:, target_idx]
    Sinv = np.linalg.pinv(cov[np.ix_(target_idx, target_idx)], hermitian=True)
    qn = np.einsum("ij,jk,ik->i", V, Sinv, V)
    return s, qn


def _gain(s, qn, noise2):
    return -0.5 * np.log(np.clip(1.0 - qn / (noise2 + s), 1e-12, 1.0))


class ScanStrategy:
    """Base class: data bookkeeping, candidate grid and cost-aware planning.

    Parameters
    ----------
    bounds : (lo, hi) or sequence of (lo, hi)
        Scan range (one pair per dimension).
    grid_step : float or sequence of float, optional
        Measure only on multiples of this step (per dimension), i.e. the
        :func:`escape.utilities.roundto` grid within ``bounds``.
    candidates : array-like, optional
        Explicit candidate positions (``(n,)`` for 1-D, ``(n, d)``).
    n_candidates : int
        Points per dimension of the default (fine, linear) candidate grid.
    motion : Axis, list of Axis or Motion, optional
        Positioning-time model; default: moves cost nothing.
    exposure : ExposureModel, optional
        Allowed exposures and noise model; default: one fixed exposure.
    explore : float
        Fraction of points placed space-filling.
    n_initial : int
        Size of the initial survey.
    reorder : bool or None
        Put each batch into the fastest visiting order.  ``None`` = only when
        a ``motion`` model was given.
    """

    ndim = 1

    def __init__(
        self,
        bounds,
        grid_step=None,
        candidates=None,
        n_candidates=601,
        motion=None,
        exposure=None,
        explore=0.0,
        n_initial=15,
        reorder=None,
    ):
        bounds = np.asarray(bounds, float).reshape(-1, 2)
        self.ndim = len(bounds)
        self.bounds = bounds
        if candidates is None:
            steps = np.broadcast_to(
                np.asarray(grid_step if grid_step is not None else np.nan, float),
                (self.ndim,),
            )
            axes = []
            for (lo, hi), st in zip(bounds, steps):
                if np.isfinite(st):
                    axes.append(grid_points(lo, hi, st))
                else:
                    axes.append(np.linspace(lo, hi, n_candidates))
            mesh = np.meshgrid(*axes, indexing="ij")
            candidates = np.column_stack([m.ravel() for m in mesh])
        candidates = np.asarray(candidates, float).reshape(-1, self.ndim)
        self._C = np.unique(candidates, axis=0)
        self.grid_step = grid_step
        self._motion_given = motion is not None
        self.motion = Motion.coerce(motion, self.ndim)
        self.exposure_model = exposure if exposure is not None else ExposureModel()
        self.explore = explore
        self._explore_credit = 0.0
        self.n_initial = n_initial
        self.reorder = reorder
        self._X = np.zeros((0, self.ndim))
        self.y = np.zeros(0)
        self.yerr = np.zeros(0)
        self.exposure = np.zeros(0)
        self.history = []
        self.position = None

    # ----------------------------------------------------------- properties
    @property
    def candidates(self):
        return self._C[:, 0] if self.ndim == 1 else self._C

    @property
    def x(self):
        """Measured positions (``(n,)`` for 1-D scans, ``(n, d)`` otherwise)."""
        return self._X[:, 0] if self.ndim == 1 else self._X

    @property
    def t(self):
        return self.x

    @property
    def n_points(self):
        return len(self.y)

    @property
    def span(self):
        return self.bounds[:, 1] - self.bounds[:, 0]

    # ----------------------------------------------------------------- data
    def initial_points(self, n=None):
        """Space-filling survey snapped to the candidate grid."""
        n = self.n_initial if n is None else n
        if self.ndim == 1:
            grid = np.linspace(*self.bounds[0], n)[:, None]
        else:
            k = int(np.ceil(n ** (1.0 / self.ndim)))
            axes = [np.linspace(lo, hi, k) for lo, hi in self.bounds]
            mesh = np.meshgrid(*axes, indexing="ij")
            grid = np.column_stack([m.ravel() for m in mesh])
        idx = self._nearest(grid)
        _, keep = np.unique(idx, return_index=True)
        pts = self._C[idx[np.sort(keep)]]
        return pts[:, 0] if self.ndim == 1 else pts

    def _nearest(self, X):
        X = np.asarray(X, float).reshape(-1, self.ndim)
        d = (((self._C[None, :, :] - X[:, None, :]) / self.span) ** 2).sum(-1)
        return d.argmin(1)

    def add(self, x, y, yerr, exposure=None, refit=True):
        """Append measured points (value and 1-sigma error per position)."""
        X = np.asarray(x, float).reshape(-1, self.ndim)
        y = np.atleast_1d(np.asarray(y, float))
        yerr = np.broadcast_to(np.asarray(yerr, float), y.shape)
        if exposure is None:
            exposure = self.exposure_model.base
        exposure = np.broadcast_to(np.asarray(exposure, float), y.shape)
        ok = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        self._X = np.vstack([self._X, X[ok]])
        self.y = np.concatenate([self.y, y[ok]])
        self.yerr = np.concatenate([self.yerr, yerr[ok]])
        self.exposure = np.concatenate([self.exposure, exposure[ok]])
        if len(X):
            self.position = X[-1].copy()
        self.exposure_model.calibrate(self.yerr, self.exposure)
        self._after_add(refit)
        self.history.append((X[:, 0] if self.ndim == 1 else X, self._snapshot()))
        return self

    def _after_add(self, refit):
        pass

    def _snapshot(self):
        return None

    # ------------------------------------------------------------- planning
    def _ready(self):
        return self.n_points >= 3

    def _begin_plan(self, noise2):
        pass

    def _utility(self, noise2, picks):
        """Expected value of measuring each candidate, shape ``(n_cand, n_exposures)``."""
        raise NotImplementedError

    def _commit(self, idx, noise2):
        pass

    def _end_plan(self):
        pass

    def _fill_distance(self, picks):
        meas = self._X
        if picks:
            meas = np.vstack([meas, self._C[picks]])
        if not len(meas):
            return np.ones(len(self._C))
        d2 = np.full(len(self._C), np.inf)
        for m in meas:
            d2 = np.minimum(d2, (((self._C - m) / self.span) ** 2).sum(1))
        return np.sqrt(d2)

    def plan(self, n=1, position=None, reorder=None):
        """Next ``n`` points (and exposures), in the order to measure them.

        Parameters
        ----------
        n : int
            Batch size.
        position : float or array-like, optional
            Current position of the scan axes (default: last measured point).
        reorder : bool, optional
            Override the ``reorder`` setting for this batch.

        Returns
        -------
        Plan
        """
        if position is None:
            position = self.position
        if position is None:
            position = self.bounds[:, 0]
        position = np.asarray(position, float).reshape(self.ndim)
        em = self.exposure_model
        options = em.options
        base_j = int(np.argmin(np.abs(options - em.base)))
        if not self._ready():
            pts = np.asarray(self.initial_points(n)).reshape(-1, self.ndim)
            return self._finish(
                pts, np.full(len(pts), em.base), np.full(len(pts), np.nan), position,
                reorder,
            )
        noise2 = em.noise(options) ** 2
        picks, exps, utils = [], [], []
        pos = position
        self._begin_plan(noise2)
        try:
            for _ in range(n):
                self._explore_credit += self.explore
                if self._explore_credit >= 1.0:
                    self._explore_credit -= 1.0
                    i, j, u = int(np.argmax(self._fill_distance(picks))), base_j, np.nan
                else:
                    U = np.asarray(self._utility(noise2, picks), float)
                    T = (
                        options[None, :]
                        + em.overhead
                        + self.motion.move_time(pos, self._C)[:, None]
                    )
                    R = U / np.maximum(T, 1e-12)
                    i, j = np.unravel_index(int(np.nanargmax(R)), R.shape)
                    u = U[i, j]
                self._commit(int(i), noise2[j])
                picks.append(int(i))
                exps.append(options[j])
                utils.append(u)
                pos = self._C[i]
        finally:
            self._end_plan()
        return self._finish(
            self._C[picks], np.array(exps), np.array(utils), position, reorder
        )

    def _finish(self, pts, exps, utils, position, reorder):
        reorder = self.reorder if reorder is None else reorder
        if reorder is None:
            reorder = self._motion_given
        if reorder and len(pts) > 1:
            order, _ = order_route(pts, position, self.motion)
            pts, exps, utils = pts[order], exps[order], utils[order]
        moves, pos = [], position
        for p in pts:
            moves.append(float(self.motion.move_time(pos, p[None, :])[0]))
            pos = p
        out = pts[:, 0] if self.ndim == 1 else pts
        return Plan(
            out, np.asarray(exps, float), np.array(moves), np.asarray(utils, float),
            self.exposure_model.overhead,
        )

    def suggest(self, n=1, noise=None):
        """Positions of :meth:`plan` (exposures ignored).

        ``noise`` sets the expected error of a point at the base exposure.
        """
        if noise is not None:
            self.exposure_model.set_noise(noise)
        return self.plan(n).positions

    def next(self, position=None):
        """Single next ``(position, exposure)``."""
        p = self.plan(1, position=position)
        return p.positions[0], p.exposures[0]


class UniformScan(ScanStrategy):
    """Baseline: sweep the grid uniformly, coarse to fine, in the preferred direction.

    Each pass measures the grid positions of the next finer subdivision
    (every 2^k-th point), so stopping at any time leaves a roughly uniform
    coverage; passes run in the motion's preferred direction.
    """

    def __init__(self, bounds, **kwargs):
        kwargs.setdefault("reorder", False)
        super().__init__(bounds, **kwargs)
        n = len(self._C)
        level = np.full(n, 0)
        stride = 1
        while stride < n:
            stride *= 2
        lvl = 0
        order = []
        seen = np.zeros(n, bool)
        while stride >= 1:
            idx = np.arange(0, n, stride)
            new = idx[~seen[idx]]
            seen[new] = True
            level[new] = lvl
            order.extend(new.tolist())
            stride //= 2
            lvl += 1
        if self.ndim == 1 and self.motion.axes[0].direction < 0:
            order = sorted(order, key=lambda k: (level[k], -k))
        self._order = order
        self._next = 0

    def _ready(self):
        return True

    def plan(self, n=1, position=None, reorder=None):
        idx = [self._order[(self._next + k) % len(self._order)] for k in range(n)]
        self._next += n
        pts = self._C[idx]
        e = self.exposure_model.base
        if position is None:
            position = self.position if self.position is not None else self.bounds[:, 0]
        return self._finish(
            pts, np.full(n, e), np.full(n, np.nan),
            np.asarray(position, float).reshape(self.ndim), reorder,
        )


class ModelScan(ScanStrategy):
    """Model-based design: fit, then maximise expected information per time.

    Subclasses implement :meth:`fit` (returning hypotheses -- fitted models
    with a ``cov``, ``jacobian(x)``, ``predict(x)``, ``bic``,
    ``param_names`` and ``errors()``) and may override :meth:`_target_index`.

    The value of a candidate is the BIC-weighted expected information gain
    on the *target* parameters (D_s-optimal criterion, linearised posterior)
    plus a model-discrimination term between competing hypotheses.
    """

    def __init__(self, bounds, targets="nonlinear", discriminate=True,
                 explore=0.3, **kwargs):
        super().__init__(bounds, explore=explore, **kwargs)
        self.targets = targets
        self.discriminate = discriminate
        self.results = []

    # ------------------------------------------------------------------ fit
    def fit(self):
        raise NotImplementedError

    def _after_add(self, refit):
        if refit:
            self.fit()

    def _snapshot(self):
        return list(self.results)

    def _ready(self):
        return bool(self.results)

    def _model_x(self, C):
        return C[:, 0] if self.ndim == 1 else C

    @property
    def best(self):
        return self.results[0] if self.results else None

    def model_weights(self, min_weight=1e-3):
        """BIC weights of the hypotheses (same order as :attr:`results`)."""
        if not self.results:
            return np.zeros(0)
        bic = np.array([r.bic for r in self.results])
        w = np.exp(-0.5 * (bic - bic.min()))
        w /= w.sum()
        w[w < min_weight] = 0.0
        return w / w.sum()

    def _target_index(self, result):
        names = result.param_names
        tg = self.targets
        if isinstance(tg, str):
            if tg == "all":
                return np.arange(len(names))
            if tg == "nonlinear":
                return np.arange(len(result.theta))
            tg = [tg]
        idx = [
            i for i, n in enumerate(names)
            for t in tg if n == t or n.endswith("_" + t)
        ]
        if not idx:
            raise ValueError(f"no parameters match targets={tg!r}; have {names}")
        return np.array(sorted(set(idx)))

    # ------------------------------------------------------------- planning
    def _begin_plan(self, noise2):
        Xm = self._model_x(self._C)
        w = self.model_weights()
        self._active = [(wi, r) for wi, r in zip(w, self.results) if wi > 0]
        self._saved_cov = [r.cov for _w, r in self._active]
        self._J = [r.jacobian(Xm) for _w, r in self._active]
        self._mu = [r.predict(Xm) for _w, r in self._active]
        self._tidx = [self._target_index(r) for _w, r in self._active]
        self._repeats = np.zeros(len(self._C))

    def _var(self, k):
        J, cov = self._J[k], self._active[k][1].cov
        return np.maximum(np.einsum("ij,jk,ik->i", J, cov, J), 0.0)

    def _utility(self, noise2, picks):
        n_c, n_e = len(self._C), len(noise2)
        U = np.zeros((n_c, n_e))
        if not self._active:
            return U
        best = self.results[0]
        b = next((k for k, (_w, r) in enumerate(self._active) if r is best), 0)
        var_b = self._var(b)
        rep = self._repeats[:, None]
        for k, (w, r) in enumerate(self._active):
            s, qn = _gain_terms(self._J[k], r.cov, self._tidx[k])
            U += w * _gain(s[:, None], qn[:, None], noise2[None, :])
            if self.discriminate and k != b:
                d2 = (self._mu[k] - self._mu[b]) ** 2
                x = d2[:, None] / (noise2[None, :] + (var_b + self._var(k))[:, None])
                # marginal value of one more measurement at a delay already in the batch
                U += w * 0.5 * np.log((1 + (rep + 1) * x) / (1 + rep * x))
        return U

    def _commit(self, idx, noise2):
        self._repeats[idx] += 1
        for k, (_w, r) in enumerate(self._active):
            j = self._J[k][idx]
            g = r.cov @ j
            r.cov = r.cov - np.outer(g, g) / (noise2 + j @ g)

    def _end_plan(self):
        for (_w, r), c in zip(self._active, self._saved_cov):
            r.cov = c

    def acquisition(self, noise=None, extra_t=()):
        """Expected information per candidate at the base exposure (no time costs).

        Returns ``(score, parts)`` with ``parts["gain"]``,
        ``parts["discrimination"]`` and ``parts["explore"]`` (distance to the
        nearest measured position).
        """
        em = self.exposure_model
        n2 = np.array([(noise if noise is not None else em.noise(em.base)) ** 2])
        extra = list(self._nearest(np.asarray(extra_t, float).reshape(-1, self.ndim)))
        dist = self._fill_distance(extra)
        if not self.results:
            return dist, {"gain": 0 * dist, "discrimination": 0 * dist, "explore": dist}
        disc_flag = self.discriminate
        self._begin_plan(n2)
        try:
            for i in extra:
                self._commit(i, n2[0])
            self.discriminate = False
            gain = self._utility(n2, extra)[:, 0]
            self.discriminate = disc_flag
            total = self._utility(n2, extra)[:, 0]
        finally:
            self.discriminate = disc_flag
            self._end_plan()
        return total, {"gain": gain, "discrimination": total - gain, "explore": dist}

    # ------------------------------------------------------------ estimates
    def _relabel(self, result):
        return result.errors()

    def _derive(self, out):
        return out

    def estimates(self):
        """Parameter estimates combined over the competing hypotheses.

        A single fit's (linearised) error ignores that the data may allow
        several distinct solutions.  All hypotheses *of the best model
        order* are combined with their BIC weights into a mixture mean and
        standard deviation (law of total variance).

        Returns
        -------
        dict
            ``name -> (mean, std)``.
        """
        if not self.results:
            return {}
        spec = self.results[0].spec
        w = self.model_weights(min_weight=0.0)
        sel = [(wi, r) for wi, r in zip(w, self.results) if r.spec == spec]
        total = sum(wi for wi, _r in sel)
        per_mode = [(wi / total, self._relabel(r)) for wi, r in sel]
        out = {}
        for name in per_mode[0][1]:
            vals = np.array([e[name] for _w, e in per_mode if name in e])
            ws = np.array([wi for wi, e in per_mode if name in e])
            ws = ws / ws.sum()
            mean = np.sum(ws * vals[:, 0])
            var = np.sum(ws * (vals[:, 1] ** 2 + vals[:, 0] ** 2)) - mean**2
            out[name] = (float(mean), float(np.sqrt(max(var, 0.0))))
        return self._derive(out)

    def _describe(self, spec):
        return str(spec)

    def summary(self):
        """Text table of :meth:`estimates` (best model order, all hypotheses)."""
        if not self.results:
            return "no fit yet"
        spec = self.results[0].spec
        n_modes = sum(r.spec == spec for r in self.results)
        lines = [
            f"{self._describe(spec)}, {n_modes} competing solution(s), "
            f"N={self.n_points} points"
        ]
        for n, (v, e) in self.estimates().items():
            lines.append(f"  {n:>14s} = {v: .5g} +/- {e:.2g}")
        return "\n".join(lines)
