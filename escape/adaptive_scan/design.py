"""Model-based delay scans for impulsive signals: steps and damped oscillations.

* :class:`StepScan` -- the signal is a (jitter-smeared) step, optionally
  with an exponential recovery.  It searches for the step position (several
  candidate positions are kept until the data decide) and then measures
  where edge position, width and recovery are learnt fastest.
* :class:`OscillationScan` (alias :class:`AdaptiveScan`) -- step plus up to
  ``max_osc`` (typically 1-3) overlapping damped oscillations, each with its
  own decay time and, with ``fit_rise=True``, its own grow-in time.

Both fit the data after every batch (:func:`~escape.adaptive_scan.fit.fit_response`),
keep competing hypotheses (model orders, frequency aliases, step positions)
weighted by BIC, and value a candidate delay by the expected information it
adds about the *target* parameters::

    gain(t) = -1/2 log(1 - v^T Sigma_SS^-1 v / (noise^2 + j^T Sigma j)),
    j = d model / d params at t,  v = (Sigma j)_S

(D_s-optimal design on the linearised posterior ``Sigma``), plus a
model-discrimination term where hypotheses disagree.  Planning (information
per unit time, exposure choice, space-filling share, visiting order) is the
shared machinery of :mod:`escape.adaptive_scan.strategy`.

Note that this is *not* "more points where the signal is steep": the
sensitivity to time zero is the slope, but the sensitivity to a decay time
peaks at the oscillation extrema (zero slope), and a parameter that is
already well known stops attracting points however steep the signal is.
"""

import numpy as np

from .fit import fit_response
from .strategy import ModelScan, information_gain

__all__ = ["OscillationScan", "AdaptiveScan", "StepScan", "information_gain"]


class OscillationScan(ModelScan):
    """Adaptive scan for a step plus damped (optionally growing-in) oscillations.

    Parameters
    ----------
    t_range : (float, float)
        Delay range that may be measured.
    max_osc : int
        Highest number of oscillation components considered (0-3 typical).
    n_exp : int
        Number of non-oscillatory (step / exponential) components.
    sigma : float, optional
        Known Gaussian timing jitter / IRF width.  ``None`` fits it.
    fit_rise : bool
        Fit each oscillation's own grow-in time.
    recovery : bool
        Non-oscillatory components may recover exponentially (rate fitted);
        ``False`` = pure steps.
    f_max : float, optional
        Highest frequency searched.  Defaults to ``0.5 / candidate spacing``.
    targets : str or list of str
        Parameters the design should learn: ``"dynamics"`` (time zero,
        smearing, rates, frequencies, rise times), ``"frequencies"``,
        ``"all"``, or names / suffixes such as ``["freq", "rate"]``.
    t0_modes : bool
        Also search for competing time-zero solutions.
    explore, discriminate, grid_step, candidates, n_candidates, motion,
    exposure, n_initial, reorder
        See :class:`~escape.adaptive_scan.strategy.ScanStrategy` and
        :class:`~escape.adaptive_scan.strategy.ModelScan`.

    Examples
    --------
    >>> scan = OscillationScan((-1, 5), max_osc=2)
    >>> t = scan.initial_points(15)
    >>> scan.add(t, *measure(t))                 # (y, yerr) from the experiment
    >>> for _ in range(10):
    ...     plan = scan.plan(5)                  # ordered points + exposures
    ...     y, yerr = measure(plan.positions, plan.exposures)
    ...     scan.add(plan.positions, y, yerr, exposure=plan.exposures)
    >>> print(scan.summary())
    """

    def __init__(
        self,
        t_range,
        candidates=None,
        n_candidates=601,
        max_osc=2,
        n_exp=1,
        sigma=None,
        fit_rise=False,
        recovery=True,
        f_max=None,
        targets="dynamics",
        explore=0.3,
        discriminate=True,
        t0_modes=False,
        **kwargs,
    ):
        super().__init__(
            [t_range], candidates=candidates, n_candidates=n_candidates,
            targets=targets, explore=explore, discriminate=discriminate, **kwargs,
        )
        self.t_range = tuple(float(v) for v in self.bounds[0])
        self.max_osc = max_osc
        self.n_exp = n_exp
        self.sigma = sigma
        self.fit_rise = fit_rise
        self.recovery = recovery
        cand = self.candidates
        if f_max is None:
            f_max = 0.5 / np.median(np.diff(cand)) if len(cand) > 1 else 1.0
        self.f_max = f_max
        self.t0_modes = t0_modes

    def fit(self):
        """Refit all model orders; :attr:`results` is sorted best-BIC first."""
        if self.n_points < 4:
            self.results = []
            return self.results
        self.results = fit_response(
            self.t, self.y, self.yerr,
            max_osc=self.max_osc, n_exp=self.n_exp, sigma=self.sigma,
            fit_rise=self.fit_rise, t_range=self.t_range, f_max=self.f_max,
            start=self.results, t0_modes=self.t0_modes,
            fit_exp_rate=self.recovery,
        )
        return self.results

    def _target_index(self, result):
        tg = self.targets
        names = result.param_names
        if isinstance(tg, str) and tg == "dynamics":
            return np.arange(len(result.theta))
        if isinstance(tg, str) and tg == "frequencies":
            idx = [i for i, n in enumerate(names) if n.endswith("_freq")]
            return np.array(idx or range(len(result.theta)))
        return super()._target_index(result)

    def _relabel(self, result):
        return _relabel_by_frequency(result)

    def _derive(self, out):
        # periods / decay times from the *combined* freq / rate (averaging
        # 1/rate would be dominated by any hypothesis with a vanishing rate)
        for name in [n for n in out if n.endswith(("_period", "_decay"))]:
            del out[name]
        # only where the rate / frequency is resolved (> 2 sigma from zero);
        # an unresolved slow recovery would otherwise print as decay ~ 1e23
        for name in list(out):
            v, e = out[name]
            if v > 2 * e and name.endswith("_freq"):
                out[name[:-5] + "_period"] = (1.0 / v, e / v**2)
            if v > 2 * e and name.endswith("_rate"):
                out[name[:-5] + "_decay"] = (1.0 / v, e / v**2)
        return out

    def _describe(self, spec):
        return f"{spec.n_osc} oscillation(s)"

    # ------------------------------------------------------------------ plot
    def plot(self, axes=None, truth=None, truth_sigma=None):
        """Data, best fit with 1-sigma band, and the acquisition function.

        ``truth`` is an optional callable ``truth(t)`` (e.g. a simulator's
        ``truth``) drawn for comparison; ``truth_sigma`` is passed on as its
        ``sigma`` if given.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            _fig, axes = plt.subplots(
                2, 1, sharex=True, figsize=(8, 6), height_ratios=(3, 1)
            )
        ax, ax2 = axes
        ax.errorbar(self.t, self.y, self.yerr, fmt=".", color="k", ms=3, lw=0.8,
                    label="measured")
        tt = self.candidates
        if truth is not None:
            yt = truth(tt) if truth_sigma is None else truth(tt, sigma=truth_sigma)
            ax.plot(tt, yt, color="C2", lw=1, label="truth")
        if self.best is not None:
            mu, sd = self.best.predict(tt), self.best.predict_std(tt)
            ax.plot(tt, mu, color="C0", label=f"fit ({self.best.spec.n_osc} osc.)")
            ax.fill_between(tt, mu - sd, mu + sd, color="C0", alpha=0.25, lw=0)
            score, parts = self.acquisition()
            ax2.plot(tt, parts["gain"], label="information gain")
            if self.discriminate:
                ax2.plot(tt, parts["discrimination"], label="model discrimination")
            ax2.plot(tt, score, color="k", lw=0.8, label="total score")
            ax2.legend(fontsize="small")
        ax.legend(fontsize="small")
        ax2.set_xlabel("delay")
        ax.set_ylabel("signal")
        ax2.set_ylabel("score")
        return axes


AdaptiveScan = OscillationScan


class StepScan(OscillationScan):
    """Adaptive scan for a step (edge) of unknown position, width and height.

    The model is an error-function step (a step smeared by a Gaussian of
    width ``sigma``, e.g. timing jitter or a beam size), with an optional
    exponential recovery after the step (``recovery=True``).  While the step
    position is uncertain, the distinct candidate positions from a grid
    search are kept as competing hypotheses and the scan measures where they
    disagree -- a noise-aware bisection.  Once the edge is found, points go
    to the flanks of the edge (position and width) and, with recovery, to
    the decay.

    Parameters are those of :class:`OscillationScan` without the oscillation
    options; ``targets`` defaults to ``"dynamics"`` (``t0``, ``sigma`` and
    the recovery rate).
    """

    def __init__(self, t_range, recovery=False, **kwargs):
        kwargs.setdefault("max_osc", 0)
        kwargs.setdefault("explore", 0.2)
        super().__init__(t_range, recovery=recovery, **kwargs)

    def _describe(self, spec):
        return "step" + (" with recovery" if spec.fit_exp_rate else "")


def _relabel_by_frequency(result):
    """``result.errors()`` with oscillation labels sorted by frequency."""
    e = result.errors()
    n = result.spec.n_osc
    freqs = [e[f"osc{k}_freq"][0] for k in range(n)]
    order = np.argsort(freqs)
    out = {}
    for name, val in e.items():
        if name.startswith("osc"):
            k, rest = name[3:].split("_", 1)
            name = f"osc{int(np.where(order == int(k))[0][0])}_{rest}"
        out[name] = val
    # list oscillations in label order (slowest first)
    return dict(sorted(out.items(), key=lambda kv: (
        kv[0].startswith("osc"), kv[0][:4] if kv[0].startswith("osc") else "",
    )))
