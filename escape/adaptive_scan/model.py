"""Impulsive-response signal model: step-like onset plus damped oscillations.

The model describes a 1-D pump-probe transient as a function of delay ``t``::

    y(t) = offset + sum_k  A_k * g_k(t) * exp(-(t - t0) / decay_k)
                               * cos(2 pi (t - t0) / period_k + phase_k)

for ``t > t0`` (and ``offset`` before time zero), with an optional grow-in
envelope per component::

    g_k(t) = 1 - exp(-(t - t0 - onset_k) / rise_k)   for t > t0 + onset_k, else 0

A component with ``period=np.inf`` does not oscillate (a step, or an
exponential rise/recovery if ``decay`` is finite).  Decay and phase are
referenced to time zero ``t0``, so a delayed ``onset`` does not shift the
oscillation phase.

Every term is a sum of ``H(u) * exp(s u)`` with complex ``s``, which allows an
exact, closed-form convolution with a Gaussian instrument response / timing
jitter of standard deviation ``sigma`` (:func:`smeared_exp_step`).  The same
building block is used by the generator in :mod:`escape.storage.test_data` and
by the fit in :mod:`escape.adaptive_scan.fit`.
"""

from dataclasses import dataclass, field, replace
from typing import List

import numpy as np
from scipy.special import erfcx

__all__ = ["Oscillation", "ImpulsiveResponse", "smeared_exp_step"]


def smeared_exp_step(u, s, sigma=0.0):
    """``H(u) * exp(s*u)`` convolved with a normalised Gaussian of width ``sigma``.

    Parameters
    ----------
    u : array-like
        Time relative to the step (real).
    s : complex
        Complex rate; ``Re(s) <= 0`` for a decaying term.
    sigma : float
        Gaussian standard deviation.  ``0`` returns the unsmeared step.

    Returns
    -------
    numpy.ndarray of complex
    """
    u = np.asarray(u, dtype=float)
    s = complex(s)
    if sigma <= 0:
        out = np.zeros(u.shape, dtype=complex)
        pos = u >= 0
        out[pos] = np.exp(s * u[pos])
        return out
    # exp(s u + s^2 sigma^2/2) * 0.5 erfc(z), z = -(u + s sigma^2)/(sigma sqrt 2),
    # rewritten via erfcx so that neither factor overflows.
    z = -(u + s * sigma**2) / (sigma * np.sqrt(2.0))
    gauss = np.exp(-(u**2) / (2.0 * sigma**2))
    out = np.empty(u.shape, dtype=complex)
    pos = z.real >= 0
    out[pos] = 0.5 * gauss[pos] * erfcx(z[pos])
    neg = ~pos
    out[neg] = np.exp(s * u[neg] + 0.5 * s**2 * sigma**2) - 0.5 * gauss[neg] * erfcx(
        -z[neg]
    )
    return out


def _component_terms(u, rate, freq, rise, onset, sigma):
    """Complex response ``g(u) exp(s u)`` (times ``exp(i*0)``) of one component.

    ``u`` is time after ``t0``; ``rate = 1/decay``; ``freq = 1/period``.
    Real part times ``a`` minus imaginary part times ``b`` gives
    ``A cos(2 pi f u + phase) ...`` with ``a + i b = A exp(i phase)``.
    """
    s = -rate + 2j * np.pi * freq
    shift = np.exp(s * onset) if onset else 1.0
    out = smeared_exp_step(u - onset, s, sigma)
    if rise > 0:
        out = out - smeared_exp_step(u - onset, s - 1.0 / rise, sigma)
    return shift * out


@dataclass
class Oscillation:
    """One component of an :class:`ImpulsiveResponse`.

    Parameters
    ----------
    amplitude : float
        Amplitude ``A`` (sign matters for ``period=inf``).
    period : float
        Oscillation period (same units as the delay).  ``np.inf`` for a
        non-oscillatory step / exponential component.
    decay : float
        Exponential decay time (lifetime) of the amplitude.  ``np.inf`` for
        no decay.
    rise : float
        Grow-in time constant of the amplitude envelope.  ``0`` means the
        component switches on instantaneously.
    onset : float
        Delay after time zero at which the grow-in starts.
    phase : float
        Phase in radians at time zero.
    """

    amplitude: float = 1.0
    period: float = np.inf
    decay: float = np.inf
    rise: float = 0.0
    onset: float = 0.0
    phase: float = 0.0

    @property
    def frequency(self):
        return 0.0 if np.isinf(self.period) else 1.0 / self.period

    @property
    def rate(self):
        return 0.0 if np.isinf(self.decay) else 1.0 / self.decay

    def evaluate(self, t, t0=0.0, sigma=0.0):
        """Component contribution at delays ``t`` (smeared by ``sigma``)."""
        c = self.amplitude * np.exp(1j * self.phase)
        terms = _component_terms(
            np.asarray(t, dtype=float) - t0,
            self.rate,
            self.frequency,
            self.rise,
            self.onset,
            sigma,
        )
        return (c * terms).real


@dataclass
class ImpulsiveResponse:
    """Sum of :class:`Oscillation` components switched on at ``t0``.

    Call it like a function: ``response(t)`` is the ideal signal,
    ``response(t, sigma=0.2)`` the signal smeared by a Gaussian timing jitter
    / instrument response of standard deviation ``sigma``.

    Examples
    --------
    The default reproduces the classic test signal
    ``1 - cos(2 pi t / 0.7) * exp(-t / 2)``:

    >>> resp = ImpulsiveResponse.default()
    >>> # two overlapping modes, the second growing in after 0.3 with 0.5 rise time
    >>> resp = ImpulsiveResponse([
    ...     Oscillation(1.0),                                   # step
    ...     Oscillation(-0.6, period=0.7, decay=2.0),
    ...     Oscillation(0.3, period=0.25, decay=1.0, rise=0.5, onset=0.3),
    ... ])
    """

    components: List[Oscillation] = field(default_factory=list)
    t0: float = 0.0
    offset: float = 0.0

    @classmethod
    def default(cls):
        """Unit step with one fully modulated, damped oscillation (period 0.7, decay 2)."""
        return cls(
            [Oscillation(1.0), Oscillation(-1.0, period=0.7, decay=2.0)],
        )

    def __call__(self, t, sigma=0.0):
        t = np.asarray(t, dtype=float)
        y = np.full(t.shape, float(self.offset))
        for comp in self.components:
            y = y + comp.evaluate(t, self.t0, sigma)
        return y

    def scaled(self, factor):
        """Copy with all component amplitudes multiplied by ``factor``."""
        return replace(
            self,
            components=[
                replace(c, amplitude=c.amplitude * factor) for c in self.components
            ],
        )

    @property
    def oscillations(self):
        """Components that actually oscillate (finite period)."""
        return [c for c in self.components if np.isfinite(c.period)]
