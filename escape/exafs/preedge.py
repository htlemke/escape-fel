"""
Pre-edge subtraction and post-edge (edge-step) normalization.

Every raw XAS spectrum mu(E) sits on top of a smooth, slowly-varying
"pre-edge" background (absorption from other elements/shells, sample
holder, detector response, etc.) and has an overall step height at
the edge that depends on sample thickness and concentration. Before
any EXAFS analysis you need to:

  1. fit and subtract a smooth line/polynomial through the pre-edge
     region (well below E0),
  2. fit a smooth polynomial through the post-edge region and
     evaluate it at E0 to get the "edge step",
  3. divide by the edge step so mu is on a scale where it goes from
     ~0 (pre-edge) to ~1 (normalized absorption, far above the edge).

This mirrors the standard approach used by Athena/Larch's pre_edge().
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PreEdgeResult:
    energy: np.ndarray
    mu: np.ndarray
    e0: float
    pre_edge: np.ndarray      # fitted pre-edge line, evaluated on `energy`
    post_edge: np.ndarray     # fitted post-edge polynomial, evaluated on `energy`
    edge_step: float          # post_edge(E0) - pre_edge(E0)
    norm: np.ndarray          # (mu - pre_edge) / edge_step
    flat: np.ndarray          # "flattened": norm with post-edge curvature removed


def _fit_poly(x, y, order):
    """Least-squares polynomial fit, returns numpy poly1d."""
    coeffs = np.polyfit(x, y, order)
    return np.poly1d(coeffs)


def pre_edge(energy, mu, e0=None,
             pre_range=(-150, -30),
             post_range=(150, None),
             post_order=2):
    """
    Subtract a pre-edge line and normalize by the post-edge edge-step.

    Parameters
    ----------
    energy, mu : array_like
        Raw energy (eV) and absorption mu(E).
    e0 : float, optional
        Edge energy. If None, estimated automatically with
        :func:`escape.exafs.energy_k.find_e0`.
    pre_range : (float, float)
        Energy window *relative to e0* used to fit the pre-edge line,
        e.g. (-150, -30) means "fit a line to the data between
        E0-150 eV and E0-30 eV". A straight line (order 1) is used,
        which is almost always sufficient and avoids over-fitting.
    post_range : (float, float or None)
        Energy window relative to e0 used to fit the smooth post-edge
        polynomial, e.g. (150, None) means "from E0+150 eV to the end
        of the scan".
    post_order : int
        Polynomial order for the post-edge fit (2 or 3 is typical).

    Returns
    -------
    PreEdgeResult
    """
    from .energy_k import find_e0

    energy = np.asarray(energy, dtype=float)
    mu = np.asarray(mu, dtype=float)

    if e0 is None:
        e0 = find_e0(energy, mu)

    # --- pre-edge line ---
    pre_lo, pre_hi = e0 + pre_range[0], e0 + pre_range[1]
    pre_mask = (energy >= pre_lo) & (energy <= pre_hi)
    if pre_mask.sum() < 2:
        raise ValueError(
            "Not enough points in the pre-edge range %s (relative to E0=%.1f) "
            "to fit a line -- widen pre_range or check E0." % (pre_range, e0)
        )
    pre_line = _fit_poly(energy[pre_mask], mu[pre_mask], 1)

    # --- post-edge polynomial ---
    post_lo = e0 + post_range[0]
    post_hi = energy.max() if post_range[1] is None else e0 + post_range[1]
    post_mask = (energy >= post_lo) & (energy <= post_hi)
    if post_mask.sum() < post_order + 1:
        raise ValueError(
            "Not enough points in the post-edge range %s (relative to E0=%.1f) "
            "to fit a degree-%d polynomial -- widen post_range or lower post_order."
            % (post_range, e0, post_order)
        )
    post_poly = _fit_poly(energy[post_mask], mu[post_mask], post_order)

    pre_edge_curve = pre_line(energy)
    post_edge_curve = post_poly(energy)

    edge_step = float(post_poly(e0) - pre_line(e0))
    if edge_step <= 0:
        raise ValueError(
            "Computed a non-positive edge step (%.4g). Check that mu(E) "
            "actually increases across the edge, or adjust pre_range/post_range."
            % edge_step
        )

    norm = (mu - pre_edge_curve) / edge_step

    # "Flattened" spectrum: remove the post-edge polynomial's curvature
    # above E0 so the normalized spectrum is flat (~1) far above the
    # edge, which is nice for plotting/comparing XANES regions. This
    # does NOT affect the EXAFS extraction, which uses `norm`/mu, not
    # `flat`.
    flat = norm.copy()
    above = energy >= e0
    flat_poly_curve = (post_edge_curve - pre_edge_curve) / edge_step
    flat[above] = norm[above] - flat_poly_curve[above] + 1.0

    return PreEdgeResult(
        energy=energy, mu=mu, e0=float(e0),
        pre_edge=pre_edge_curve, post_edge=post_edge_curve,
        edge_step=edge_step, norm=norm, flat=flat,
    )
