"""
Background (mu0) removal to extract the EXAFS chi(k) function.

chi(k) is defined as

    chi(k) = ( mu(k) - mu0(k) ) / edge_step

where mu0(k) is a hypothetical "bare atom" background absorption --
i.e. what mu(E) would look like if there were no neighboring atoms to
scatter the photoelectron. mu0 is not measured directly; it has to be
estimated from the data itself, which is the classic hard problem in
EXAFS reduction.

This module implements a simplified version of the **AUTOBK** algorithm
(M. Newville, P. Livins, Y. Yacoby, J. J. Rehr & E. A. Stern,
*Phys. Rev. B* **47**, 14126 (1993), doi:10.1103/PhysRevB.47.14126),
the same idea used by IFEFFIT/Larch/Athena
(M. Newville, *J. Synchrotron Rad.* **8**, 322 (2001);
https://xraypy.github.io/xraylarch/):

  1. Represent mu0(k) as a cubic spline with knots evenly spaced in k.
     The knot spacing is tied to a physical length scale `rbkg`
     (Angstroms): a spline with knots spaced `~pi/(2*rbkg)` apart in k
     is, by construction, too "stiff" to reproduce EXAFS oscillations
     coming from atoms farther than `rbkg` away -- so fitting such a
     spline through the data removes the smooth background while
     leaving the genuine EXAFS wiggles behind.
  2. Instead of just fitting the spline to mu(E) directly (which tends
     to also eat some low-R EXAFS signal), the knot heights are
     refined by minimizing the Fourier-transformed chi(R) signal
     below `rbkg` -- i.e. we ask the background to be "as smooth as
     possible in R-space below rbkg", which is exactly the physical
     statement "there is no real atomic shell closer than rbkg".

This is a simplified, educational re-implementation -- it captures the
essential physics and gives results very close to Larch/Ifeffit for
typical data, but for publication-quality analysis of difficult data
sets you may still want to cross-check with Athena/Larch.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .energy_k import energy_to_k
from .fourier import ft_windowed


@dataclass
class BackgroundResult:
    energy: np.ndarray
    mu: np.ndarray
    e0: float
    edge_step: float
    k: np.ndarray            # k-grid (from the raw energy points > e0)
    mu0: np.ndarray          # background mu0, evaluated on `k`
    chi: np.ndarray          # extracted EXAFS chi(k)
    rbkg: float
    knot_k: np.ndarray
    knot_mu: np.ndarray


def _make_knots(kmin, kmax, rbkg):
    """Evenly spaced knots in k, spacing set by the rbkg length scale."""
    spacing = np.pi / (2.0 * rbkg)
    n_knots = max(int(np.ceil((kmax - kmin) / spacing)) + 1, 4)
    return np.linspace(kmin, kmax, n_knots)


def autobk(energy, mu, e0, edge_step, rbkg=1.0, kmin=0.0, kmax=None,
           kweight=1, window="hanning", dk=1.0, refine=True):
    """
    Extract chi(k) from mu(E) using a spline background constrained
    by a minimum-bond-distance assumption `rbkg`.

    Parameters
    ----------
    energy, mu : array_like
        Full measured spectrum.
    e0 : float
        Edge energy (eV), e.g. from :func:`escape.exafs.preedge.pre_edge`.
    edge_step : float
        Edge-step normalization, e.g. from `pre_edge`.
    rbkg : float
        Assumed minimum physical distance (Angstrom) to the nearest
        real atomic shell. Everything in chi(R) below `rbkg` is
        treated as unphysical background and suppressed. Typical
        values are 0.8-1.2 Ang for most materials (must be less than
        the true nearest-neighbor distance!).
    kmin, kmax : float
        k-range (Ang^-1) over which the background is fit / EXAFS is
        extracted. kmax defaults to the k corresponding to the last
        energy point.
    kweight : int
        k-weighting used *only* for the internal R-space minimization
        used to refine the background (not applied to the returned
        chi(k), which is always k^0 / un-weighted).
    window : str
        FT window function used for the internal minimization, see
        :mod:`escape.exafs.fourier`.
    dk : float
        Taper width (Ang^-1) for the FT window used in the internal
        minimization.
    refine : bool
        If True (default), refine the initial spline fit by directly
        minimizing the low-R chi(R) amplitude (the full AUTOBK idea).
        If False, just use the initial least-squares spline fit
        (faster, slightly less accurate background).

    Returns
    -------
    BackgroundResult
    """
    energy = np.asarray(energy, dtype=float)
    mu = np.asarray(mu, dtype=float)

    mask = energy >= e0 - 1e-6
    e_above = energy[mask]
    mu_above = mu[mask]
    k_raw = energy_to_k(e_above, e0)

    if kmax is None:
        kmax = k_raw.max()
    fit_mask = (k_raw >= kmin) & (k_raw <= kmax)
    k_fit = k_raw[fit_mask]
    mu_fit = mu_above[fit_mask]

    knot_k = _make_knots(k_fit.min(), k_fit.max(), rbkg)

    # Initial guess for knot heights: least-squares smoothing spline
    # through the data at those knot locations.
    from scipy.interpolate import LSQUnivariateSpline
    interior_knots = knot_k[1:-1]
    lsq_spline = LSQUnivariateSpline(k_fit, mu_fit, interior_knots, k=3)
    knot_mu0 = lsq_spline(knot_k)

    def background_from_knots(knot_y):
        cs = CubicSpline(knot_k, knot_y, bc_type="natural")
        return cs

    def chi_from_knots(knot_y):
        cs = background_from_knots(knot_y)
        mu0_fit = cs(k_fit)
        return (mu_fit - mu0_fit) / edge_step

    if refine:
        def objective(knot_y):
            chi = chi_from_knots(knot_y)
            r, chir, _ = ft_windowed(k_fit, chi, kweight=kweight,
                                      window=window, dk=dk, kmin=kmin, kmax=kmax)
            low_r = r < rbkg
            # Sum of squared magnitude below rbkg: the thing that
            # should vanish if mu0 is a good "no real neighbors"
            # background.
            return float(np.sum(np.abs(chir[low_r]) ** 2))

        res = minimize(objective, knot_mu0, method="Nelder-Mead",
                        options=dict(maxiter=200 * len(knot_mu0),
                                     xatol=1e-6, fatol=1e-10, adaptive=True))
        knot_mu_final = res.x
    else:
        knot_mu_final = knot_mu0

    cs_final = background_from_knots(knot_mu_final)
    mu0_fit = cs_final(k_fit)
    chi_fit = (mu_fit - mu0_fit) / edge_step

    return BackgroundResult(
        energy=e_above[fit_mask], mu=mu_fit, e0=float(e0), edge_step=float(edge_step),
        k=k_fit, mu0=mu0_fit, chi=chi_fit, rbkg=float(rbkg),
        knot_k=knot_k, knot_mu=knot_mu_final,
    )
