"""
Energy <-> photoelectron wavenumber (k) conversion.

A note on naming ("k" vs "q")
------------------------------
In modern EXAFS literature (and in this package) the photoelectron
wavenumber is almost universally called **k**, with units of
inverse Angstroms (Ang^-1). This is what you compute from the
measured energy E and the absorption-edge energy E0:

    k = sqrt(ETOK * (E - E0))

Some older papers (going back to Sayers, Stern and Lytle in the
1970s) used the symbol **q** for this same quantity, sometimes
reserving "k" for the *photoelectron momentum before an empirical
phase-shift correction* and "q" for the corrected wavenumber, i.e.
q(k) = k - phase-shift-derived correction. You will also see "q"
used (unrelated to EXAFS) as the momentum-transfer variable in
diffraction/scattering. To avoid ambiguity, this module:

  * uses ``k`` as the canonical name everywhere,
  * provides ``energy_to_q`` / ``q_to_energy`` as plain aliases of
    the k-conversion functions, so code or notebooks written with
    the "q" convention still work,
  * does NOT apply any phase-shift correction automatically -- if
    you need a phase-corrected axis, do it explicitly after fitting
    (see the demo notebooks).
"""

from __future__ import annotations
import numpy as np
from .constants import ETOK


def energy_to_k(energy, e0):
    """
    Convert photon energy (eV) to photoelectron wavenumber k (Ang^-1).

    Parameters
    ----------
    energy : array_like
        Photon energy in eV.
    e0 : float
        Absorption edge energy (E0) in eV.

    Returns
    -------
    k : ndarray
        Photoelectron wavenumber in Ang^-1. Values below E0 would be
        imaginary (the photoelectron doesn't propagate) and are
        clipped to 0 rather than returning NaN, matching the usual
        convention in EXAFS software.
    """
    energy = np.asarray(energy, dtype=float)
    de = energy - e0
    k = np.sqrt(np.clip(de, 0, None) * ETOK)
    return k


def k_to_energy(k, e0):
    """
    Convert photoelectron wavenumber k (Ang^-1) back to photon energy (eV).

    Parameters
    ----------
    k : array_like
        Photoelectron wavenumber, Ang^-1.
    e0 : float
        Absorption edge energy (E0) in eV.

    Returns
    -------
    energy : ndarray
        Photon energy in eV.
    """
    k = np.asarray(k, dtype=float)
    return e0 + (k ** 2) / ETOK


# ---------------------------------------------------------------
# Aliases for people/papers that call the photoelectron wavenumber
# "q" instead of "k". These are identical to the functions above.
# ---------------------------------------------------------------
def energy_to_q(energy, e0):
    """Alias of :func:`energy_to_k` (see module docstring for naming note)."""
    return energy_to_k(energy, e0)


def q_to_energy(q, e0):
    """Alias of :func:`k_to_energy` (see module docstring for naming note)."""
    return k_to_energy(q, e0)


def find_e0(energy, mu):
    """
    Estimate the absorption edge energy E0 as the energy of the
    maximum of the first derivative dmu/dE (the steepest rise of
    the edge). This is the standard, simple way to get a first
    guess for E0; you can always override it manually.

    Parameters
    ----------
    energy, mu : array_like
        Raw (or pre-edge-subtracted) absorption spectrum.

    Returns
    -------
    e0 : float
        Estimated edge energy, in the same units as `energy`.
    """
    energy = np.asarray(energy, dtype=float)
    mu = np.asarray(mu, dtype=float)
    deriv = np.gradient(mu, energy)
    # Restrict the search to the central 90% of the scan so noisy
    # endpoints can't be mistaken for the edge.
    n = len(energy)
    lo, hi = int(0.03 * n), int(0.97 * n)
    idx = lo + np.argmax(deriv[lo:hi])
    return float(energy[idx])
