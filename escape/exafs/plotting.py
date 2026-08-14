"""
Quick-look plotting helpers for the three standard EXAFS views:
mu(E) with background, chi(k) k-weighted, and |chi(R)|.

These are thin matplotlib wrappers meant to save typing in notebooks;
for publication figures you will likely want to build your own plots
directly with matplotlib, using the arrays these functions return
from the rest of the package.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_bkg(preedge_result, bkg_result, ax=None, emin=None, emax=None):
    """Plot mu(E) together with the fitted spline background mu0(E)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    e0 = preedge_result.e0
    ax.plot(preedge_result.energy, preedge_result.mu, label="mu(E)", lw=1.2)
    from .energy_k import k_to_energy
    e_bkg = k_to_energy(bkg_result.k, e0)
    mu0_full = bkg_result.mu0 * preedge_result.edge_step + \
        np.interp(e_bkg, preedge_result.energy, preedge_result.pre_edge)
    ax.plot(e_bkg, mu0_full, label="mu0(E) background", lw=1.2, ls="--")
    ax.axvline(e0, color="gray", lw=0.8, ls=":", label="E0")
    if emin is not None or emax is not None:
        lo = e0 + emin if emin is not None else preedge_result.energy.min()
        hi = e0 + emax if emax is not None else preedge_result.energy.max()
        ax.set_xlim(lo, hi)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$\mu(E)$")
    ax.legend()
    ax.set_title("Absorption spectrum and background")
    return ax


def plot_chik(k, chi, kweight=2, ax=None):
    """Plot k^kweight * chi(k)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    from .utils import kweight_chi
    y = kweight_chi(k, chi, kweight)
    ax.plot(k, y, lw=1.2)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    ax.set_ylabel(rf"$k^{kweight} \chi(k)$ ($\mathrm{{\AA}}^{{-{kweight}}}$)")
    ax.set_title("EXAFS oscillations")
    return ax


def plot_chir(r, chir, ax=None, rmax=6.0, show_imag=False):
    """Plot |chi(R)| (and optionally Re[chi(R)])."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    mag = np.abs(chir)
    ax.plot(r, mag, lw=1.4, label="|chi(R)|")
    if show_imag:
        ax.plot(r, chir.real, lw=0.9, alpha=0.6, label="Re[chi(R)]")
        ax.legend()
    ax.set_xlim(0, rmax)
    ax.set_xlabel(r"$R$ ($\mathrm{\AA}$, not phase-corrected)")
    ax.set_ylabel(r"$|\chi(R)|$")
    ax.set_title("Fourier-transformed EXAFS (pseudo radial distribution)")
    return ax
