"""
Fourier-transform tools: k-space windows and the forward chi(k) -> chi(R)
transform used to turn EXAFS oscillations into a pseudo-radial-distribution
function.

Transforming k-weighted EXAFS oscillations into an R-space
pseudo-radial-distribution function is the original idea of
D. E. Sayers, E. A. Stern & F. W. Lytle, *Phys. Rev. Lett.* **27**,
1204 (1971), doi:10.1103/PhysRevLett.27.1204.  The exact normalisation and
window conventions here follow the common EXAFS-community choice (same as
IFEFFIT/Larch/Athena, https://xraypy.github.io/xraylarch/):

    chi(R) = (1/sqrt(2*pi)) * Integral[ k^kweight * chi(k) * Omega(k) * exp(2i k R) dk ]

implemented numerically with a zero-padded FFT on a uniform k-grid.
`Omega(k)` is a window function that tapers chi(k) smoothly to zero at
the ends of the fit range, avoiding truncation ripples ("sinc wiggles")
in R-space.
"""

from __future__ import annotations
import numpy as np
from .constants import TWO_PI


def window(k, kmin, kmax, dk=1.0, kind="hanning"):
    """
    Build a tapered window function Omega(k), 0 outside [kmin, kmax],
    rising smoothly from 0 to 1 over a width `dk` at each edge, and
    equal to 1 in the interior.

    Parameters
    ----------
    k : array_like
        k-grid the window will be evaluated on.
    kmin, kmax : float
        Window range in Ang^-1.
    dk : float
        Taper width in Ang^-1 at each edge.
    kind : {"hanning", "kaiser", "welch", "boxcar"}
        Taper shape. "hanning" (cosine taper) is the most common
        default and a safe choice for beginners. "kaiser" gives lower
        side lobes at the cost of a slightly wider main peak.

    Returns
    -------
    win : ndarray, same shape as k
    """
    k = np.asarray(k, dtype=float)
    win = np.zeros_like(k)
    dk = max(dk, 1e-6)

    inside = (k >= kmin) & (k <= kmax)
    left_taper = (k >= kmin - dk) & (k < kmin)
    right_taper = (k > kmax) & (k <= kmax + dk)

    win[inside] = 1.0

    if kind == "boxcar":
        pass  # already a hard step, nothing else to do

    elif kind == "hanning":
        x = (k[left_taper] - (kmin - dk)) / dk
        win[left_taper] = 0.5 * (1 - np.cos(np.pi * x))
        x = (kmax + dk - k[right_taper]) / dk
        win[right_taper] = 0.5 * (1 - np.cos(np.pi * x))

    elif kind == "welch":
        x = (k[left_taper] - (kmin - dk)) / dk
        win[left_taper] = 1 - (1 - x) ** 2
        x = (kmax + dk - k[right_taper]) / dk
        win[right_taper] = 1 - (1 - x) ** 2

    elif kind == "kaiser":
        beta = 6.0  # reasonable default: low side lobes, moderate width
        x = (k[left_taper] - (kmin - dk)) / dk
        win[left_taper] = np.i0(beta * np.sqrt(np.clip(1 - (1 - x) ** 2, 0, 1))) / np.i0(beta)
        x = (kmax + dk - k[right_taper]) / dk
        win[right_taper] = np.i0(beta * np.sqrt(np.clip(1 - (1 - x) ** 2, 0, 1))) / np.i0(beta)

    else:
        raise ValueError(f"Unknown window kind: {kind!r}")

    return win


def ft_windowed(k, chi, kweight=2, window="hanning",
                 kmin=None, kmax=None, dk=1.0, nfft=2048, kstep=0.05):
    """
    Forward Fourier transform chi(k) -> chi(R).

    The input chi(k) does not need to be on a uniform grid; it is
    linearly interpolated onto a uniform k-grid with spacing `kstep`
    before the FFT (this is what real data coming off a monochromator,
    which is uniform in E not k, requires anyway).

    Parameters
    ----------
    k, chi : array_like
        EXAFS signal, e.g. from :func:`escape.exafs.background.autobk`.
    kweight : int
        Power of k used to weight chi(k) before transforming
        (k*chi(k), k^2*chi(k), k^3*chi(k) are all standard choices;
        higher k-weight emphasizes high-k / heavy-scatterer contributions).
    window : str
        Window shape, passed to :func:`window`.
    kmin, kmax : float
        Range of the transform. Defaults to the full range of `k`.
    dk : float
        Window taper width, Ang^-1.
    nfft : int
        FFT length (zero-padding). Larger nfft gives a finer (but not
        more informative) grid in R.
    kstep : float
        Grid spacing (Ang^-1) used for the uniform-k resampling.

    Returns
    -------
    r : ndarray
        Radial distance grid, Angstrom.
    chir : ndarray (complex)
        Complex chi(R).
    kuniform : ndarray
        The uniform k-grid actually used (handy for diagnostics/plots).
    """
    window_kind = window  # keep the string; module-level window() fetched via globals()

    k = np.asarray(k, dtype=float)
    chi = np.asarray(chi, dtype=float)
    if kmin is None:
        kmin = k.min()
    if kmax is None:
        kmax = k.max()

    kuniform = np.arange(0, kmax + dk + kstep, kstep)
    chi_uniform = np.interp(kuniform, k, chi, left=0.0, right=0.0)

    win = globals()["window"](kuniform, kmin, kmax, dk=dk, kind=window_kind)

    weighted = chi_uniform * (kuniform ** kweight) * win

    # zero-pad and FFT.
    # We want chi(R) = (1/sqrt(pi)) * Integral chi(k) exp(2i k R) dk, and
    # approximate the integral with an M-point FFT of samples spaced by
    # `kstep`. Matching FFT's exp(-2*pi*i*m*j/M) kernel to exp(2*i*k_j*R_m)
    # (k_j = j*kstep) gives the FFT-bin-to-R mapping R_m = pi*m/(M*kstep).
    n = max(nfft, len(kuniform))
    M = 2 * n
    fft_vals = np.fft.fft(weighted, n=M) * kstep
    # positive-R half, matching exp(2ikR) convention
    r = np.pi * np.arange(n) / (M * kstep)
    chir = fft_vals[:n] / np.sqrt(np.pi)

    return r, chir, kuniform
