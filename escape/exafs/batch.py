"""Batch EXAFS reduction over ``escape`` Arrays (dask-parallel).

The single-spectrum functions in the rest of :mod:`escape.exafs` operate on one
``mu(E)`` curve.  At an FEL or a scanning beamline you routinely have **many**
spectra -- one per pulse, per scan step, or per repeat -- stacked along the
event axis of an :class:`escape.Array`.  This module reduces such a stack
**lazily and in parallel** by plugging the per-spectrum reduction into
``escape``'s :meth:`escape.Array.map_index_blocks` (dask ``map_blocks``), so
nothing is pulled into memory until you ``.compute()``.

The design mirrors :func:`escape.wavefront.propagate_array`: each function takes
an escape Array whose **event axis is first** and whose remaining axis is the
spectral axis, and returns a new lazy Array on a *fixed* output grid so every
event stays aligned.

Typical use::

    import numpy as np
    from escape import exafs

    kgrid = exafs.common_k_grid(0, 16, 0.05)

    # mu_arr: escape.Array of shape (n_events, n_energy), shared `energy` axis
    chi = exafs.reduce_array(mu_arr, energy, e0=8980.5, edge_step=2.83,
                             kgrid=kgrid, rbkg=1.0)      # -> Array (n_events, n_k)
    chik_mean = chi.mean(axis=0).compute()

    r, chir = exafs.ft_array(chi, kgrid, kweight=2, kmin=3, kmax=13)
    chir_mean = chir.mean(axis=0).compute()
"""

from __future__ import annotations

import numpy as np

from .energy_k import energy_to_k, find_e0
from .background import autobk
from .fourier import ft_windowed

__all__ = [
    "optical_density",
    "common_k_grid",
    "reduce_spectrum",
    "reduce_array",
    "ft_array",
]


def optical_density(i0, i1):
    """Transmission-mode absorption ``mu = -ln(I1 / I0)``.

    Works element-wise on plain NumPy arrays *and* on dask arrays, so it can be
    applied to raw escape Arrays of detector/diode readings before reduction::

        mu = i1_arr.map_index_blocks(lambda b, r: optical_density(r, b), i0_arr)

    (Fluorescence-yield data is already proportional to ``mu``; skip this step.)
    """
    return -np.log(np.asarray(i1) / np.asarray(i0))


def common_k_grid(kmin=0.0, kmax=16.0, kstep=0.05):
    """Uniform k-grid (Ang^-1) shared by every reduced spectrum in a stack."""
    return np.arange(kmin, kmax + kstep, kstep)


def reduce_spectrum(mu, energy, kgrid, e0=None, edge_step=None,
                    rbkg=1.0, refine=False, **autobk_kwargs):
    """Reduce one ``mu(E)`` spectrum to ``chi(k)`` sampled on ``kgrid``.

    A thin wrapper around :func:`escape.exafs.autobk` that (a) optionally finds
    ``e0`` per spectrum and (b) interpolates the resulting ``chi(k)`` onto the
    shared ``kgrid`` so a whole stack can be assembled into one array.

    Parameters
    ----------
    mu, energy : ndarray
        One spectrum (1-D), same length.
    kgrid : ndarray
        Output k-grid (see :func:`common_k_grid`).
    e0 : float, optional
        Edge energy (eV).  If ``None`` it is estimated per spectrum with
        :func:`escape.exafs.find_e0` -- fine for clean data, but for noisy
        single-pulse spectra prefer passing a fixed ``e0`` from a reference.
    edge_step : float, optional
        Edge-step normalisation.  If ``None`` a crude estimate (post- minus
        pre-edge mean) is used; passing a fixed value from a reference spectrum
        is strongly recommended for a stack so all events share one scale.
    rbkg, refine, autobk_kwargs
        Forwarded to :func:`escape.exafs.autobk`.  ``refine=False`` (the fast
        least-squares spline, no per-spectrum Nelder-Mead) is the default here
        because a batch may contain thousands of spectra.

    Returns
    -------
    ndarray
        ``chi(k)`` on ``kgrid`` (zeros outside the measured k-range).
    """
    mu = np.asarray(mu, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if e0 is None:
        e0 = find_e0(energy, mu)
    if edge_step is None:
        # crude fallback: mean above the edge minus mean below it
        below = energy < e0
        above = energy > e0 + 50.0
        edge_step = float(np.nanmean(mu[above]) - np.nanmean(mu[below]))
        if not np.isfinite(edge_step) or edge_step <= 0:
            edge_step = 1.0
    bkg = autobk(energy, mu, e0, edge_step, rbkg=rbkg, refine=refine, **autobk_kwargs)
    return np.interp(kgrid, bkg.k, bkg.chi, left=0.0, right=0.0)


def reduce_array(mu_array, energy, kgrid, e0=None, edge_step=None,
                 rbkg=1.0, refine=False):
    """Lazily reduce an escape Array of spectra to an Array of ``chi(k)``.

    Parameters
    ----------
    mu_array : escape.Array
        Absorption spectra, shape ``(n_events, n_energy)``; the ``energy`` axis
        is shared across events.
    energy : ndarray
        The common energy axis (length ``n_energy``).
    kgrid : ndarray
        Shared output k-grid; see :func:`common_k_grid`.
    e0, edge_step, rbkg, refine
        Forwarded to :func:`reduce_spectrum`.  Pass fixed ``e0``/``edge_step``
        (e.g. from the run-averaged spectrum) so every event lands on the same
        scale -- the usual choice for a stable series.

    Returns
    -------
    escape.Array
        Lazy Array of shape ``(n_events, len(kgrid))`` holding ``chi(k)``,
        sharing the input's event index and scan metadata.
    """
    energy = np.asarray(energy, dtype=float)
    kgrid = np.asarray(kgrid, dtype=float)

    def _block(block):
        # block: (n_events_in_chunk, n_energy) -> (n_events_in_chunk, n_k)
        out = np.empty((block.shape[0], kgrid.size), dtype=float)
        for i in range(block.shape[0]):
            out[i] = reduce_spectrum(block[i], energy, kgrid, e0=e0,
                                     edge_step=edge_step, rbkg=rbkg, refine=refine)
        return out

    return mu_array.map_index_blocks(
        _block, new_element_size=(kgrid.size,), dtype=float
    )


def ft_array(chi_array, kgrid, kweight=2, window="hanning",
             kmin=3.0, kmax=None, dk=1.0, rmax=8.0, kstep=0.05):
    """Lazily Fourier-transform an escape Array of ``chi(k)`` to ``chi(R)``.

    Applies :func:`escape.exafs.ft_windowed` per event over dask.

    Parameters
    ----------
    chi_array : escape.Array
        ``chi(k)`` stack of shape ``(n_events, len(kgrid))`` (e.g. from
        :func:`reduce_array`).
    kgrid : ndarray
        The k-grid the ``chi`` was sampled on.
    kweight, window, kmin, kmax, dk, kstep
        Forwarded to :func:`escape.exafs.ft_windowed`.
    rmax : float
        Largest R (Ang) to keep in the output (trims the FFT tail).

    Returns
    -------
    r : ndarray
        The R-grid (Ang) shared by every event.
    chir_array : escape.Array
        Lazy complex Array of shape ``(n_events, len(r))`` holding ``chi(R)``.
    """
    kgrid = np.asarray(kgrid, dtype=float)
    # determine the shared R-grid once from a probe transform
    r_full, _, _ = ft_windowed(kgrid, np.zeros_like(kgrid), kweight=kweight,
                               window=window, kmin=kmin, kmax=kmax, dk=dk, kstep=kstep)
    keep = r_full <= rmax
    r = r_full[keep]

    def _block(block):
        out = np.empty((block.shape[0], r.size), dtype=complex)
        for i in range(block.shape[0]):
            _, chir, _ = ft_windowed(kgrid, block[i], kweight=kweight,
                                     window=window, kmin=kmin, kmax=kmax,
                                     dk=dk, kstep=kstep)
            out[i] = chir[keep]
        return out

    chir_array = chi_array.map_index_blocks(
        _block, new_element_size=(r.size,), dtype=complex
    )
    return r, chir_array
