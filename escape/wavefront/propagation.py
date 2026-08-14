"""Free-space propagation of coherent X-ray fields.

This module implements the *angular-spectrum* (a.k.a. Fresnel transfer
function) propagator that is at the heart of every wavefront-sensing
calculation:

* **forward** propagation is used to *simulate* the Talbot self-image that a
  grating casts onto a detector, and
* **backward** propagation (negative distance) is used to *reconstruct* where
  a beam came from -- e.g. to propagate a measured detector field back to the
  focus or to any requested plane.

The maths is deliberately kept in one small, well-documented place so it can be
read top to bottom.  It is written against plain NumPy arrays and is therefore
trivially wrapped for lazy, chunked evaluation on stacks of detector images via
:func:`propagate_array`, which plugs into ``escape``'s
:meth:`escape.Array.map_index_blocks` machinery.

The angular-spectrum / Fresnel transfer-function method is standard Fourier
optics (J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., 2005, ch. 3-4).
The paraxial phase convention matches M. Seaberg's ``lcls_beamline_toolbox``
(https://github.com/mseaberg) so simulated and reconstructed fields agree.

Why no numba here?
------------------
The propagator is *entirely* FFT-bound.  ``numpy.fft`` (and, for stacks,
``dask.array.fft``) already dispatch to a well-tuned, memory-clean C library
(pocketfft / MKL).  Adding a ``numba`` kernel would buy nothing and would
re-introduce exactly the kind of thread-pool / reference-holding leak that has
bitten the ``jungfrau_utils`` numba image corrections when they are called
repeatedly from inside dask workers.  See :ref:`the notes at the bottom of this
file <numba-dask-notes>` for the details and for the pattern to use if you
*must* mix numba with dask.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "wavelength_from_energy",
    "energy_from_wavelength",
    "fresnel_number",
    "angular_spectrum_propagate",
    "propagate_array",
]

# Physical constants (SI) ---------------------------------------------------
_H_EV_S = 4.135667696e-15  # Planck constant in eV*s
_C_M_S = 299792458.0  # speed of light in m/s
_HC_EV_M = _H_EV_S * _C_M_S  # ~1.239841984e-6 eV*m


def wavelength_from_energy(energy_eV):
    """Photon wavelength ``lambda0`` (m) from photon energy (eV).

    ``lambda0 = h c / E``.  Works on scalars and arrays.
    """
    return _HC_EV_M / np.asarray(energy_eV, dtype=float)


def energy_from_wavelength(lambda0_m):
    """Photon energy (eV) from wavelength (m) -- inverse of
    :func:`wavelength_from_energy`."""
    return _HC_EV_M / np.asarray(lambda0_m, dtype=float)


def fresnel_number(aperture_m, z_m, lambda0_m):
    """Fresnel number ``a^2 / (lambda z)``.

    A quick sanity gauge: ``N_F >> 1`` is the near field (geometric / Talbot
    regime), ``N_F << 1`` is the far field (Fraunhofer regime).
    """
    return aperture_m ** 2 / (lambda0_m * z_m)


def _spatial_frequencies(shape, dx, dy=None):
    """Return meshed FFT spatial frequencies ``(fx, fy)`` for an array of
    ``shape`` sampled at pixel sizes ``dx`` (and ``dy``).

    Uses :func:`numpy.fft.fftfreq` ordering (DC at index 0) so the transfer
    function can be applied to a *non-shifted* FFT -- no ``fftshift`` bookkeeping
    and no accidental half-pixel offsets.
    """
    if dy is None:
        dy = dx
    n, m = shape[-2], shape[-1]
    fx = np.fft.fftfreq(m, d=dx)
    fy = np.fft.fftfreq(n, d=dy)
    fx, fy = np.meshgrid(fx, fy)
    return fx, fy


def angular_spectrum_propagate(
    field,
    dx,
    z,
    lambda0,
    dy=None,
    paraxial=True,
    evanescent="drop",
):
    """Propagate a complex field a distance ``z`` in free space.

    The field is advanced with the angular-spectrum method::

        E(z) = IFFT{ FFT{E(0)} * H(fx, fy; z) }

    with transfer function

    * paraxial (Fresnel)      ``H = exp(-i pi lambda z (fx^2 + fy^2))``
    * non-paraxial (rigorous) ``H = exp(i 2pi/lambda z sqrt(1 - (lambda fx)^2 - (lambda fy)^2))``

    The two agree to better than a percent whenever the numerical aperture is
    small, which is essentially always true for hard X-rays.  The paraxial form
    matches the convention used in M. Seaberg's ``lcls_beamline_toolbox`` (there
    written ``phi = -k0/2 (lambda fx)^2 dz``) so simulated and reconstructed
    fields are mutually consistent.

    Parameters
    ----------
    field : ndarray, complex
        Complex field ``E`` on a regular grid.  The **last two axes** are the
        spatial ``(y, x)`` axes; any leading axes (e.g. an event/pulse axis)
        are treated as a batch and propagated independently.  This is what makes
        the function drop straight into :meth:`escape.Array.map_index_blocks`.
    dx : float
        Pixel size along ``x`` (m).
    z : float
        Propagation distance (m).  **Positive = downstream** (forward, toward
        the detector), **negative = upstream** (backward, toward the focus).
    lambda0 : float
        Wavelength (m).  See :func:`wavelength_from_energy`.
    dy : float, optional
        Pixel size along ``y`` (m); defaults to ``dx`` (square pixels).
    paraxial : bool, optional
        Use the Fresnel transfer function (default) or the rigorous one.
    evanescent : {"drop", "keep"}, optional
        Only used when ``paraxial=False``.  ``"drop"`` zeroes evanescent
        components (frequencies outside the light circle); this is the physical
        choice for propagation over any appreciable distance.

    Returns
    -------
    ndarray, complex
        The propagated field, same shape and dtype-kind as the input.

    Notes
    -----
    The grid is assumed identical at input and output (no Fresnel *scaling*).
    For very large propagation distances where the beam would grow outside the
    window, either pad the field first or use a two-step / scaled propagator.
    """
    field = np.asarray(field)
    fx, fy = _spatial_frequencies(field.shape, dx, dy)

    if paraxial:
        H = np.exp(-1j * np.pi * lambda0 * z * (fx ** 2 + fy ** 2))
    else:
        # rigorous angular spectrum
        arg = 1.0 - (lambda0 * fx) ** 2 - (lambda0 * fy) ** 2
        kz = 2 * np.pi / lambda0 * np.sqrt(np.clip(arg, 0.0, None))
        H = np.exp(1j * kz * z)
        if evanescent == "drop":
            H = np.where(arg < 0, 0.0, H)

    # FFT over the two spatial axes only; leading (batch/event) axes ride along.
    g = np.fft.fft2(field, axes=(-2, -1))
    g = g * H  # broadcasts H(N,M) across any leading batch axes
    out = np.fft.ifft2(g, axes=(-2, -1))
    return out.astype(np.result_type(field.dtype, np.complex64))


def propagate_array(
    array,
    dx,
    z,
    lambda0,
    dy=None,
    paraxial=True,
):
    """Lazily propagate every event of an ``escape`` :class:`~escape.Array`.

    ``array`` holds a stack of complex fields with shape ``(n_events, N, M)``
    (typically a reconstructed detector field, one per FEL pulse).  Each event
    is propagated independently by ``z`` using
    :func:`angular_spectrum_propagate`, block-wise, so nothing is pulled into
    memory until you ``.compute()``.

    This is the escape-native entry point for **backpropagating a beam profile
    to a requested distance** across a whole run: pass a negative ``z`` to walk
    the measured field upstream toward the focus.

    Parameters
    ----------
    array : escape.Array
        Complex-valued Array; event axis first, two spatial axes following.
    dx, z, lambda0, dy, paraxial
        Forwarded to :func:`angular_spectrum_propagate`.

    Returns
    -------
    escape.Array
        A new lazy Array of propagated complex fields, sharing the input's
        event index and scan metadata.

    Examples
    --------
    >>> focus = propagate_array(det_field, dx=6.5e-6, z=-2.3,
    ...                         lambda0=wavelength_from_energy(9500.0))
    >>> intensity = focus.map_index_blocks(lambda b: np.abs(b) ** 2,
    ...                                    dtype=float)
    >>> intensity.mean().compute()
    """

    def _block(block):
        # block shape == (events_in_chunk, N, M); propagate the whole chunk in
        # one vectorised FFT call -- no python-level per-event loop.
        return angular_spectrum_propagate(
            block, dx=dx, z=z, lambda0=lambda0, dy=dy, paraxial=paraxial
        )

    return array.map_index_blocks(_block, dtype=np.complex64)


# ---------------------------------------------------------------------------
# .. _numba-dask-notes:
#
# Mixing numba with dask without leaking memory
# ---------------------------------------------------------------------------
# The propagation above intentionally uses only ``numpy.fft`` because it is
# both fast and memory-clean.  If a *genuinely* element-wise, non-FFT kernel
# needs numba speed-ups (as the jungfrau_utils gain/pedestal corrections do),
# the following rules keep it from leaking when called from dask workers:
#
# 1. Compile ONCE at import, module level -- never inside the mapped function.
#    A ``@njit`` decorator placed on a closure recompiles (and caches a new
#    reference) on every task, which is the most common "leak".
#
# 2. Do NOT use ``@njit(parallel=True)`` / ``prange`` / ``@guvectorize(
#    target='parallel')`` inside a dask graph.  Numba's own threading layer
#    (TBB/OMP) spins up a pool per process that is not released between tasks
#    and fights dask's scheduler for cores.  Let *dask* provide the
#    parallelism (one thread per chunk) and keep the kernel single-threaded:
#    ``@njit(cache=True, fastmath=True)`` with a plain Python ``range``.
#    If you must, pin it with ``NUMBA_NUM_THREADS=1`` /
#    ``numba.config.THREADING_LAYER = 'workqueue'`` in the worker.
#
# 3. Feed the kernel a C-contiguous copy (``np.ascontiguousarray(block)``) and
#    write into a pre-allocated ``out`` array; returning views of numba-managed
#    buffers can keep chunks alive.
#
# 4. Wrap the numba call in a thin python function and hand *that* to
#    ``map_index_blocks`` -- exactly the ``_block`` pattern used above -- so the
#    compiled artifact is shared, not rebuilt.
#
# Sketch::
#
#     from numba import njit
#
#     @njit(cache=True, fastmath=True)          # rule 1 & 2
#     def _apply_gain(block, gain, pede, out):
#         nev, n, m = block.shape
#         for e in range(nev):                  # plain range, dask parallelises
#             for i in range(n):
#                 for j in range(m):
#                     out[e, i, j] = (block[e, i, j] - pede[i, j]) * gain[i, j]
#         return out
#
#     def gain_correct(block, gain, pede):
#         block = np.ascontiguousarray(block)   # rule 3
#         out = np.empty_like(block, dtype=np.float32)
#         return _apply_gain(block, gain, pede, out)
#
#     corrected = raw.map_index_blocks(gain_correct, gain, pede, dtype='f4')
