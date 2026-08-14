"""Talbot (single-grating shearing) X-ray wavefront sensing.

A distilled, streamlined re-implementation of the fractional-Talbot wavefront
sensor developed by M. Seaberg *et al.* at LCLS/SLAC
(`lcls_beamline_toolbox <https://github.com/mseaberg/lcls_beamline_toolbox>`_
and the ``wfs_interface`` GUI), rewritten around ``escape`` conventions and
plain NumPy so it reads as a teaching implementation.

Background reading
------------------
* Y. Liu, M. Seaberg, Y. Feng, K. Li, Y. Ding, G. Marcus, D. Fritz, X. Shi,
  W. Grizolli, L. Assoufid, P. Walter & A. Sakdinawat, "X-ray free-electron
  laser wavefront sensing using the fractional Talbot effect", *J. Synchrotron
  Rad.* **27**(2), 254-261 (2020).  doi:10.1107/S1600577519017107
* Y. Liu, M. Seaberg, D. Zhu, J. Krzywinski, F. Seiboth, C. Hardin, D. Cocco,
  A. Aquila, B. Nagler, H. J. Lee, S. Boutet & Y. Feng, "High-accuracy
  wavefront sensing for x-ray free electron lasers", *Optica* **5**(8),
  967-975 (2018).  doi:10.1364/OPTICA.5.000967

Reference implementation (the code this module distils, by M. Seaberg):
``lcls_beamline_toolbox`` and ``wfs_interface`` at https://github.com/mseaberg.

Underlying algorithms: Fourier-fringe demodulation -- Takeda, Ina & Kobayashi,
*J. Opt. Soc. Am.* **72**(1), 156-160 (1982); gradient integration --
Frankot & Chellappa, *IEEE Trans. PAMI* **10**(4), 439-451 (1988).

The physics in one paragraph
----------------------------
A 2-D checkerboard **pi-phase** grating (etched in diamond/Si) is placed in the
beam.  A short distance ``zT`` downstream -- a *fractional Talbot distance* --
the grating produces a high-contrast intensity self-image (a chequerboard of
spots) on a detector of pixel size ``dx``.  Local wavefront *slope* tilts the
beam and hence **displaces** the local fringe; wavefront *curvature* magnifies
or shrinks the fringe **period**.  Recovering those distortions is a classic
**Fourier-fringe (Takeda) demodulation**:

1. FFT the detector image.  The chequerboard puts energy into a zero order and
   four first-order peaks (``+/-x`` and ``+/-y``).
2. Band-pass one first-order peak, inverse-FFT, and take its phase.  After the
   grating carrier is removed this residual phase is the beam's transverse
   **phase gradient** sheared by ``s = lambda zT / dg`` -- one along ``x`` from
   the x-peak, one along ``y`` from the y-peak.
3. Integrate the two gradients into a single 2-D wavefront (here via the
   FFT-based Frankot-Chellappa least-squares integrator).
4. The quadratic (defocus) term of that wavefront gives the radius of
   curvature and hence the distance to the focus; combining the reconstructed
   phase with the beam amplitude gives a complex field that can be
   *back-propagated* (see :mod:`escape.wavefront.propagation`) to any plane.

Everything downstream of :func:`reconstruct_wavefront` is grating-agnostic; the
only Talbot-specific step is the demodulation in :func:`fourier_fringe_gradients`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

import numpy as np
from scipy.fft import dctn, idctn
from scipy.sparse.linalg import cg, LinearOperator

from .propagation import (
    wavelength_from_energy,
    angular_spectrum_propagate,
)

__all__ = [
    "wavelength_from_energy",
    "checkerboard_grating",
    "talbot_distance",
    "simulate_talbot_image",
    "fourier_fringe_gradients",
    "integrate_gradients",
    "reconstruct_wavefront",
    "Wavefront",
]


# ---------------------------------------------------------------------------
# small centered-FFT helpers (peak picking is easiest with DC in the middle)
# ---------------------------------------------------------------------------
def _nfft(a):
    """Centered 2-D forward FFT (DC in the array centre)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(a)))


def _infft(a):
    """Centered 2-D inverse FFT (inverse of :func:`_nfft`)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(a)))


def _unwrap_masked(phase, mask):
    """2-D phase unwrap inside ``mask``; identity fallback if unavailable.

    Uses ``skimage.restoration.unwrap_phase`` when present (the reliability-
    guided algorithm of Herraez, Burton, Lalor & Gdeisat, *Appl. Opt.* **41**,
    7437 (2002)), restricted to the illuminated aperture so dark-field noise
    does not seed spurious wraps.  If scikit-image
    is not installed the wrapped phase is returned unchanged, which simply caps
    the usable dynamic range (see the module docs).
    """
    try:
        from skimage.restoration import unwrap_phase
    except Exception:
        return phase
    masked = np.ma.array(phase, mask=~mask)
    return np.asarray(unwrap_phase(masked).filled(0.0))


def _coords(shape, dx, dy=None):
    """Centred real-space coordinate meshes ``(x, y)`` (m)."""
    if dy is None:
        dy = dx
    n, m = shape
    x = (np.arange(m) - m / 2) * dx
    y = (np.arange(n) - n / 2) * dy
    return np.meshgrid(x, y)


def _freq_coords(shape, dx, dy=None):
    """Centred spatial-frequency meshes ``(fx, fy)`` (1/m) matching ``_nfft``."""
    if dy is None:
        dy = dx
    n, m = shape
    fx = np.fft.fftshift(np.fft.fftfreq(m, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(n, d=dy))
    return np.meshgrid(fx, fy)


# ---------------------------------------------------------------------------
# grating + geometry
# ---------------------------------------------------------------------------
def checkerboard_grating(shape, dx, period, phase=np.pi, dy=None):
    """Complex transmission of a 2-D checkerboard phase grating.

    The chequerboard alternates between transmission ``1`` and
    ``exp(i * phase)`` in a checker pattern of half-period ``period / 2``.  For
    the standard ``phase = pi`` grating the zero order is suppressed and the
    diffracted energy lands in the diagonal first orders -- the configuration
    used for Talbot wavefront sensing.

    Parameters
    ----------
    shape : (int, int)
        ``(N, M)`` pixel dimensions.
    dx : float
        Pixel size (m).
    period : float
        Grating period ``dg`` (m) -- one full black/white/black/white cycle.
    phase : float, optional
        Phase step of the etched squares (radians); ``pi`` by default.
    dy : float, optional
        Pixel size along ``y``; defaults to ``dx``.

    Returns
    -------
    ndarray, complex
        Transmission function ``t(x, y)``.
    """
    x, y = _coords(shape, dx, dy)
    # square wave in x and y, XOR-ed into a checkerboard
    sx = np.mod(np.floor(x / (period / 2)), 2)
    sy = np.mod(np.floor(y / (period / 2)), 2)
    checker = np.mod(sx + sy, 2)
    return np.where(checker > 0, np.exp(1j * phase), 1.0).astype(np.complex128)


def mesh_grating(shape, dx, period, contrast=0.6, dy=None):
    """Complex transmission of a smooth 2-D "mesh" (crossed-sinusoid) grating.

    ``t(x, y) = (1 + c cos(2 pi x / p)) (1 + c cos(2 pi y / p))``.

    Unlike the pi-checkerboard -- whose fractional-Talbot revivals are highly
    non-linear and only reconstruct cleanly with careful per-plane calibration --
    this grating has clean first orders at exactly ``+/-1/period`` and its Talbot
    self-image is a faithful, magnifying replica.  That makes it the preferred
    grating for **teaching** and for verifying a reconstruction pipeline end to
    end, even though real hard-X-ray sensors use etched checkerboards for
    efficiency.

    Parameters
    ----------
    shape : (int, int)
        ``(N, M)`` pixel dimensions.
    dx : float
        Pixel size (m).
    period : float
        Grating period (m).
    contrast : float, optional
        Modulation depth ``c`` in ``[0, 1]`` (default 0.6).
    dy : float, optional
        Pixel size along ``y``; defaults to ``dx``.

    Returns
    -------
    ndarray, complex
        Transmission function ``t(x, y)`` (real-valued amplitude, complex dtype).
    """
    x, y = _coords(shape, dx, dy)
    t = (1 + contrast * np.cos(2 * np.pi * x / period)) * (
        1 + contrast * np.cos(2 * np.pi * y / period)
    )
    return t.astype(np.complex128)


def talbot_distance(period, lambda0, fraction=1.0):
    """Fractional Talbot distance for a checkerboard/phase grating.

    The full Talbot distance is ``zT_full = 2 * dg^2 / lambda``.  Self-images
    recur at the *fractional* Talbot planes; the smooth :func:`mesh_grating`
    reconstructs cleanly at ``fraction = 1/2`` (a faithful, magnifying replica),
    whereas the realistic pi-:func:`checkerboard_grating` has a sharper, more
    non-linear revival structure best characterised near ``fraction = 1/8``.

    Parameters
    ----------
    period : float
        Grating period ``dg`` (m).
    lambda0 : float
        Wavelength (m).
    fraction : float, optional
        Multiple of the full Talbot distance ``2 dg^2 / lambda`` to return.

    Returns
    -------
    float
        Grating-to-detector distance (m).
    """
    return fraction * 2.0 * period ** 2 / lambda0


# ---------------------------------------------------------------------------
# forward simulation (didactic / for generating test data)
# ---------------------------------------------------------------------------
def simulate_talbot_image(
    shape,
    dx,
    period,
    lambda0,
    zT,
    incident_phase=None,
    incident_amplitude=None,
    grating="mesh",
    grating_phase=np.pi,
    dy=None,
):
    """Simulate the Talbot detector image for a known incident wavefront.

    Multiplies an incident complex field by the grating and Fresnel-propagates
    it to the detector plane, returning the detected **intensity**.  Handy for
    building test data and for teaching: feed it a known parabolic
    ``incident_phase`` (pure defocus) and check that
    :func:`reconstruct_wavefront` recovers the same radius of curvature.

    Parameters
    ----------
    shape, dx, period, lambda0, zT : see module conventions
        Detector shape, pixel size (m), grating period (m), wavelength (m),
        grating-to-detector distance (m).
    incident_phase : ndarray, optional
        Phase (radians) of the beam *at the grating*.  Default: flat (0).
    incident_amplitude : ndarray, optional
        Amplitude of the beam at the grating.  Default: uniform illumination.
    grating : {"mesh", "checkerboard"} or ndarray, optional
        Which grating to use, or a precomputed complex transmission array.
        ``"mesh"`` (default) is the smooth crossed-sinusoid grating whose
        self-image reconstructs cleanly; ``"checkerboard"`` is the realistic
        pi-phase grating (see :func:`mesh_grating` for why mesh is preferred
        for teaching).
    grating_phase : float, optional
        Phase step used when ``grating="checkerboard"`` (default ``pi``).
    dy : float, optional
        Pixel size along ``y``.

    Returns
    -------
    ndarray, float
        Detector intensity image ``|E(zT)|^2``.
    """
    if incident_amplitude is None:
        incident_amplitude = np.ones(shape)
    if incident_phase is None:
        incident_phase = np.zeros(shape)

    if isinstance(grating, str):
        if grating == "mesh":
            g = mesh_grating(shape, dx, period, dy=dy)
        elif grating == "checkerboard":
            g = checkerboard_grating(shape, dx, period, grating_phase, dy)
        else:
            raise ValueError("grating must be 'mesh', 'checkerboard', or an array")
    else:
        g = np.asarray(grating)

    field = incident_amplitude * np.exp(1j * incident_phase) * g
    detector = angular_spectrum_propagate(field, dx=dx, z=zT, lambda0=lambda0, dy=dy)
    return np.abs(detector) ** 2


def parabolic_phase(shape, dx, radius, lambda0, dy=None):
    """Phase (radians) of a spherical wave of radius of curvature ``radius``.

    ``phi = pi (x^2 + y^2) / (lambda * radius)``.  Positive ``radius`` is a
    diverging beam (source upstream), negative is converging (focus
    downstream).  Useful as ``incident_phase`` for
    :func:`simulate_talbot_image`.
    """
    x, y = _coords(shape, dx, dy)
    return np.pi * (x ** 2 + y ** 2) / (lambda0 * radius)


# ---------------------------------------------------------------------------
# reconstruction step 1: Fourier-fringe demodulation -> phase gradients
# ---------------------------------------------------------------------------
def _find_axis_peak(F, fx, fy, axis, f_min, f_max, off_axis_frac=0.25):
    """Locate the brightest first-order fringe peak on the ``+x`` or ``+y`` axis.

    The peak is searched inside an annular band ``f_min < |f_along| < f_max``
    and close to the chosen axis (``|f_perp| < off_axis_frac * f_max``).  This
    is deliberately located from a **reference** image where the fringe carrier
    is well defined; the same carrier is then reused to demodulate the
    measurement, which is what keeps the absolute wavefront (including defocus)
    from being silently subtracted away.

    Parameters
    ----------
    F : ndarray, complex
        Centred FFT of the (reference) image.
    fx, fy : ndarray
        Frequency meshes.
    axis : {"x", "y"}
        Which first order to isolate.
    f_min, f_max : float
        Radial search band (1/m); brackets the expected fringe frequency and
        excludes both the DC lobe and higher harmonics.
    off_axis_frac : float
        How far off-axis (perpendicular) to allow the peak, as a fraction of
        ``f_max``.

    Returns
    -------
    (float, float)
        The ``(fx0, fy0)`` of the located peak.
    """
    mag = np.abs(F)
    if axis == "x":
        along, perp = fx, fy
    else:
        along, perp = fy, fx
    box = (
        (along > f_min)
        & (along < f_max)
        & (np.abs(perp) < off_axis_frac * f_max)
    )
    search = np.where(box, mag, 0.0)
    idx = np.unravel_index(np.argmax(search), search.shape)
    return fx[idx], fy[idx]


def fourier_fringe_gradients(
    image,
    reference,
    dx,
    lambda0,
    zT,
    period=None,
    dy=None,
    peak_radius_frac=0.4,
    visibility_frac=0.15,
    unwrap=True,
):
    """Demodulate a Talbot image into two transverse wavefront gradients.

    This is the Takeda/Seaberg core, done **differentially against a reference
    image** (a flat / known-wavefront exposure -- exactly what is recorded during
    a WFS calibration).  Working against a reference is what makes the *absolute*
    wavefront -- defocus included -- recoverable: the reference fixes the fringe
    carrier, and the measured minus reference phase is the pure beam wavefront.

    Steps, per axis (``x`` then ``y``):

    1. Locate the first-order fringe peak in the **reference** FFT.
    2. Band-pass that peak in both reference and measurement, inverse-FFT, and
       remove the carrier -> two slowly varying complex "bands".
    3. The phase of ``band_meas * conj(band_ref)`` is the sheared wavefront
       difference; dividing by the shear ``s = f_carrier * lambda * zT`` (and a
       sign) gives the beam phase gradient ``d(phi)/dx`` in rad/m.

    Parameters
    ----------
    image : ndarray, float
        Measured Talbot detector image ``(N, M)``.
    reference : ndarray, float
        Reference image with a known (typically flat) wavefront, same geometry.
    dx : float
        Detector pixel size (m).
    lambda0 : float
        Wavelength (m).
    zT : float
        Grating-to-detector distance (m).
    period : float, optional
        Grating / fringe period hint (m), used only to bracket the peak search.
        If omitted the search spans a broad band and picks the dominant order.
    dy : float, optional
        Pixel size along ``y``; defaults to ``dx``.
    peak_radius_frac : float, optional
        Band-pass radius as a fraction of the carrier frequency.  Smaller =
        smoother, lower resolution.
    visibility_frac : float, optional
        Fringe-visibility threshold (fraction of peak) defining the illuminated
        aperture (``info["mask"]``); used to weight fits and to bound the
        unwrap, not to zero the gradients.
    unwrap : bool, optional
        2-D phase-unwrap the cross-phase inside the aperture (default ``True``)
        to extend the dynamic range past +/-pi.  Uses ``scikit-image`` when
        available and is a no-op otherwise.

    Returns
    -------
    grad_x, grad_y : ndarray, float
        Beam phase gradients ``d(phi)/dx`` and ``d(phi)/dy`` (rad/m), zeroed
        outside the valid (well-illuminated) region.
    info : dict
        Diagnostics: ``fx_peak``, ``fy_peak``, ``shear_x``, ``shear_y``,
        ``visibility``, ``mask`` (valid region), ``amplitude`` (beam amplitude
        estimate at the detector), ``fourier`` (the measurement FFT).
    """
    image = np.asarray(image, dtype=float)
    reference = np.asarray(reference, dtype=float)
    shape = image.shape
    fx, fy = _freq_coords(shape, dx, dy)
    x, y = _coords(shape, dx, dy)

    # frequency band in which to hunt for the first-order carrier
    if period is not None:
        f_hint = 1.0 / period
        f_min, f_max = 0.4 * f_hint, 3.0 * f_hint
    else:
        f_nyq = 0.5 / dx
        f_min, f_max = 0.05 * f_nyq, 0.95 * f_nyq

    Fr = _nfft(reference)
    F = _nfft(image)

    # carriers located from the REFERENCE so measurement and reference share them
    fx_px, fy_px = _find_axis_peak(Fr, fx, fy, "x", f_min, f_max)
    fx_py, fy_py = _find_axis_peak(Fr, fx, fy, "y", f_min, f_max)

    def _demod(Fany, fx0, fy0):
        radius = peak_radius_frac * np.hypot(fx0, fy0)
        mask = (fx - fx0) ** 2 + (fy - fy0) ** 2 < radius ** 2
        band = _infft(Fany * mask)
        carrier = np.exp(-1j * 2 * np.pi * (fx0 * x + fy0 * y))
        return band * carrier

    bx_ref = _demod(Fr, fx_px, fy_px)
    by_ref = _demod(Fr, fx_py, fy_py)
    bx = _demod(F, fx_px, fy_px)
    by = _demod(F, fx_py, fy_py)

    # sheared wavefront difference from the cross-phase (measurement vs reference)
    res_x = -np.angle(bx * np.conj(bx_ref))
    res_y = -np.angle(by * np.conj(by_ref))

    # fringe visibility: high inside the illuminated beam, ~0 in the dark
    visibility = np.abs(bx) * np.abs(by)
    mask = visibility > visibility_frac * visibility.max()

    # For strongly curved beams the cross-phase exceeds +/-pi and wraps.  A 2-D
    # phase unwrap inside the aperture extends the sensor's dynamic range by
    # many wraps (focus distances of order a metre instead of tens of metres).
    # Falls back gracefully to the wrapped result if scikit-image is absent.
    if unwrap:
        res_x = _unwrap_masked(res_x, mask)
        res_y = _unwrap_masked(res_y, mask)

    # shear the grating imposes at distance zT for a carrier at f_carrier:
    #     res = f_carrier * lambda * zT * d(phi)/dx      (see docs derivation)
    shear_x = np.hypot(fx_px, fy_px) * lambda0 * zT
    shear_y = np.hypot(fx_py, fy_py) * lambda0 * zT

    # The visibility is returned as a *weight* (not a hard mask) so the weighted
    # integrator can down-weight -- rather than truncate -- the noisy exterior;
    # hard-zeroing the gradient there would inject a boundary discontinuity that
    # corrupts non-symmetric aberrations (coma, trefoil, ...).
    grad_x = res_x / shear_x
    grad_y = res_y / shear_y

    info = {
        "fx_peak": fx_px,
        "fy_peak": fy_py,
        "shear_x": shear_x,
        "shear_y": shear_y,
        "visibility": visibility,
        "mask": mask,
        "amplitude": np.sqrt(visibility),
        "fourier": F,
    }
    return grad_x, grad_y, info


# ---------------------------------------------------------------------------
# reconstruction step 2: integrate gradients -> wavefront
# ---------------------------------------------------------------------------
def _integrate_frankot_chellappa(grad_x, grad_y, dx, dy):
    """Fast unweighted least-squares integration (three FFTs).

    Closed-form Fourier solution of the surface whose gradient best matches
    ``(grad_x, grad_y)``.  Assumes periodicity, so it is excellent for smooth,
    full-frame gradients but injects amplitude errors on non-symmetric modes
    when the gradient is only valid inside a sub-aperture -- use the weighted
    solver for that case.
    """
    n, m = grad_x.shape
    fx = np.fft.fftfreq(m, d=dx)
    fy = np.fft.fftfreq(n, d=dy)
    fx, fy = np.meshgrid(fx, fy)

    Gx = np.fft.fft2(grad_x)
    Gy = np.fft.fft2(grad_y)

    denom = (2 * np.pi) * (fx ** 2 + fy ** 2)
    denom[0, 0] = 1.0  # avoid division by zero at DC
    Phi = -1j * (fx * Gx + fy * Gy) / denom
    Phi[0, 0] = 0.0  # undefined piston -> 0
    phi = np.real(np.fft.ifft2(Phi))
    return phi - phi.mean()


def _integrate_weighted(grad_x, grad_y, weight, dx, dy, floor=0.02,
                        tol=1e-4, maxiter=200):
    """Weighted least-squares integration via preconditioned conjugate gradient.

    Minimises ``sum w[(dphi/dx - gx)^2 + (dphi/dy - gy)^2]`` where ``w`` is the
    fringe-visibility weight.  Down-weighting (rather than truncating) the dark
    exterior removes the boundary artefacts that otherwise corrupt coma-like
    aberrations.  The unweighted DCT Poisson solver is used as a preconditioner
    so only a few dozen CG iterations are needed.
    """
    n, m = grad_x.shape
    w = weight / weight.max() + floor  # floor keeps the operator well-conditioned

    def Dfx(p):
        d = np.zeros_like(p); d[:, :-1] = (p[:, 1:] - p[:, :-1]) / dx; return d

    def Dfy(p):
        d = np.zeros_like(p); d[:-1, :] = (p[1:, :] - p[:-1, :]) / dy; return d

    def DfxT(q):  # adjoint of Dfx
        d = np.zeros_like(q); d[:, :-1] -= q[:, :-1] / dx; d[:, 1:] += q[:, :-1] / dx; return d

    def DfyT(q):
        d = np.zeros_like(q); d[:-1, :] -= q[:-1, :] / dy; d[1:, :] += q[:-1, :] / dy; return d

    def A(pv):
        p = pv.reshape(n, m)
        return (DfxT(w * Dfx(p)) + DfyT(w * Dfy(p))).ravel()

    b = (DfxT(w * grad_x) + DfyT(w * grad_y)).ravel()

    # DCT (Neumann) Poisson preconditioner
    i = np.arange(n)[:, None]; j = np.arange(m)[None, :]
    denom = (2 * np.cos(np.pi * i / n) - 2) / dy ** 2 + (2 * np.cos(np.pi * j / m) - 2) / dx ** 2
    denom[0, 0] = 1.0

    def M(rv):
        R = dctn(rv.reshape(n, m), type=2, norm="ortho")
        P = R / denom; P[0, 0] = 0.0
        return idctn(P, type=2, norm="ortho").ravel()

    N = n * m
    phi, _ = cg(
        LinearOperator((N, N), matvec=A), b,
        rtol=tol, maxiter=maxiter, M=LinearOperator((N, N), matvec=M),
    )
    phi = phi.reshape(n, m)
    return phi - phi.mean()


def integrate_gradients(grad_x, grad_y, dx, dy=None, weight=None):
    """Least-squares integrate two phase gradients into a wavefront.

    Solves for the surface ``phi`` whose gradient best matches
    ``(grad_x, grad_y)``.

    Parameters
    ----------
    grad_x, grad_y : ndarray
        Gradients ``d(phi)/dx`` and ``d(phi)/dy`` (rad/m).
    dx : float
        Pixel size (m).
    dy : float, optional
        Pixel size along ``y``; defaults to ``dx``.
    weight : ndarray, optional
        Per-pixel confidence (e.g. fringe visibility).  If given, a **weighted**
        least-squares solve (preconditioned CG) is used, which correctly handles
        gradients that are only trustworthy inside the illuminated aperture.  If
        omitted, the fast unweighted Frankot-Chellappa FFT solution is returned.

    Returns
    -------
    ndarray, float
        Reconstructed wavefront phase ``phi`` (rad), mean removed.
    """
    if dy is None:
        dy = dx
    if weight is None:
        return _integrate_frankot_chellappa(grad_x, grad_y, dx, dy)
    return _integrate_weighted(grad_x, grad_y, weight, dx, dy)


# ---------------------------------------------------------------------------
# reconstruction result container
# ---------------------------------------------------------------------------
@dataclass
class Wavefront:
    """Result of a Talbot reconstruction.

    Attributes
    ----------
    phase : ndarray
        Reconstructed wavefront phase ``phi(x, y)`` (rad), piston removed.
    amplitude : ndarray
        Beam amplitude estimate at the detector plane (from the zero order).
    x, y : ndarray
        Coordinate meshes (m).
    dx : float
        Pixel size (m).
    lambda0 : float
        Wavelength (m).
    grad_x, grad_y : ndarray
        The demodulated phase gradients (rad/m).
    info : dict
        Diagnostics from :func:`fourier_fringe_gradients`.
    """

    phase: np.ndarray
    amplitude: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    lambda0: float
    grad_x: np.ndarray = _dc_field(repr=False, default=None)
    grad_y: np.ndarray = _dc_field(repr=False, default=None)
    info: dict = _dc_field(repr=False, default_factory=dict)

    @property
    def field(self):
        """Complex field ``amplitude * exp(i phase)`` at the detector plane."""
        return self.amplitude * np.exp(1j * self.phase)

    def _fit_weight(self):
        """Visibility weight restricted to the illuminated aperture.

        Uses the ``mask`` from the demodulation (if present) so the noisy /
        extrapolated exterior does not bias the low-order fits.
        """
        w = self.amplitude ** 2
        mask = self.info.get("mask")
        if mask is not None:
            w = w * mask
        return w

    def radius_of_curvature(self):
        """Best-fit spherical radius of curvature (m) of the wavefront.

        Fits the *measured gradients* to those of a sphere,
        ``d(phi)/dx = 2c x`` and ``d(phi)/dy = 2c y`` with ``c = pi/(lambda R)``,
        weighted by fringe visibility inside the aperture.  Fitting the
        gradients (rather than the integrated phase) makes focus-finding immune
        to any integration-boundary bias.

        Positive ``R`` means the wavefront diverges (a virtual source / focus
        upstream); negative means it converges to a focus downstream at ``|R|``.
        """
        if self.grad_x is None:
            # fall back to a phase fit if gradients were not retained
            rr = self.x ** 2 + self.y ** 2
            w = self._fit_weight()
            A = rr - np.average(rr, weights=w)
            b = self.phase - np.average(self.phase, weights=w)
            c = np.sum(w * A * b) / np.sum(w * A * A)
        else:
            w = self._fit_weight()
            num = np.sum(w * (self.grad_x * self.x + self.grad_y * self.y))
            den = np.sum(w * 2.0 * (self.x ** 2 + self.y ** 2))
            c = num / den
        if c == 0:
            return np.inf
        return np.pi / (self.lambda0 * c)

    def distance_to_focus(self):
        """Signed distance (m) from the detector plane to the beam focus.

        Equal to ``-radius_of_curvature`` for the sign convention where a
        converging beam focuses a positive distance downstream.
        """
        return -self.radius_of_curvature()

    def rms(self, remove_sphere=False):
        """RMS wavefront error (rad) over the illuminated aperture.

        If ``remove_sphere``, subtract the best-fit curvature first (i.e. report
        aberrations beyond defocus)."""
        w = self._fit_weight()
        phi = self.phase
        if remove_sphere:
            R = self.radius_of_curvature()
            if np.isfinite(R):
                sphere = np.pi * (self.x ** 2 + self.y ** 2) / (self.lambda0 * R)
                phi = phi - sphere
        phi = phi - np.average(phi, weights=w)
        return np.sqrt(np.average(phi ** 2, weights=w))


def reconstruct_wavefront(
    image,
    reference,
    dx,
    lambda0=None,
    energy=None,
    zT=None,
    period=None,
    dy=None,
    peak_radius_frac=0.4,
    visibility_frac=0.15,
):
    """Reconstruct the wavefront from a Talbot image and a reference image.

    Convenience wrapper that chains :func:`fourier_fringe_gradients` and
    :func:`integrate_gradients` and packages everything into a
    :class:`Wavefront`.

    Parameters
    ----------
    image : ndarray
        Measured Talbot detector image ``(N, M)``.
    reference : ndarray
        Reference image (known / flat wavefront), same geometry -- see
        :func:`fourier_fringe_gradients`.
    dx : float
        Detector pixel size (m).
    lambda0 : float, optional
        Wavelength (m).  Provide this or ``energy``.
    energy : float, optional
        Photon energy (eV); converted to ``lambda0`` if the latter is absent.
    zT : float
        Grating-to-detector distance (m).  Required.
    period : float, optional
        Grating period hint (m) to bracket the carrier search.
    dy : float, optional
        Pixel size along ``y``; defaults to ``dx``.
    peak_radius_frac, visibility_frac : float, optional
        Passed through to :func:`fourier_fringe_gradients`.

    Returns
    -------
    Wavefront
    """
    if lambda0 is None:
        if energy is None:
            raise ValueError("provide either lambda0 (m) or energy (eV)")
        lambda0 = float(wavelength_from_energy(energy))
    if zT is None:
        raise ValueError("zT (grating-to-detector distance, m) is required")

    grad_x, grad_y, info = fourier_fringe_gradients(
        image, reference, dx, lambda0, zT, period=period, dy=dy,
        peak_radius_frac=peak_radius_frac, visibility_frac=visibility_frac,
    )
    phase = integrate_gradients(grad_x, grad_y, dx, dy, weight=info["visibility"])
    x, y = _coords(image.shape, dx, dy)
    return Wavefront(
        phase=phase,
        amplitude=info["amplitude"],
        x=x,
        y=y,
        dx=dx,
        lambda0=lambda0,
        grad_x=grad_x,
        grad_y=grad_y,
        info=info,
    )
