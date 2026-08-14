"""X-ray Talbot wavefront sensing for ``escape``.

A compact, didactic implementation of single-grating (Talbot) wavefront sensing
for X-ray FEL beams, distilled from M. Seaberg's ``lcls_beamline_toolbox`` and
rewritten around ``escape`` / dask conventions.

Two halves:

* :mod:`escape.wavefront.talbot` -- forward simulation of Talbot self-images and
  reconstruction of the wavefront from a detector image (Fourier-fringe
  demodulation + Frankot-Chellappa gradient integration).
* :mod:`escape.wavefront.propagation` -- angular-spectrum free-space propagation,
  including :func:`~escape.wavefront.propagation.propagate_array` to
  back-propagate a whole run's worth of reconstructed fields lazily over dask.

Quick start::

    import numpy as np
    from escape import wavefront as wf

    lam = wf.wavelength_from_energy(9500.0)          # eV -> m
    zT = wf.talbot_distance(period := 1e-6, lam, fraction=1/8)

    # simulate a defocused beam's Talbot image ...
    phi_in = wf.talbot.parabolic_phase((512, 512), 5e-6, radius=-3.0, lambda0=lam)
    img = wf.simulate_talbot_image((512, 512), 5e-6, period, lam, zT,
                                   incident_phase=phi_in)

    # ... and reconstruct it
    w = wf.reconstruct_wavefront(img, dx=5e-6, period=period, lambda0=lam, zT=zT)
    print(w.distance_to_focus())
"""

from . import talbot
from . import propagation

from .talbot import (
    wavelength_from_energy,
    checkerboard_grating,
    mesh_grating,
    talbot_distance,
    simulate_talbot_image,
    parabolic_phase,
    fourier_fringe_gradients,
    integrate_gradients,
    reconstruct_wavefront,
    Wavefront,
)
from .propagation import (
    energy_from_wavelength,
    fresnel_number,
    angular_spectrum_propagate,
    propagate_array,
)

__all__ = [
    "talbot",
    "propagation",
    "wavelength_from_energy",
    "energy_from_wavelength",
    "fresnel_number",
    "checkerboard_grating",
    "mesh_grating",
    "talbot_distance",
    "simulate_talbot_image",
    "parabolic_phase",
    "fourier_fringe_gradients",
    "integrate_gradients",
    "reconstruct_wavefront",
    "Wavefront",
    "angular_spectrum_propagate",
    "propagate_array",
]
