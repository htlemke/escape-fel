"""Otti: the motorized detector arm on the THC lid, and what its detector sees.

The geometry is eco's ``calc_otti`` (``eco.endstations.bernina_sample_environments``):
a JUNGFRAU module (1024 x 512 pixels of 75 um) faces down at 179.287 mm
from the arm's pivot, which sits 46 mm to the side of and 77.436 mm above
the sample. Lab frame as there: x horizontal (towards the lid centre), y up,
z along the beam; azimuth = atan2(x, z), elevation above the horizontal
plane. The detector orientation is

    R = Ry(nu) Rx(del) Ry(det)

with ``nu`` (SARES20-EXP:MOT_RY) turning the arm about the vertical through
the pivot, ``del`` (MOT_RZ) tilting it, and ``det`` the detector's turn
about its own normal (encoder DET_RY plus a stored offset in eco).

``det`` is not driven: the detector hangs in a cardan (gimbal) mount on the
upward-pointing flange, so it follows from nu and del. Three models:

* ``"twist-free"`` - det = offset - nu, eco's ``otti_det="auto"``: the
  detector is carried along without turning about its normal (the tilt is
  a single rotation about a horizontal axis).
* ``"cardan"`` - a gimbal whose outer axis is fixed to the flange,
  horizontal at azimuth ``psi``. A gimbal never turns about the third axis,
  which is not the same as twist-free once both gimbal angles are non-zero:
  the detector turns by an extra ~del^2/2 * sin 2(nu - psi) (0.4 deg at
  del = 10, 4 deg at 30, 10 deg at 45). ``psi`` and the offset are not known
  yet; ``fit_cardan`` gets both from recorded (nu, del, encoder) triples.
* ``"encoder"`` - det as given (the measured value).

DEL_MAX = 44 deg is where the reach of this model matches the envelope from
the original thc_angles app (``data/otti.npy``, CAD/ray-traced) to 0.6 deg rms
over all azimuths; it is the tilt the lid allows, not a motor limit.
"""
from __future__ import annotations

import numpy as np

DISTANCE = 179.287                         # pivot to detector face [mm]
PIVOT = np.array([46.0, 77.436, 0.0])      # pivot relative to the sample [mm]
NX, NZ, PIXEL = 1024, 512, 0.075           # JUNGFRAU module
DEL_MAX = 44.0
SPIN_MODELS = ("twist-free", "cardan", "encoder")


def _rx(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.radians(a)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def cardan_det(nu, de, psi=0.0):
    """det (without offset) of a detector in a gimbal whose outer axis is
    horizontal at azimuth ``psi``: the detector normal n = Ry(nu) Rx(del) y is
    reached as R_u(a) R_v(b) y (u outer, v = y x u inner axis)."""
    y = np.array([0.0, 1.0, 0.0])
    n = _ry(nu) @ _rx(de) @ y
    u = _ry(psi) @ np.array([1.0, 0.0, 0.0])
    v = np.cross(y, u)
    b = np.degrees(np.arcsin(np.clip(n @ u, -1, 1)))
    a = np.degrees(np.arctan2(-(n @ v), n @ y))
    G = _rot(u, a) @ _rot(v, b)
    D = _rx(-de) @ _ry(-nu) @ G                # = Ry(det)
    return float(np.degrees(np.arctan2(D[0, 2], D[0, 0])))


def spin(nu, de, model="twist-free", psi=0.0, det=0.0, offset=0.0):
    """det [deg] under ``model``; ``det`` is the encoder value for "encoder"."""
    if model == "twist-free":
        return float(_wrap(offset - nu))
    if model == "cardan":
        return float(_wrap(offset + cardan_det(nu, de, psi)))
    if model == "encoder":
        return float(det)
    raise ValueError(f"spin model {model!r}: one of {SPIN_MODELS}")


def orientation(nu, de, det):
    return _ry(nu) @ _rx(de) @ _ry(det)


def pixel_positions(nu, de, det, step=1):
    """Lab positions (mm) of the pixel centres, shape (NX/step, NZ/step, 3)."""
    x = (np.arange(0, NX, step) - (NX - 1) / 2) * PIXEL
    z = (np.arange(0, NZ, step) - (NZ - 1) / 2) * PIXEL
    X, Z = np.meshgrid(x, z, indexing="ij")
    P = np.stack([X, np.full_like(X, DISTANCE), Z], -1)
    return P @ orientation(nu, de, det).T + PIVOT


def angles(P):
    """azimuth, elevation, 2 theta [deg] and distance [mm] of lab positions ``P`` (..., 3)."""
    P = np.asarray(P, float)
    r = np.linalg.norm(P, axis=-1)
    az = np.degrees(np.arctan2(P[..., 0], P[..., 2]))
    el = np.degrees(np.arctan2(P[..., 1], np.hypot(P[..., 0], P[..., 2])))
    tth = np.degrees(np.arccos(np.clip(P[..., 2] / r, -1, 1)))
    return az, el, tth, r


def outline(nu, de, det, n=40):
    """Lab positions of the module's edge, a closed loop starting at pixel (0, 0)."""
    s = np.linspace(-0.5, 0.5, n)
    w, h = NX * PIXEL, NZ * PIXEL
    edge = np.concatenate([np.stack([s * w, np.full(n, -h / 2)], 1), np.stack([np.full(n, w / 2), s * h], 1),
                           np.stack([-s * w, np.full(n, h / 2)], 1), np.stack([np.full(n, -w / 2), -s * h], 1)])
    edge = np.vstack([edge, edge[:1]])
    P = np.stack([edge[:, 0], np.full(len(edge), DISTANCE), edge[:, 1]], 1)
    return P @ orientation(nu, de, det).T + PIVOT


def summary(nu, de, det):
    """Numbers for one position: centre azimuth/elevation/2theta, ranges, distance."""
    P = pixel_positions(nu, de, det, step=8)
    az, el, tth, r = angles(P)
    normal = orientation(nu, de, det) @ np.array([0.0, 1.0, 0.0])
    caz, cel, ctth, cr = angles(PIVOT + DISTANCE * normal)
    # solid angle: pixel area * cos(incidence) / r^2, summed over the module
    omega = NX * NZ * PIXEL ** 2 * np.mean(np.abs(P @ normal) / r ** 3)
    return {"azimuth": float(caz), "elevation": float(cel), "2theta": float(ctth),
            "2theta min": float(tth.min()), "2theta max": float(tth.max()),
            "elevation min": float(el.min()), "elevation max": float(el.max()),
            "distance": float(cr), "solid angle [msr]": float(omega * 1e3)}


def reach(del_max=DEL_MAX, nu_step=2.0, bins=360):
    """Lowest elevation the module edge reaches per azimuth over all nu at
    del = del_max: (azimuth centres, elevation) - everything above is reachable."""
    az, el = [], []
    for nu in np.arange(-180, 180, nu_step):
        a, e, _, _ = angles(outline(nu, del_max, -nu, n=30))
        az.append(a)
        el.append(e)
    az, el = np.concatenate(az), np.concatenate(el)
    edges = np.linspace(-180, 180, bins + 1)
    idx = np.clip(np.digitize(az, edges) - 1, 0, bins - 1)
    low = np.full(bins, np.nan)
    np.fmin.at(low, idx, el)
    return 0.5 * (edges[1:] + edges[:-1]), low


def to_gamma_delta(az, el):
    """The THC app's (gamma, delta) plane, 0..180 each, as for data/otti.npy:
    azimuth >= 0 as is, azimuth < 0 folded to (-azimuth, 180 - elevation)."""
    az, el = np.asarray(az, float), np.asarray(el, float)
    neg = az < 0
    return np.where(neg, -az, az), np.where(neg, 180 - el, el)


def fit_cardan(nu, de, det_measured, psi_step=0.25):
    """Fit the cardan model to recorded positions: returns (psi, offset, rms)
    for det_measured ~ offset + cardan_det(nu, del, psi). det_measured is in
    eco's convention (otti_det readback + otti_det_rotation)."""
    nu, de, dm = (np.atleast_1d(np.asarray(a, float)) for a in (nu, de, det_measured))
    best = None
    for psi in np.arange(0, 180, psi_step):
        model = np.array([cardan_det(a, b, psi) for a, b in zip(nu, de)])
        d = np.radians(dm - model)
        offset = np.degrees(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
        rms = float(np.sqrt(np.mean(_wrap(dm - model - offset) ** 2)))
        if best is None or rms < best[2]:
            best = (float(psi), float(offset), rms)
    return best
