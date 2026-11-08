"""Reciprocal-space calculations behind eco's ``RecSpace``, on top of hklforge.

Pure numpy + hklforge, no eco, EPICS or file access, so everything here can
be tested and benchmarked on its own. eco's ``eco.utilities.recspace_hklforge``
wraps it into eco Assemblies (``eco.utilities.recspace_calc`` re-exports this
module). The geometry and frames are ``escape.bernina.geometry``'s.

Conventions:

- The lab frame is SwissFEL/Bernina's: z+ along the beam, y+ up, x+ = y x z
  (left, looking along the beam). The diffractometer is You's (1999) 4S+2D
  geometry (diffcalc's) written in that frame; angles mean exactly what they
  mean in diffcalc, only vectors and matrices differ, by ``YOU_TO_LAB``.
  The out-of-plane detector circle diffcalc calls ``nu`` is ``gamma`` here
  (the Bernina name), also inside the hklforge geometry.
- UB in the phi frame (the lab frame at all angles 0) with the 2 pi
  convention, q = UB h, abs(q) = 2 pi / d. A diffcalc UB is YOU_TO_LAB @ UB
  here (``from_you``).
- Stored reflection positions are lists in diffcalc's ``Position`` order
  (mu, delta, gamma, eta, chi, phi), energies in eV.
- Orientations are (hkl, xyz) with xyz a lab direction at the given position
  (all zeros if none), i.e. a phi-frame direction for ``position=None``.
- Reference vector default n_hkl = (1, 0, 0) (behind alpha, beta, psi, naz,
  tau), surface normal default n_phi = (1, 0, 0) (betain, betaout): diffcalc's
  ``UBCalculation`` defaults, the latter turned into this frame.

U from two orientations or one is hklforge's (``hklforge.orient``); the
reflection bookkeeping around it (references by index or tag, the
single-reflection refinement and the least-squares fit) lives here, written
to give diffcalc's numbers.
"""

from __future__ import annotations

import re

import numpy as np
from hklforge import Degenerate, ModeError
from hklforge import plan as _hkl_plan
from hklforge.geometry import wavevector
from hklforge.lattice import b_matrix
from hklforge.orient import _any_perpendicular, _rotation, u_from_one, u_from_two  # noqa: F401
from hklforge.virtual import virtual_angles

from .geometry import (BEAM, DETECTOR_CIRCLES, GEOMETRY, POSITION_ORDER, SAMPLE_CIRCLES,  # noqa: F401
                       UP, YOU_BEAM, YOU_TO_LAB, YOU_UP, _turn, en2lam, frame_rotation, from_you,
                       lam2en, to_you, turn_crystal_data)

# rows of a solution table: the six circles, then the virtual angles
VIRTUAL_ORDER = ("theta", "ttheta", "qaz", "alpha", "naz", "tau", "psi", "beta",
                 "betain", "betaout", "omega")

# constraint keys taking a boolean (in the diffcalc constraint dict they are
# None or True)
BOOL_CONSTRAINTS = ("a_eq_b", "bin_eq_bout", "bisect")
DETECTOR_POLICIES = ("h1", "between")

DEFAULT_REFERENCE = {"hkl": [1, 0, 0]}
DEFAULT_SURFACE = {"phi": [1, 0, 0]}   # diffcalc's phi (0, 0, 1), in this frame

SMALL = 1e-4  # diffcalc's (Vlieg's) threshold for parallel references


# --- units and frames --------------------------------------------------------


def position_to_angles(position) -> dict:
    """[mu, delta, gamma, eta, chi, phi] (diffcalc order) -> {name: degrees}."""
    return {n: float(v) for n, v in zip(POSITION_ORDER, position)}


def angles_to_position(angles: dict) -> list:
    return [float(angles[n]) for n in POSITION_ORDER]


def q_phi_unit(angles: dict) -> np.ndarray:
    """Scattering vector in the phi frame in units of abs(k), diffcalc's ``get_q_phi``."""
    return GEOMETRY.q_phi(angles, 1.0)


def vector_phi(spec, ub) -> np.ndarray | None:
    """Unit phi-frame vector of a reference/surface spec.

    ``spec`` is ``{"hkl": [h, k, l]}`` (a reciprocal-lattice direction, turned
    with UB, so it follows the crystal) or ``{"phi": [x, y, z]}`` (fixed in
    the phi frame, i.e. to the sample stage).
    """
    if not spec:
        return None
    if spec.get("hkl") is not None:
        v = np.asarray(ub, float) @ np.asarray(spec["hkl"], float)
    else:
        v = np.asarray(spec["phi"], float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def vector_hkl(spec, ub) -> np.ndarray | None:
    """The same spec expressed as (unnormalised) Miller indices."""
    if not spec:
        return None
    if spec.get("hkl") is not None:
        return np.asarray(spec["hkl"], float)
    return np.linalg.solve(np.asarray(ub, float), np.asarray(spec["phi"], float))


def _up() -> np.ndarray:
    return np.asarray(UP, float)


def _axis_name(v) -> str:
    v = np.asarray(v, float)
    i = int(np.argmax(np.abs(v)))
    if np.isclose(abs(v[i]), 1.0):
        return f"{'xyz'[i]}{'+' if v[i] > 0 else '-'}"
    return "(" + ", ".join(f"{x:.3g}" for x in v) + ")"


def parse_triple(text) -> list | None:
    """Three numbers from text as people write Miller indices or vectors:
    "0 0 1", "0,0,1", "(0,0,1)", "[1 -1 0]", or "(001)" / "1-10" (one digit
    each). None if the text is not that."""
    t = str(text).strip().strip("()[]{} ").strip()
    if not t:
        return None
    parts = [x for x in re.split(r"[,\s;]+", t) if x]
    if len(parts) == 1 and re.fullmatch(r"(-?\d){3}", t):
        parts = re.findall(r"-?\d", t)
    if len(parts) != 3:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def lab_direction(spec) -> list:
    """A lab direction at all angles 0 (the diffractometer's own frame).

    ``spec`` is a 3-vector, an axis ('x+', '-z', ...), or a name: 'beam'
    (the incoming beam), 'up'/'down', or a circle ('phi', '-phi', 'chi',
    ...: its rotation axis at all angles 0). Returns a unit 3-vector list.
    """
    if isinstance(spec, str) and parse_triple(spec) is not None:
        spec = parse_triple(spec)
    if not isinstance(spec, str):
        v = np.asarray(spec, float)
        if v.shape != (3,) or not np.linalg.norm(v):
            raise ValueError(f"direction {spec!r}: give 3 numbers or a name like 'beam', 'up', 'phi', 'z-'")
        return (v / np.linalg.norm(v) + 0.0).tolist()
    s = spec.strip().lower()
    name = s.strip("+- ")
    sign = -1.0 if "-" in s else 1.0
    circles = {c.name: c.axis for c in GEOMETRY.sample + GEOMETRY.detector}
    if name in ("x", "y", "z"):
        v = np.eye(3)["xyz".index(name)]
    elif name == "beam":
        v = np.asarray(GEOMETRY.beam, float)
    elif name in ("up", "vertical"):
        v = _up()
    elif name == "down":
        v = -_up()
    elif name in circles:
        v = np.asarray(circles[name], float)
    else:
        raise ValueError(f"direction {spec!r}: use 'beam', 'up', 'down', an axis like 'z-', "
                         f"a circle ({', '.join(circles)}) or 3 numbers")
    return (sign * v + 0.0).tolist()


def frame_description() -> str:
    """The diffractometer frame at all angles 0, in words, from GEOMETRY."""
    up, beam = _up(), np.asarray(GEOMETRY.beam, float)
    left = np.cross(up, beam)       # looking along the beam
    rows = [("up", up), ("along the incoming X-ray beam", beam), ("left, looking along the beam", left)]
    rows.sort(key=lambda r: int(np.argmax(np.abs(r[1]))))
    lines = ["Diffractometer frame at all angles 0:"]
    lines += [f"  {_axis_name(v):4} {text}" for text, v in rows]
    last = GEOMETRY.sample[-1]
    lines.append(f"  last sample rotation {last.name}: axis along {_axis_name(last.axis)} "
                 f"(positive {last.name} turns right-handed about it)")
    lines.append("  sample circles, outermost first: "
                 + ", ".join(f"{c.name} {_axis_name(c.axis)}" for c in GEOMETRY.sample))
    order = ", then ".join(c.name for c in reversed(GEOMETRY.sample))
    lines.append(f"  a position is reached from all angles 0 by turning {order},")
    lines.append("  each about its axis as listed (the innermost circle turns the crystal first)")
    return "\n".join(lines)


# --- UB from references --------------------------------------------------------


def lattice_b(unit_cell: dict) -> np.ndarray:
    return b_matrix(*(float(unit_cell[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")))


def reflection_vector(reflection: dict) -> np.ndarray:
    """Phi-frame direction of a stored reflection's measured Q."""
    return q_phi_unit(position_to_angles(reflection["position"]))


def orientation_vector(orientation: dict) -> np.ndarray:
    """Phi-frame direction of a stored orientation's xyz."""
    pos = orientation.get("position")
    xyz = np.asarray(orientation["xyz"], float)
    if pos is None:
        return xyz
    return GEOMETRY.sample_matrix(position_to_angles(pos)).T @ xyz


def find_reference(reflections, orientations, idx):
    """diffcalc's lookup: an int is a reflection index, falling back to the
    orientation with that index if there are not enough reflections; a str
    is a tag, searched in the reflections first. Returns (hkl, phi direction,
    description)."""
    if isinstance(idx, str):
        for kind, items, vec in (("reflection", reflections, reflection_vector),
                                 ("orientation", orientations, orientation_vector)):
            for i, item in enumerate(items):
                if item.get("tag") == idx:
                    return item["hkl"], vec(item), f"{kind} {i} ({idx})"
        raise ValueError(f"no reflection or orientation tagged {idx!r}")
    if idx < len(reflections):
        return reflections[idx]["hkl"], reflection_vector(reflections[idx]), f"reflection {idx}"
    if idx < len(orientations):
        return orientations[idx]["hkl"], orientation_vector(orientations[idx]), f"orientation {idx}"
    raise ValueError(f"no reflection or orientation with index {idx} "
                     f"({len(reflections)} reflections, {len(orientations)} orientations)")


def calc_u(unit_cell, reflections, orientations, idx1=0, idx2=1, U0=None):
    """U from two references (``idx2=None``: from reflection ``idx1`` alone,
    as the smallest change of ``U0``, see ``u_from_one``).

    Returns (U, description of what was used).
    """
    B = lattice_b(unit_cell)
    h1, v1, d1 = find_reference(reflections, orientations, idx1)
    if idx2 is None:
        start = "the current U" if U0 is not None and len(U0) else "U = 1"
        return u_from_one(B, h1, v1, U0), f"{d1} only (rotation about it kept from {start})"
    h2, v2, d2 = find_reference(reflections, orientations, idx2)
    return u_from_two(B, h1, v1, h2, v2), f"{d1} and {d2}"


def fit_ub(reflections, indices=None):
    """Least-squares UB from 3 or more reflections, lattice unconstrained.

    Same algorithm as diffcalc's ``_fit_ub_uncon`` (which is what the eco
    diffcalc wrapper always gets, since it passes all six lattice parameters
    and diffcalc then treats the crystal as triclinic). Returns (U, lattice
    dict with a, b, c, alpha, beta, gamma, rms residual of q in 1/angstrom).
    """
    if indices is None:
        indices = range(len(reflections))
    refls = [reflections[i] for i in indices]
    if len(refls) < 3:
        raise ValueError("Need at least 3 reference reflections to fit UB matrix.")
    X = np.array([r["hkl"] for r in refls], float)
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError("The reflections' hkl are coplanar, which leaves UB undetermined; "
                         "add a reflection out of their plane.")
    Y =np.array([reflection_vector(r) * wavevector(r["energy"]) for r in refls])
    Bt, *_ = np.linalg.lstsq(X, Y, rcond=None)      # rows: a*, b*, c* in the phi frame
    b1, b2, b3 = Bt
    e1 = b1 / np.linalg.norm(b1)
    e2 = b2 - e1 * (b2 @ e1)
    e2 /= np.linalg.norm(e2)
    e3 = b3 - e1 * (b3 @ e1) - e2 * (b3 @ e2)
    e3 /= np.linalg.norm(e3)
    U = np.column_stack([e1, e2, e3])
    V = b1 @ np.cross(b2, b3)
    a1, a2, a3 = (2 * np.pi * np.cross(x, y) / V for x, y in ((b2, b3), (b3, b1), (b1, b2)))
    la, lb, lc = (float(np.linalg.norm(x)) for x in (a1, a2, a3))
    ang = lambda x, y, nx, ny: float(np.degrees(np.arccos(np.clip(x @ y / (nx * ny), -1, 1))))
    lattice = {"a": la, "b": lb, "c": lc,
               "alpha": ang(a2, a3, lb, lc), "beta": ang(a1, a3, la, lc), "gamma": ang(a1, a2, la, lb)}
    rms = float(np.sqrt(np.mean(np.sum((X @ Bt - Y) ** 2, axis=1))))
    return U, lattice, rms


def refine_from_reflection(unit_cell, U, ub, hkl, angles, energy):
    """What one measured reflection says about lattice and orientation.

    Returns (scaled unit cell or None, miscut rotation matrix or None,
    scale, miscut angle in degrees), following diffcalc's ``refine_ub``:
    the lattice constants along non-zero indices are scaled by
    d_measured / d_calculated, and U is rotated by the smallest rotation
    taking UB h onto the measured Q (accurate up to a rotation about Q).
    """
    q = q_phi_unit(angles)
    d_meas = en2lam(energy) / np.linalg.norm(q)
    d_calc = 2 * np.pi / np.linalg.norm(lattice_b(unit_cell) @ np.asarray(hkl, float))
    scale = float(d_meas / d_calc)
    new_cell = None
    if abs(scale - 1) >= SMALL:
        new_cell = dict(unit_cell)
        for key, idx in zip(("a", "b", "c"), hkl):
            if abs(idx) > SMALL:
                new_cell[key] = unit_cell[key] * scale
    hphi = np.asarray(ub, float) @ np.asarray(hkl, float)
    axis = np.cross(hphi, q)
    if np.linalg.norm(axis) < SMALL:
        return new_cell, None, scale, 0.0
    angle = np.arccos(np.clip(hphi @ q / (np.linalg.norm(hphi) * np.linalg.norm(q)), -1, 1))
    return new_cell, _rotation(axis, angle), scale, float(np.degrees(angle))


# --- solving -----------------------------------------------------------------

_PLANS: dict = {}


def make_mode(constraints: dict) -> tuple[dict, str]:
    """diffcalc-style constraint dict (None = free) -> (hklforge mode, detector policy).

    Extra keys over diffcalc: ``h2`` (a second reflection kept in Bragg
    condition, list of 3 or None) and ``h2_detector`` ("h1": detector on the
    first reflection, "between": on the bisector of both outgoing beams).
    """
    mode, detector = {}, "h1"
    for key, val in constraints.items():
        if key == "h2_detector":
            detector = val or "h1"
            continue
        if val is None or val == "":
            continue
        key = "gamma" if key == "nu" else key
        if key in BOOL_CONSTRAINTS:
            if val:
                mode[key] = True
        elif key == "h2":
            if len(val):
                mode["h2"] = tuple(float(x) for x in val)
        else:
            mode[key] = float(val)
    if "h2" not in mode:
        detector = "h1"
    return mode, detector


def get_plan(mode: dict, detector: str = "h1"):
    key = (tuple(sorted((n, repr(v)) for n, v in mode.items())), detector)
    if key not in _PLANS:
        _PLANS[key] = _hkl_plan(GEOMETRY, mode, detector)
    return _PLANS[key]


def solve(ub, energy, hkl, constraints: dict, reference=DEFAULT_REFERENCE,
          surface=DEFAULT_SURFACE) -> list[dict]:
    """Every position putting ``hkl`` in Bragg condition under ``constraints``.

    Each solution is a dict of the six circles plus all virtual angles, in
    degrees, circles wrapped to (-180, 180]. Raises ``ModeError`` for a mode
    that does not fix a finite set of positions.
    """
    mode, detector = make_mode(constraints)
    pl = get_plan(mode, detector)
    n_ref, n_surf = vector_phi(reference, ub), vector_phi(surface, ub)
    k = wavevector(energy)
    out = []
    for a in pl.run(ub, k, hkl, n_surf, n_ref):
        row = {n: a[n] for n in POSITION_ORDER}
        row.update(virtual_angles(GEOMETRY, a, n_surf, n_ref))
        out.append(row)
    return out


def calc_hkl(ub, energy, angles: dict) -> np.ndarray:
    return np.linalg.solve(np.asarray(ub, float), GEOMETRY.q_phi(angles, wavevector(energy)))


def pseudo_angles(ub, angles: dict, reference=DEFAULT_REFERENCE, surface=DEFAULT_SURFACE) -> dict:
    return virtual_angles(GEOMETRY, angles, vector_phi(surface, ub), vector_phi(reference, ub))


def spots(ub, energy, angles: dict, hkls, width=77.0, height=77.0, margin=0.9) -> dict:
    """Outgoing beams of several reflections at one position, their largest
    separation and the largest detector distance keeping them all on a
    ``width`` x ``height`` face (``tdet_max``, in the units of width)."""
    from hklforge import Diffractometer

    d = Diffractometer(GEOMETRY, ub, energy=energy)
    return d.spots(angles, hkls, width=width, height=height, margin=margin)


def wrap_into_limits(value, limits, strict=True):
    """``value`` or an equivalent value +-360 n inside ``limits``, else None.

    hklforge wraps free circles to (-180, 180]; diffcalc may return e.g.
    190 for the same position. A motor with limits (0, 360) can drive both,
    so a solution is not rejected just for its representation.
    """
    lo, hi = limits
    for n in (0, -1, 1, -2, 2):
        v = value + 360.0 * n
        if (lo < v < hi) if strict else (lo <= v <= hi):
            return v
    return None


__all__ = [
    "GEOMETRY", "POSITION_ORDER", "VIRTUAL_ORDER", "ModeError", "Degenerate",
    "en2lam", "lam2en", "position_to_angles", "angles_to_position", "q_phi_unit",
    "vector_phi", "vector_hkl", "lattice_b", "calc_u", "fit_ub", "refine_from_reflection",
    "make_mode", "solve", "calc_hkl", "pseudo_angles", "spots", "wrap_into_limits",
]


def in_ranges(angles: dict, ranges: dict) -> bool:
    """Whether every circle named in ``ranges`` ({name: (lo, hi)}) lies in its
    range, allowing the same +-360 representations as ``wrap_into_limits``."""
    return all(wrap_into_limits(float(angles[n]), r, strict=False) is not None for n, r in ranges.items())
