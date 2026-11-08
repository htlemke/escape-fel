"""Diffractometer angles for Bernina's You (1999) six-circle geometry, on hklforge,
for a ``Sample`` given by its cell and orientations (the prep apps' view;
``escape.bernina.recspace`` has the same calculations in the form eco's
``RecSpace`` stores a crystal: reflections, positions, UB fits).

Conventions (identical to eco's recspace):

- **Lab frame**: SwissFEL/Bernina axes, z+ along the beam, y+ up,
  x+ = y x z (left, looking along the beam). Everything a user types or reads
  is in this frame; there is no second (diffcalc) frame any more.
- **Circles**: mu, eta, chi, phi (sample), gamma, delta (detector). gamma is
  the out-of-plane detector circle diffcalc calls ``nu``.
- **UB** maps (h, k, l) to Q in the phi frame (the lab frame with all angles
  at 0), 2 pi convention: abs(UB h) = 2 pi / d, abs(k) = 2 pi / lambda.
- **Orientations** ``(h, k, l) > (x, y, z)``: that crystal direction points
  along that lab direction with all angles at 0. Two fix U, one fixes it up
  to a rotation about that direction.
- **Surface normal** (behind betain, betaout, bin_eq_bout) and **reference
  vector** (behind alpha, beta, psi, naz, tau) are separate (h, k, l)
  directions. The surface normal is optional; the reference defaults to
  (1 0 0), as in eco and diffcalc. alpha/beta are angles to the *reference*,
  betain/betaout to the *surface*.
- **Two reflections**: the ``h2`` constraint keeps a second reflection in
  Bragg condition; ``detector`` says where the detector goes then:
  ``"h1"`` on the first (requested) reflection, ``"between"`` halfway
  between both outgoing beams. The sample angles are shared; the detector
  circles and the virtual angles that follow the outgoing beam
  (``PER_REFLECTION``) are given for each reflection the detector is not on
  (``reflection_angles``). alpha, betain and naz depend on the sample only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hklforge import Degenerate, Diffractometer
from hklforge import plan as _hkl_plan
from hklforge.geometry import wavevector
from hklforge.lattice import angle_between as _angle_between
from hklforge.lattice import b_matrix
from hklforge.orient import u_from_one, u_from_two
from hklforge.virtual import virtual_angles

from .geometry import (BEAM, CIRCLES, GEOMETRY, LEGACY_NAMES, SAMPLE_CIRCLES, UP,  # noqa: F401
                       YOU_TO_LAB, en2lam, k_magnitude, parse_direction, parse_vector, to_you)

VIRTUAL_CONSTRAINTS = ("qaz", "naz", "alpha", "beta", "psi", "betain", "betaout", "omega")
BOOL_CONSTRAINTS = ("a_eq_b", "bin_eq_bout", "bisect")
CONSTRAINTS = CIRCLES + VIRTUAL_CONSTRAINTS + BOOL_CONSTRAINTS
SURFACE_CONSTRAINTS = ("betain", "betaout", "bin_eq_bout")
DETECTOR_POLICIES = {"h1": "detector on reflection 1", "between": "detector between both"}
# with two reflections: what differs between them (the rest only depends on the sample)
PER_REFLECTION = ("gamma", "delta", "ttheta", "theta", "qaz", "beta", "betaout", "psi", "tau", "omega")
ROW_ORDER = CIRCLES + ("theta", "ttheta", "qaz", "naz", "alpha", "beta", "psi", "tau",
                       "betain", "betaout", "omega")
UNSUPPORTED = ("mu_is_gam",)

DEFAULT_REFERENCE = (1.0, 0.0, 0.0)
SMALL = 1e-4

# --- orientations -------------------------------------------------------------------


def parse_orientations(specs) -> list[tuple[tuple, tuple]]:
    """['(1,1,1) > (1,0,0)', '(0,0,1) > beam', ...] -> [(hkl, xyz), ...];
    blank lines and lines without '>' are skipped."""
    out = []
    for spec in specs:
        if ">" not in spec:
            continue
        hkl_s, xyz_s = spec.split(">", 1)
        out.append((parse_vector(hkl_s), parse_direction(xyz_s)))
    return out


# --- UB from orientations ----------------------------------------------------------


_u_from_two = u_from_two          # names used before the move to hklforge.orient
_u_from_one = u_from_one


@dataclass
class Sample:
    """A crystal mounted on the diffractometer.

    ``cell`` = dict(a, b, c, alpha, beta, gamma) in angstrom and degrees;
    ``orientations`` = [(hkl, lab xyz), ...] at all angles 0 (the first two
    are used); ``surface`` and ``reference`` are (h, k, l) or None.
    """
    cell: dict
    orientations: list
    surface: tuple | None = None
    reference: tuple | None = DEFAULT_REFERENCE
    B: np.ndarray = field(init=False, repr=False)
    U: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        c = self.cell
        self.B = b_matrix(*(float(c[k]) for k in ("a", "b", "c", "alpha", "beta", "gamma")))
        if not self.orientations:
            raise ValueError('no orientation: give at least one line "(h,k,l) > (x,y,z)"')
        (h1, v1), *rest = self.orientations
        self.U = _u_from_two(self.B, h1, v1, *rest[0]) if rest else _u_from_one(self.B, h1, v1)

    @classmethod
    def from_text(cls, cell: dict, orientations: str, surface: str = "", reference: str = "(1,0,0)"):
        """From the app's text fields; an empty surface/reference means none."""
        lines = [ln.strip() for ln in str(orientations).splitlines() if ln.strip()]
        surf = parse_vector(surface) if str(surface).strip() else None
        ref = parse_vector(reference) if str(reference).strip() else None
        return cls(dict(cell), parse_orientations(lines), surf, ref)

    @property
    def UB(self) -> np.ndarray:
        return self.U @ self.B

    def _phi(self, hkl):
        if hkl is None:
            return None
        v = self.UB @ np.asarray(hkl, float)
        return v / np.linalg.norm(v)

    @property
    def n_surface(self):
        """Unit surface normal in the phi frame, or None."""
        return self._phi(self.surface)

    @property
    def n_reference(self):
        return self._phi(self.reference)

    def d_spacing(self, hkl) -> float:
        return 2 * np.pi / np.linalg.norm(self.B @ np.asarray(hkl, float))

    # --- forward calculations ---

    def hkl_at(self, angles: dict, energy: float) -> np.ndarray:
        """(h, k, l) in Bragg condition at these angles."""
        return np.linalg.solve(self.UB, GEOMETRY.q_phi(_complete(angles), wavevector(energy)))

    def pseudo_angles(self, angles: dict) -> dict:
        return virtual_angles(GEOMETRY, _complete(angles), self.n_surface, self.n_reference)

    def spots(self, angles: dict, hkls, energy: float, width=77.0, height=77.0) -> dict:
        """Outgoing beams of several reflections (see hklforge ``Diffractometer.spots``)."""
        d = Diffractometer(GEOMETRY, self.UB, energy=energy)
        return d.spots(_complete(angles), hkls, width=width, height=height)

    # --- solving ---

    def solve(self, hkl, energy: float, constraints: dict, detector: str = "h1") -> pd.DataFrame:
        """Every position putting ``hkl`` in Bragg condition.

        ``constraints``: {name: value} of the held ones (circles, virtual
        angles, booleans True, ``h2``: a second (h, k, l)). Returns a
        DataFrame with one column per solution ('sol. 0', ...) and the angles
        as rows (``ROW_ORDER``); empty if there is none.
        """
        mode = make_mode(constraints)
        if any(k in mode for k in SURFACE_CONSTRAINTS) and self.surface is None:
            raise ValueError("betain, betaout and bin_eq_bout need a surface normal (sample tab)")
        if "h2" not in mode:
            detector = "h1"
        pl = _plan(mode, detector)
        k = wavevector(energy)
        try:
            positions = pl.run(self.UB, k, tuple(float(x) for x in hkl), self.n_surface, self.n_reference)
        except Degenerate as e:
            raise ValueError(f"({' '.join(f'{x:g}' for x in hkl)}) has a continuum of solutions under "
                             f"these constraints ({e}); hold another angle instead") from None
        rows = []
        for a in positions:
            row = {n: float(a[n]) for n in CIRCLES}
            row.update(virtual_angles(GEOMETRY, a, self.n_surface, self.n_reference))
            rows.append(row)
        df = pd.DataFrame(rows, index=[f"sol. {i}" for i in range(len(rows))]).T
        if df.empty:
            return df
        ordered = [r for r in ROW_ORDER if r in df.index] + [r for r in df.index if r not in ROW_ORDER]
        return df.reindex(ordered).astype(float)


_PLANS: dict = {}


def make_mode(constraints: dict) -> dict:
    """{name: value or None/False} -> hklforge mode (held constraints only).
    Accepts the diffcalc-era names (gam, nu); refuses unsupported ones."""
    mode = {}
    for key, val in constraints.items():
        key = LEGACY_NAMES.get(key, key)
        if key in UNSUPPORTED:
            if val:
                raise ValueError(f"{key} is not supported by hklforge; hold gamma and mu instead")
            continue
        if val is None or val is False or (isinstance(val, str) and val == ""):
            continue
        if key in BOOL_CONSTRAINTS:
            mode[key] = True
        elif key == "h2":
            mode["h2"] = tuple(float(x) for x in val)
        elif key in CIRCLES or key in VIRTUAL_CONSTRAINTS:
            mode[key] = float(val)
        else:
            raise ValueError(f"unknown constraint {key!r}")
    return mode


def _plan(mode: dict, detector: str):
    key = (tuple(sorted((n, repr(v)) for n, v in mode.items())), detector)
    if key not in _PLANS:
        _PLANS[key] = _hkl_plan(GEOMETRY, mode, detector)
    return _PLANS[key]


def _complete(angles: dict) -> dict:
    return {n: float(angles.get(n, 0.0)) for n in CIRCLES}


def solution_angles(df: pd.DataFrame, column) -> dict:
    """The six circles of one solution column as {name: float}."""
    col = df.columns[column] if isinstance(column, int) else column
    s = df[col]
    return {n: (float(s[n]) if n in s.index and pd.notna(s[n]) else 0.0) for n in CIRCLES}


def detector_towards(kout, near: dict | None = None) -> dict:
    """gamma and delta that point the detector arm along ``kout`` (lab frame,
    any length): of the two settings, the one closest to ``near``'s circles."""
    return GEOMETRY.detector_towards(kout, near)


def reflection_angles(sample: Sample, angles: dict, hkl, energy: float) -> dict:
    """The ``PER_REFLECTION`` angles of reflection ``hkl`` at these sample
    angles: where the detector has to go to sit on it, and the virtual
    angles of its outgoing beam (as if the detector were there)."""
    a = _complete(angles)
    q = GEOMETRY.sample_matrix(a) @ sample.UB @ np.asarray(hkl, float) / wavevector(energy)
    det = detector_towards(np.asarray(GEOMETRY.beam, float) + q, near=a)
    va = virtual_angles(GEOMETRY, {**a, **det}, sample.n_surface, sample.n_reference)
    va.update(det)
    return {n: va[n] for n in PER_REFLECTION if n in va}


def angle_between(sample: Sample, h1, h2) -> float:
    """Angle between the scattering vectors of two reflections [deg]."""
    return _angle_between(sample.UB, h1, h2)


# --- filters ---------------------------------------------------------------------


def wrap_into(value, lo, hi):
    """``value`` or value +- 360 n inside [lo, hi], else None (a circle at
    190 is the same position as at -170)."""
    for n in (0, -1, 1, -2, 2):
        v = value + 360.0 * n
        if lo <= v <= hi:
            return v
    return None


def passes(solution, ranges: dict) -> bool:
    """Whether a solution (Series or dict) lies in {name: (min or None, max
    or None)}. A circle with both bounds may be shifted by 360 degrees, as
    eco's solution ranges; anything else is compared as it is."""
    for name, (lo, hi) in ranges.items():
        name = LEGACY_NAMES.get(name, name)
        v = solution.get(name)
        if v is None or not np.isfinite(v):
            continue
        if name in CIRCLES and lo is not None and hi is not None:
            if wrap_into(float(v), lo, hi) is None:
                return False
        elif (lo is not None and v < lo) or (hi is not None and v > hi):
            return False
    return True


# --- many reflections --------------------------------------------------------------


def reflection_table(sample: Sample, constraints: dict, reflections, energy: float,
                     ranges: dict | None = None, solution: int = 0, detector: str = "h1",
                     progress=None) -> pd.DataFrame:
    """One row per (h, k, l): solution number ``solution`` among those passing
    ``ranges``; an 'error' column says why a row has none. ``progress(i, n)``
    is called before each reflection."""
    rows = []
    for i, hkl in enumerate(reflections):
        if progress is not None:
            progress(i, len(reflections))
        try:
            df = sample.solve(hkl, energy, constraints, detector)
            cols = [c for c in df.columns if passes(df[c], ranges or {})]
            if not cols:
                row = {"error": "no solution" if df.empty else f"none of {len(df.columns)} passes the filters"}
            else:
                row = df[cols[min(solution, len(cols) - 1)]].to_dict()
                row["solutions"] = len(cols)
                row["error"] = ""
        except Exception as e:
            row = {"error": str(e)}
        rows.append(row)
    index = pd.MultiIndex.from_tuples([tuple(r) for r in reflections], names=["h", "k", "l"])
    return pd.DataFrame(rows, index=index)


# --- the detector ----------------------------------------------------------------

# detector sizes and module layouts: escape.bernina.detectors
from .detectors import (DETECTOR_MODULES, DETECTOR_PRESETS, JF_MODULE, JF_PIXEL,  # noqa: E402,F401
                        JUNGFRAU_LAYOUTS, jungfrau_modules, on_modules)


def _detector_frame(angles):
    D = GEOMETRY.detector_matrix(_complete(angles))
    return D @ np.asarray(BEAM), D @ np.array([1.0, 0, 0]), D @ np.array([0, 1.0, 0])


def detector_map(sample: Sample, angles: dict, energy: float, distance: float,
                 width: float = 77.0, height: float = 77.0, n: int = 41) -> dict:
    """h, k, l, abs(Q) and 2 theta over the detector face at these angles:
    arrays (n_v, n_u) plus the face coordinates u (width) and v (height) in mm.

    Each point of the face sees Q = k' - k of the beam scattered towards it;
    the crystal is in Bragg condition only where that Q is a lattice vector.
    """
    nx = max(2, int(round(n * width / max(width, height))))
    ny = max(2, int(round(n * height / max(width, height))))
    arm, ex, ey = _detector_frame(angles)
    u = np.linspace(-width / 2, width / 2, nx)
    v = np.linspace(-height / 2, height / 2, ny)
    U, V = np.meshgrid(u, v)
    P = distance * arm + U[..., None] * ex + V[..., None] * ey
    k = wavevector(energy)
    kout = P / np.linalg.norm(P, axis=-1, keepdims=True)
    q_lab = k * (kout - np.asarray(BEAM))
    S = GEOMETRY.sample_matrix(_complete(angles))
    q_phi = q_lab @ S                     # each row S^T q
    hkl = np.linalg.solve(sample.UB, q_phi.reshape(-1, 3).T).T.reshape(q_phi.shape)
    tth = np.degrees(np.arccos(np.clip(kout @ np.asarray(BEAM), -1, 1)))
    return {"h": hkl[..., 0], "k": hkl[..., 1], "l": hkl[..., 2],
            "Q": np.linalg.norm(q_lab, axis=-1), "ttheta": tth, "u": u, "v": v}


def detector_summary(m: dict) -> pd.DataFrame:
    """min/max of h, k, l, abs(Q) [1/A] and 2 theta over the face."""
    return pd.DataFrame({x: [float(np.min(m[x])), float(np.max(m[x]))] for x in ("h", "k", "l", "Q", "ttheta")},
                        index=["min", "max"])


def reflections_on_detector(sample: Sample, angles: dict, energy: float, distance: float,
                            width: float = 77.0, height: float = 77.0, tolerance: float = 0.005,
                            allowed=None) -> pd.DataFrame:
    """Integer reflections close to Bragg condition at these angles whose
    outgoing beam lands on the detector.

    ``tolerance``: allowed distance of the lattice point from the Ewald
    sphere as a fraction of abs(k) (bandwidth, mosaicity, divergence; 0.005
    is a 0.5 % band). ``allowed(hkl) -> bool`` drops forbidden reflections.
    Columns: position on the face (u, v in mm), excitation error, d, 2 theta.
    """
    k = wavevector(energy)
    S = GEOMETRY.sample_matrix(_complete(angles))
    arm, ex, ey = _detector_frame(angles)
    beam = np.asarray(BEAM)
    qmax = 2 * k * (1 + tolerance)            # the limiting sphere
    lim = [int(np.ceil(qmax / np.linalg.norm(sample.UB @ e))) for e in np.eye(3)]
    if np.prod([2 * m + 1 for m in lim]) > 3e6:
        raise ValueError("too many lattice points below 2 theta = 180 for this check (large cell, high energy)")
    rng = [np.arange(-m, m + 1) for m in lim]
    H = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    kout = (S @ sample.UB @ H.T).T + k * beam
    excitation = (np.linalg.norm(kout, axis=1) - k) / k
    sel = np.abs(excitation) <= tolerance
    rows = []
    for hkl, ko, exc in zip(H[sel], kout[sel], excitation[sel]):
        ku = ko / np.linalg.norm(ko)
        along = ku @ arm
        if along <= 0:
            continue
        p = ku * distance / along
        u, v = p @ ex, p @ ey
        if abs(u) > width / 2 or abs(v) > height / 2:
            continue
        hkl_t = tuple(int(x) for x in hkl)
        if allowed is not None and not allowed(hkl_t):
            continue
        rows.append({"h": hkl_t[0], "k": hkl_t[1], "l": hkl_t[2], "u [mm]": u, "v [mm]": v,
                     "excitation": exc, "d [A]": sample.d_spacing(hkl),
                     "ttheta": float(np.degrees(np.arccos(np.clip(ku @ beam, -1, 1))))})
    df = pd.DataFrame(rows)
    return df.sort_values("excitation", key=np.abs).reset_index(drop=True) if len(df) else df


# --- reachability --------------------------------------------------------------------


def reachable_reflections(sample: Sample, energy: float, constraints: dict, hkls,
                          ranges: dict | None = None, aperture=None, detector: str = "h1",
                          progress=None) -> pd.DataFrame:
    """For each (h, k, l): can it be reached under these constraints, inside
    the min/max ranges and (optionally) through ``aperture``?

    ``aperture(gamma, delta) -> bool`` checks the detector angles, e.g. a THC
    envelope (``bernina-scripts' ``thc_angles.aperture_check````). One row per reflection with the
    first solution passing everything, how many do, and if none, why not.
    ``progress(i, n)`` is called before each reflection.
    """
    rows = []
    for i, hkl in enumerate(hkls):
        if progress is not None:
            progress(i, len(hkls))
        row = {"h": hkl[0], "k": hkl[1], "l": hkl[2], "d [A]": sample.d_spacing(hkl)}
        try:
            df = sample.solve(hkl, energy, constraints, detector)
        except Exception as e:
            rows.append({**row, "reachable": False, "why not": str(e)})
            continue
        if df.empty:
            beyond = np.linalg.norm(sample.B @ np.asarray(hkl, float)) > 2 * wavevector(energy)
            rows.append({**row, "reachable": False,
                         "why not": "2 theta > 180: energy too low" if beyond else "no solution under these constraints"})
            continue
        good = [c for c in df.columns if passes(df[c], ranges or {})]
        if not good:
            rows.append({**row, "reachable": False, "why not": f"{len(df.columns)} solution(s), none in the min/max ranges"})
            continue
        if aperture is not None:
            through = [c for c in good if aperture(df.loc["gamma", c], df.loc["delta", c])]
            if not through:
                rows.append({**row, "reachable": False, "why not": f"{len(good)} solution(s), none through the aperture"})
                continue
            good = through
        best = df[good[0]]
        rows.append({**row, "reachable": True, "why not": "", "solutions": len(good),
                     **{n: float(best[n]) for n in CIRCLES}, "ttheta": float(best["ttheta"])})
    return pd.DataFrame(rows)


# --- the 3D view -------------------------------------------------------------------


def view_payload(sample: Sample | None, angles: dict, energy: float, label: str = "",
                 targets=None, n: int = 0) -> dict:
    """The message the 3D page (``escape.bernina.VIEWER_HTML``) reads. The page
    computes its mechanism on You's axes, so UB and the surface normal are
    handed over in that frame."""
    msg = {"angles": _complete(angles), "kMag": k_magnitude(energy), "label": label, "n": n}
    if sample is not None:
        msg["ub"] = to_you(sample.UB).tolist()
        ns = sample.n_surface
        msg["nPhi"] = None if ns is None else to_you(ns).tolist()
    if targets:
        msg["targets"] = [list(map(float, t)) for t in targets]
    return msg
