"""The Bernina diffractometer: You's (1999) six-circle geometry in the SwissFEL frame.

- **Lab frame**: SwissFEL/Bernina axes, z+ along the beam, y+ up,
  x+ = y x z (left, looking along the beam). Everything a user types or reads
  is in this frame.
- **Circles**: mu, eta, chi, phi (sample), gamma, delta (detector). gamma is
  the out-of-plane detector circle diffcalc calls ``nu``; angles mean exactly
  what they mean in diffcalc, only vectors and matrices differ, by
  ``YOU_TO_LAB`` (``from_you``/``to_you``).
- **UB** maps (h, k, l) to Q in the phi frame (the lab frame with all angles
  at 0), 2 pi convention: abs(UB h) = 2 pi / d, abs(k) = 2 pi / lambda.

``GEOMETRY`` is the hklforge geometry everything here solves with.
"""
from __future__ import annotations

import re

import numpy as np
from hklforge import Geometry
from hklforge.geometry import HC_EV_ANGSTROM, Circle, wavevector

# the six circles, in the order of tables and inputs
CIRCLES = ("gamma", "delta", "mu", "eta", "chi", "phi")
SAMPLE_CIRCLES = ("mu", "eta", "chi", "phi")
DETECTOR_CIRCLES = ("gamma", "delta")
# diffcalc's Position order, used for stored reflection positions in eco
POSITION_ORDER = ("mu", "delta", "gamma", "eta", "chi", "phi")
# names in configs written by diffcalc-era code
LEGACY_NAMES = {"gam": "gamma", "nu": "gamma"}

# the SwissFEL/Bernina lab frame: beam along z+, up along y+, x+ = y+ cross z+
BEAM = (0.0, 0.0, 1.0)
UP = (0.0, 1.0, 0.0)
# You's / diffcalc's frame (hklforge's builtin psic): beam along y+, up along x+
YOU_BEAM = (0.0, 1.0, 0.0)
YOU_UP = (1.0, 0.0, 0.0)


def frame_rotation(beam_from, up_from, beam_to=BEAM, up_to=UP) -> np.ndarray:
    """M with v_to = M @ v_from, for two right-handed lab frames given by
    their beam and up directions."""
    def triad(b, u):
        b, u = np.asarray(b, float), np.asarray(u, float)
        return np.column_stack([b, u, np.cross(b, u)])
    return triad(beam_to, up_to) @ triad(beam_from, up_from).T


YOU_TO_LAB = frame_rotation(YOU_BEAM, YOU_UP)


def _turn(v) -> tuple:
    return tuple(float(x) + 0.0 for x in np.round(YOU_TO_LAB @ np.asarray(v, float), 15))


def _bernina_geometry() -> Geometry:
    # the same as psic.reframed(beam="z+", up="y+") plus the gamma name; by
    # hand so that it does not depend on that method
    psic = Geometry.builtin("psic")
    return Geometry(
        name="bernina_you",
        sample=[Circle(c.name, _turn(c.axis)) for c in psic.sample],
        detector=[Circle("gamma" if c.name == "nu" else c.name, _turn(c.axis)) for c in psic.detector],
        beam=_turn(psic.beam),
        description="psic (You 1999) in the Bernina frame (beam z+, up y+), "
                    "with the out-of-plane detector circle named gamma",
        labels={"x": "transverse", "y": "vertical", "z": "beam"},
        azimuth_zero=_turn(psic.azimuth_zero),
        bisect_circle=psic.bisect_circle,
    )


GEOMETRY = _bernina_geometry()


def from_you(a) -> np.ndarray:
    """A diffcalc (You frame) vector or UB/U matrix in this lab frame."""
    return YOU_TO_LAB @ np.asarray(a, float)


def to_you(a) -> np.ndarray:
    """A vector or UB/U matrix of this lab frame in diffcalc's (You) frame
    (also what the 3D page is built on)."""
    return YOU_TO_LAB.T @ np.asarray(a, float)


def turn_crystal_data(values, M):
    """Crystal file values (u_matrix, ub_matrix, orientations, vectors) with
    every lab/phi-frame vector turned by M; angles and hkl stay as they are."""
    out = dict(values)
    for key in ("u_matrix", "ub_matrix"):
        if len(out.get(key) or []):
            out[key] = (M @ np.asarray(out[key], float)).tolist()
    if "orientations" in out:
        out["orientations"] = [{**o, "xyz": (M @ np.asarray(o["xyz"], float)).tolist()}
                               for o in out["orientations"]]
    if "vectors" in out:
        out["vectors"] = {k: ({"phi": (M @ np.asarray(v["phi"], float)).tolist()} if "phi" in v else v)
                          for k, v in out["vectors"].items()}
    return out


# --- units -------------------------------------------------------------------------


def en2lam(energy_eV):
    """Photon energy in eV -> wavelength in angstrom."""
    return HC_EV_ANGSTROM / energy_eV


def lam2en(lam):
    """Wavelength in angstrom -> energy in eV."""
    return HC_EV_ANGSTROM / lam


def k_magnitude(energy: float) -> float:
    """abs(k) = 2 pi / lambda in 1/angstrom (the convention of UB)."""
    return wavevector(energy)


# --- directions as people write them -------------------------------------------------


def parse_vector(text) -> tuple:
    """Three numbers as people write them: "(1,1,1)", "1 1 1", "[1, -1, 0]",
    "(001)", "1-10" (one digit each)."""
    t = str(text).strip().strip("()[]{} ").strip()
    parts = [x for x in re.split(r"[,\s;]+", t) if x]
    if len(parts) == 1 and re.fullmatch(r"(-?\d){3}", t):
        parts = re.findall(r"-?\d", t)
    if len(parts) != 3:
        raise ValueError(f"{text!r}: give three numbers, e.g. (1,0,0)")
    return tuple(float(x) for x in parts)


LAB_WORDS = {"beam": BEAM, "up": UP, "down": (0.0, -1.0, 0.0),
             "x+": (1.0, 0.0, 0.0), "x-": (-1.0, 0.0, 0.0), "y+": UP, "y-": (0.0, -1.0, 0.0),
             "z+": BEAM, "z-": (0.0, 0.0, -1.0)}


def parse_direction(text) -> tuple:
    """A lab direction: three numbers or a word (beam, up, down, x+, z-, ...)."""
    word = str(text).strip().lower()
    if word in LAB_WORDS:
        return LAB_WORDS[word]
    return parse_vector(text)
