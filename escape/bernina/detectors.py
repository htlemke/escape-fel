"""Detector faces: sizes and module layouts (Bernina's JUNGFRAU detectors).

A detector is a flat rectangle (width, height in mm) centred on the
detector arm; ``DETECTOR_MODULES`` gives the sensitive modules of the ones
with gaps, as (u0, v0, u1, v1) rectangles in face coordinates (mm, centred).
"""
from __future__ import annotations

import numpy as np

# a flat detector centred on the detector arm, its width along x and its
# height along y with all angles 0 (as drawn in the 3D view). JUNGFRAU
# module: 1024 x 512 pixels of 75 um (gaps between chips ignored).
DETECTOR_PRESETS = {
    "77 x 77 mm": (77.0, 77.0),
    "JUNGFRAU 1 module (76.8 x 38.4 mm)": (76.8, 38.4),
    "JUNGFRAU 2 modules (76.8 x 76.8 mm)": (76.8, 76.8),
    "JUNGFRAU 4 modules (153.6 x 76.8 mm)": (153.6, 76.8),
}

# Bernina's JUNGFRAU detectors with their module gaps, from jungfrau_utils
# (geometry.py, detector_geometry): the origin of each module in pixels of the
# assembled image, where a module is 1030 x 514 pixels (4 x 2 chips of 256 plus
# the 2-pixel gaps of the double-size pixels between chips). Image x is drawn
# along the width, image y along the height, row 0 at the bottom; the detector
# is centred on the arm.
JF_PIXEL = 0.075
JF_MODULE = (1030, 514)
JUNGFRAU_LAYOUTS = {
    "JF01T03": ((0, 0, 0), (0, 550, 1100)),
    "JF07T32": ((68, 1107, 2146, 3185) * 4 + (0, 1039, 2078, 3117) * 4,
                tuple(550 * row + dy for row in range(8) for dy in (0, 0, 68, 68))),
}


def jungfrau_modules(name: str, rotated: bool = False):
    """(width, height, modules) of a JUNGFRAU layout in mm; modules are
    (u0, v0, u1, v1) rectangles in face coordinates, centred on the detector.
    ``rotated``: turned by 90 degrees on the arm, image x along the height."""
    ox, oy = (np.asarray(o, float) for o in JUNGFRAU_LAYOUTS[name])
    width = float((ox.max() + JF_MODULE[0]) * JF_PIXEL)
    height = float((oy.max() + JF_MODULE[1]) * JF_PIXEL)
    modules = tuple((x * JF_PIXEL - width / 2, y * JF_PIXEL - height / 2,
                     (x + JF_MODULE[0]) * JF_PIXEL - width / 2, (y + JF_MODULE[1]) * JF_PIXEL - height / 2)
                    for x, y in zip(ox, oy))
    if rotated:   # (u, v) -> (-v, u)
        return height, width, tuple((-v1, u0, -v0, u1) for u0, v0, u1, v1 in modules)
    return width, height, modules


# module rectangles of the presets that have gaps between modules
DETECTOR_MODULES = {}
for _name, _what in (("JF01T03", "Bernina 1.5M, 3 modules"), ("JF07T32", "Bernina 16M, 32 modules")):
    for _rotated in (False, True):
        _w, _h, _modules = jungfrau_modules(_name, _rotated)
        _label = f"{_name} {_what}{', turned 90°' if _rotated else ''} ({_w:.1f} x {_h:.1f} mm)"
        DETECTOR_PRESETS[_label] = (round(_w, 4), round(_h, 4))
        DETECTOR_MODULES[_label] = _modules


def on_modules(modules, u, v):
    """True where face points (u, v) lie on a module; everywhere if ``modules`` is None."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if not modules:
        return np.isfinite(u) & np.isfinite(v)
    hit = np.zeros(np.broadcast(u, v).shape, bool)
    for u0, v0, u1, v1 in modules:
        hit |= (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    return hit
