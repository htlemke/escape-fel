"""Detector view: what a flat detector face sees in reciprocal space.

Every point of the face receives the beam scattered along k', so it sees
Q = k' - k: a point on the Ewald sphere. This module evaluates that Q, and
quantities derived from it, over the face at one pose of the diffractometer,
and draws the face as an image with optional overlays. The options follow
the usual ways such images are annotated:

* integer h, k, l lines and the hkl of every point (area-detector reciprocal
  space mapping, e.g. BINoculars, rsMap3D, fast_rsm);
* Brillouin zones: which reciprocal-lattice point is nearest, the zone
  boundaries, and the distance |q| from the nearest point, as in diffuse
  and time-resolved phonon scattering (Trigo et al. 2013, Zhu et al. 2015,
  where detector images are read zone by zone);
* predicted Bragg spots, powder (Debye-Scherrer) rings of the allowed
  reflections and the direct beam (crystallography image viewers, e.g.
  DIALS: predictions, resolution rings, beam centre);
* surface diffraction: exit angle, Q along and across the surface, the
  sample horizon, the Yoneda line at the critical angle and where crystal
  truncation rods cross the face (GIXS tools: pygix, GIXSGUI);
* intensity factors of the geometry: polarization (horizontal, as at
  SwissFEL) and the solid angle of a pixel.

Lattice points are those of the cell entered; with a space group, the
Brillouin zones and powder rings use only the points its lattice centring
(or, for the rings, its full reflection conditions) allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from . import diffraction as dif

# key: (label, colormap, kind); kind: "lin", "centred" (symmetric around 0),
# "log", "zone" (categorical) or "cyclic"
COLOURS = {
    "h": ("h", "viridis", "lin"),
    "k": ("k", "viridis", "lin"),
    "l": ("l", "viridis", "lin"),
    "h_red": ("h - nearest integer", "RdBu_r", "centred"),
    "k_red": ("k - nearest integer", "RdBu_r", "centred"),
    "l_red": ("l - nearest integer", "RdBu_r", "centred"),
    "Q": ("|Q| [1/Å]", "viridis", "lin"),
    "d": ("d = 2π/|Q| [Å]", "viridis", "lin"),
    "ttheta": ("2θ [deg]", "viridis", "lin"),
    "azimuth": ("azimuth around the beam [deg]", "twilight", "cyclic"),
    "zone": ("Brillouin zone (nearest lattice point)", None, "zone"),
    "q_red": ("|q| from the nearest lattice point [1/Å]", "magma", "lin"),
    "tds": ("diffuse estimate |Q|²/|q|² (log)", "inferno", "log"),
    "alpha_f": ("exit angle to the surface [deg]", "RdBu_r", "centred"),
    "q_perp": ("Q along the surface normal [1/Å]", "viridis", "lin"),
    "q_par": ("Q in the surface plane [1/Å]", "viridis", "lin"),
    "polarization": ("polarization factor (horizontal)", "cividis", "lin"),
    "solid_angle": ("solid angle of a pixel (relative)", "cividis", "lin"),
}
SURFACE_COLOURS = ("alpha_f", "q_perp", "q_par")

OVERLAYS = {
    "integer": "integer h / k / l lines",
    "cells": "cell boundaries (hkl ± ½)",
    "bz": "Brillouin-zone boundaries",
    "zone_labels": "zone labels (nearest hkl)",
    "spots": "Bragg spots (within the band)",
    "rings": "powder rings of the allowed reflections",
    "beam": "direct beam",
    "horizon": "sample horizon and Yoneda line",
    "rods": "truncation-rod crossings",
}
DEFAULT_OVERLAYS = ("integer", "spots", "beam")

# vectorised lattice-centring conditions (the lattice itself, not glides/screws)
_CENTRING = {
    "P": lambda h, k, l: np.ones(np.shape(h), bool),
    "A": lambda h, k, l: (k + l) % 2 == 0,
    "B": lambda h, k, l: (h + l) % 2 == 0,
    "C": lambda h, k, l: (h + k) % 2 == 0,
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: ((h % 2) == (k % 2)) & ((k % 2) == (l % 2)),
    "R": lambda h, k, l: (-h + k + l) % 3 == 0,
}

# 27 colours, so that neighbouring zones (whose indices differ by at most 2)
# never share one: the colour is (h mod 3, k mod 3, l mod 3)
_ZONE_RGB = None


def _zone_colours():
    global _ZONE_RGB
    if _ZONE_RGB is None:
        import matplotlib as mpl
        cols = [mpl.colormaps["tab20"](i) for i in range(20)] + \
               [mpl.colormaps["tab20b"](i) for i in (0, 4, 8, 12, 16, 2, 6)]
        _ZONE_RGB = np.array([c[:3] for c in cols])
    return _ZONE_RGB


def lattice_centring(space_group) -> str:
    """The centring letter whose lattice the zones use ("P" without a space group)."""
    if space_group is None:
        return "P"
    c = getattr(space_group, "centring", "P") or "P"
    if c == "R" and getattr(space_group, "rhombohedral_axes", False):
        return "P"
    return c if c in _CENTRING else "P"


@dataclass
class Pose:
    """One diffractometer pose with its detector."""
    sample: dif.Sample
    angles: dict
    energy: float
    distance: float
    width: float = 77.0
    height: float = 77.0
    centring: str = "P"
    alpha_c: float | None = None          # critical angle [deg] for the Yoneda line
    modules: tuple | None = None          # (u0, v0, u1, v1) [mm] of each module; None: one full face
    arm: np.ndarray = field(init=False)
    ex: np.ndarray = field(init=False)
    ey: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    k: float = field(init=False)
    n_lab: np.ndarray | None = field(init=False)

    def __post_init__(self):
        self.arm, self.ex, self.ey = dif._detector_frame(self.angles)
        self.S = dif.GEOMETRY.sample_matrix(dif._complete(self.angles))
        self.k = dif.wavevector(self.energy)
        self.UB = self.sample.UB
        ns = self.sample.n_surface
        self.n_lab = None if ns is None else self.S @ (np.asarray(ns, float) / np.linalg.norm(ns))

    @property
    def alpha_in(self) -> float | None:
        """Incidence angle to the surface [deg]."""
        if self.n_lab is None:
            return None
        return float(np.degrees(np.arcsin(np.clip(-np.asarray(dif.BEAM) @ self.n_lab, -1, 1))))

    def to_face(self, kout):
        """(u, v) in mm where directions ``kout`` (..., 3) hit the face; NaN where they miss."""
        kout = np.asarray(kout, float)
        ku = kout / np.linalg.norm(kout, axis=-1, keepdims=True)
        along = ku @ self.arm
        with np.errstate(divide="ignore", invalid="ignore"):
            p = ku * (self.distance / along)[..., None]
        u, v = p @ self.ex, p @ self.ey
        miss = (along <= 0) | (np.abs(u) > self.width / 2) | (np.abs(v) > self.height / 2)
        return np.where(miss, np.nan, u), np.where(miss, np.nan, v)


def evaluate(pose: Pose, U, V) -> dict:
    """Everything the view can show, at face coordinates ``U``, ``V`` (mm, same shape)."""
    U, V = np.asarray(U, float), np.asarray(V, float)
    P = pose.distance * pose.arm + U[..., None] * pose.ex + V[..., None] * pose.ey
    r = np.linalg.norm(P, axis=-1)
    kout = P / r[..., None]
    beam = np.asarray(dif.BEAM)
    q_lab = pose.k * (kout - beam)
    q_phi = q_lab @ pose.S                                   # S^T q per point
    hkl = np.linalg.solve(pose.UB, q_phi.reshape(-1, 3).T).T.reshape(q_phi.shape)
    Q = np.linalg.norm(q_lab, axis=-1)
    out = {"h": hkl[..., 0], "k": hkl[..., 1], "l": hkl[..., 2], "Q": Q,
           "ttheta": np.degrees(np.arccos(np.clip(kout @ beam, -1, 1)))}
    with np.errstate(divide="ignore"):
        out["d"] = np.where(Q > 0, 2 * np.pi / Q, np.inf)
    for c in "hkl":
        out[c + "_red"] = out[c] - np.round(out[c])
    # azimuth around the beam: 0 along x+ (horizontal), 90 along y+ (up)
    out["azimuth"] = np.degrees(np.arctan2(kout[..., 1], kout[..., 0]))

    node, qred = nearest_nodes(pose, hkl, q_phi)
    out["node"] = node
    out["q_red"] = np.linalg.norm(qred, axis=-1)
    with np.errstate(divide="ignore"):
        out["tds"] = np.log10(np.where(out["q_red"] > 0, Q ** 2 / out["q_red"] ** 2, np.inf))
    out["zone"] = ((node[..., 0] % 3) * 9 + (node[..., 1] % 3) * 3 + node[..., 2] % 3).astype(float)

    if pose.n_lab is not None:
        n = pose.n_lab
        out["alpha_f"] = np.degrees(np.arcsin(np.clip(kout @ n, -1, 1)))
        out["q_perp"] = q_lab @ n
        out["q_par"] = np.linalg.norm(q_lab - out["q_perp"][..., None] * n, axis=-1)
    pol = np.array([1.0, 0.0, 0.0])                          # SwissFEL: horizontal
    out["polarization"] = 1 - (kout @ pol) ** 2
    cos = kout @ pose.arm
    out["solid_angle"] = cos ** 3                            # (distance / r)^2 * cos
    return out


def nearest_nodes(pose: Pose, hkl, q_phi):
    """The nearest allowed lattice point (in |Q|, i.e. the Brillouin zone) of
    each point, and q = Q - G from it. Searches the integer points around
    each hkl: +-1 for a primitive lattice, +-2 for a centred one."""
    r = 1 if pose.centring == "P" else 2
    off = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    base = np.round(hkl).astype(int)
    flat_base = base.reshape(-1, 3)
    flat_q = q_phi.reshape(-1, 3)
    cand = flat_base[:, None, :] + off[None, :, :]                       # (N, M, 3)
    ok = _CENTRING[pose.centring](cand[..., 0], cand[..., 1], cand[..., 2])
    G = cand @ pose.UB.T                                                  # (N, M, 3)
    dist = np.linalg.norm(flat_q[:, None, :] - G, axis=-1)
    dist = np.where(ok, dist, np.inf)
    best = np.argmin(dist, axis=1)
    node = cand[np.arange(len(cand)), best]
    qred = flat_q - G[np.arange(len(cand)), best]
    return node.reshape(base.shape), qred.reshape(q_phi.shape)


def face_grid(pose: Pose, n: int = 161) -> dict:
    """``evaluate`` on a regular grid over the face, n points across the longer side."""
    side = max(pose.width, pose.height)
    nu = max(2, int(round(n * pose.width / side)))
    nv = max(2, int(round(n * pose.height / side)))
    u = np.linspace(-pose.width / 2, pose.width / 2, nu)
    v = np.linspace(-pose.height / 2, pose.height / 2, nv)
    U, V = np.meshgrid(u, v)
    data = evaluate(pose, U, V)
    data["u"], data["v"] = u, v
    return data


def point_text(pose: Pose, u: float, v: float) -> str:
    """One line describing the point (u, v) of the face, for hover and click."""
    d = evaluate(pose, np.array([[u]]), np.array([[v]]))
    g = lambda key: float(d[key][0, 0])                       # noqa: E731
    node = d["node"][0, 0]
    txt = (f"u {u:.1f} v {v:.1f} mm | hkl {g('h'):.3f} {g('k'):.3f} {g('l'):.3f} | |Q| {g('Q'):.4f} 1/Å, "
           f"d {g('d'):.4f} Å, 2θ {g('ttheta'):.2f}° | zone ({node[0]} {node[1]} {node[2]}), "
           f"|q| {g('q_red'):.4f} 1/Å")
    if "alpha_f" in d:
        txt += f" | αf {g('alpha_f'):.3f}°"
    return txt


# --- overlays computed from the lattice --------------------------------------------


def _lattice_points(pose: Pose, qmax: float, limit: float = 2e6):
    """All integer hkl with |G| <= qmax (without 000), or None if too many."""
    lim = [int(np.ceil(qmax / np.linalg.norm(pose.UB @ e))) for e in np.eye(3)]
    if np.prod([2 * m + 1 for m in lim]) > limit:
        return None
    H = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in lim], indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    return H[np.linalg.norm(H @ pose.UB.T, axis=1) <= qmax]


def powder_rings(pose: Pose, qmin: float, qmax: float, allowed=None, max_rings: int = 40):
    """[(|G|, (h, k, l) of the family)] of the allowed reflections with qmin <= |G| <= qmax."""
    H = _lattice_points(pose, qmax)
    if H is None:
        return None
    G = np.linalg.norm(H @ pose.UB.T, axis=1)
    sel = G >= qmin
    H, G = H[sel], G[sel]
    rings = {}
    # per |G| the representative with the fewest minus signs, then the largest indices
    order = sorted(zip(map(tuple, H.tolist()), G),
                   key=lambda x: (round(x[1], 6), sum(c < 0 for c in x[0]), tuple(-c for c in x[0])))
    for hkl, q in order:
        key = round(q, 5)
        if key in rings:
            continue
        if allowed is not None and not allowed(hkl):
            continue
        rings[key] = (q, hkl)
    return sorted(rings.values())[:max_rings]


def rod_crossings(pose: Pose, max_rods: int = 5000):
    """Where crystal truncation rods (lines through every lattice point along
    the surface normal) cross the Ewald sphere and hit the face:
    [(u, v, (h, k, l) at the crossing)]."""
    if pose.n_lab is None:
        return []
    n_phi = pose.S.T @ pose.n_lab
    H = _lattice_points(pose, 2 * pose.k * 1.05)
    if H is None:
        return None
    G = H @ pose.UB.T
    G_par = G - (G @ n_phi)[:, None] * n_phi
    _, first = np.unique(np.round(G_par, 5), axis=0, return_index=True)
    G = G[first][:max_rods]
    beam = np.asarray(dif.BEAM) * pose.k
    a = beam + G @ pose.S.T                       # k + S G
    b = pose.S @ n_phi
    ab = a @ b
    disc = ab ** 2 - (np.einsum("ij,ij->i", a, a) - pose.k ** 2)
    out = []
    UBinv = np.linalg.inv(pose.UB)
    for sign in (-1, 1):
        with np.errstate(invalid="ignore"):
            t = -ab + sign * np.sqrt(disc)
        ok = disc >= 0
        kout = a[ok] + t[ok, None] * b
        u, v = pose.to_face(kout)
        hit = ~np.isnan(u)
        q_phi = (kout[hit] - beam) @ pose.S
        hkl = q_phi @ UBinv.T
        out += [(float(x), float(y), tuple(np.round(h, 2))) for x, y, h in zip(u[hit], v[hit], hkl)]
    return out


def direct_beam(pose: Pose):
    u, v = pose.to_face(np.asarray(dif.BEAM, float))
    return None if np.isnan(u) else (float(u), float(v))


# --- drawing -----------------------------------------------------------------------


def _fmt_hkl(h):
    return " ".join(f"{int(x)}" if float(x).is_integer() else f"{x:g}" for x in h)


def draw(fig, pose: Pose, data: dict, colour: str = "l", overlays=DEFAULT_OVERLAYS,
         spots=None, allowed=None, title: str = "") -> list[str]:
    """Draw the face into ``fig`` (cleared first). Returns notes on what was
    left out (no surface normal, too many lattice points, ...)."""
    from matplotlib.patches import Rectangle

    notes = []
    overlays = set(overlays)
    fig.clf()
    ax = fig.add_subplot(111)
    u, v = data["u"], data["v"]
    if colour in SURFACE_COLOURS and colour not in data:
        notes.append(f"{COLOURS[colour][0]}: needs a surface normal (sample tab); showing l")
        colour = "l"
    label, cmap, kind = COLOURS[colour]
    z = data[colour]
    if kind == "zone":
        rgb = _zone_colours()[data["zone"].astype(int) % 27]
        ax.imshow(rgb, origin="lower", extent=(u[0], u[-1], v[0], v[-1]), interpolation="nearest",
                  aspect="equal")
    else:
        kw = {}
        if kind == "centred":
            m = np.nanmax(np.abs(z[np.isfinite(z)])) if np.isfinite(z).any() else 1
            kw = {"vmin": -m, "vmax": m}
        elif kind == "cyclic":
            kw = {"vmin": -180, "vmax": 180}
        elif kind == "log":
            fin = z[np.isfinite(z)]
            if fin.size:
                kw = {"vmin": np.percentile(fin, 1), "vmax": np.percentile(fin, 99.5)}
        im = ax.pcolormesh(u, v, np.where(np.isfinite(z), z, np.nan), shading="auto", cmap=cmap, **kw)
        fig.colorbar(im, ax=ax, label=label, shrink=0.85)

    def contours(values, levels, **kw):
        if len(levels) and len(levels) < 60:
            ax.contour(u, v, values, levels=levels, **kw)

    if "integer" in overlays:
        for comp, col in (("h", "w"), ("k", "orange"), ("l", "r")):
            vals = data[comp]
            contours(vals, np.arange(np.ceil(vals.min()), np.floor(vals.max()) + 1),
                     colors=col, linewidths=0.7, alpha=0.8)
    if "cells" in overlays:
        for comp in "hkl":
            vals = data[comp]
            contours(vals, np.arange(np.ceil(vals.min() - 0.5), np.floor(vals.max() - 0.5) + 1) + 0.5,
                     colors="k", linewidths=0.6, linestyles="--", alpha=0.6)
    node = data["node"]
    if "bz" in overlays:
        _zone_boundaries(ax, pose, data)
    if "zone_labels" in overlays:
        keys, inverse = np.unique(node.reshape(-1, 3), axis=0, return_inverse=True)
        inverse = inverse.reshape(node.shape[:2])
        if len(keys) <= 60:
            U, V = np.meshgrid(u, v)
            for i, key in enumerate(keys):
                m = inverse == i
                if m.sum() < 4:
                    continue
                ax.text(U[m].mean(), V[m].mean(), f"({_fmt_hkl(key)})", ha="center", va="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.4", alpha=0.75, lw=0.5))
        else:
            notes.append(f"{len(keys)} zones on the face: zone labels left out (at most 60)")
    if "rings" in overlays:
        qmin, qmax = float(np.nanmin(data["Q"])), float(np.nanmax(data["Q"]))
        rings = powder_rings(pose, qmin, qmax, allowed)
        if rings is None:
            notes.append("powder rings: too many lattice points (large cell)")
        elif rings:
            levels = sorted({q for q, _ in rings})
            cs = ax.contour(u, v, data["Q"], levels=levels, colors="cyan", linewidths=0.8, linestyles=":")
            fam = {round(q, 5): f"{_fmt_hkl(h)}" for q, h in rings}
            ax.clabel(cs, fmt=lambda q: fam.get(round(q, 5), ""), fontsize=7, inline=True)
    if "horizon" in overlays:
        if "alpha_f" not in data:
            notes.append("horizon: needs a surface normal (sample tab)")
        else:
            a = data["alpha_f"]
            if a.min() < 0 < a.max():
                ax.contour(u, v, a, levels=[0], colors="deepskyblue", linewidths=1.4)
            if pose.alpha_c and a.min() < pose.alpha_c < a.max():
                ax.contour(u, v, a, levels=[pose.alpha_c], colors="gold", linewidths=1.2, linestyles="--")
            elif pose.alpha_c is None:
                notes.append("Yoneda line: needs the critical angle (formula and density in the grazing tab)")
    if "rods" in overlays:
        rods = rod_crossings(pose)
        if pose.n_lab is None:
            notes.append("truncation rods: need a surface normal (sample tab)")
        elif rods is None:
            notes.append("truncation rods: too many lattice points (large cell)")
        elif rods:
            ax.plot([r[0] for r in rods], [r[1] for r in rods], "D", mfc="none", mec="deepskyblue", ms=7)
            for x, y, h in rods[:25]:
                ax.annotate(_fmt_hkl(h), (x, y), color="deepskyblue", fontsize=7, xytext=(4, -9),
                            textcoords="offset points")
    if pose.modules:
        # the gaps between modules greyed, the modules outlined
        U, V = np.meshgrid(u, v)
        gap = ~dif.on_modules(pose.modules, U, V)
        ax.imshow(np.where(gap, 1.0, np.nan), origin="lower", extent=(u[0], u[-1], v[0], v[-1]),
                  cmap="Greys", vmin=0, vmax=1.5, alpha=0.75, interpolation="nearest", aspect="equal")
        for u0, v0, u1, v1 in pose.modules:
            ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, ec="0.25", lw=0.6))
    if "spots" in overlays and spots is not None and len(spots):
        hit = dif.on_modules(pose.modules, spots["u [mm]"], spots["v [mm]"])
        ax.plot(spots["u [mm]"][hit], spots["v [mm]"][hit], "o", mfc="none", mec="w", mew=1.5, ms=10)
        if not hit.all():
            ax.plot(spots["u [mm]"][~hit], spots["v [mm]"][~hit], "o", mfc="none", mec="0.5", mew=1.2, ms=10,
                    ls="none")
            notes.append(f"{int((~hit).sum())} reflection(s) fall into gaps between modules (grey circles)")
        for (_, r), on in list(zip(spots.iterrows(), hit))[:30]:
            ax.annotate(f"{int(r.h)} {int(r.k)} {int(r.l)}", (r["u [mm]"], r["v [mm]"]),
                        color="w" if on else "0.5", fontsize=8, xytext=(5, 5), textcoords="offset points")
    if "beam" in overlays:
        db = direct_beam(pose)
        if db is not None:
            ax.plot(*db, "+", color="red", ms=14, mew=2)
            ax.annotate("direct beam", db, color="red", fontsize=8, xytext=(6, -12), textcoords="offset points")
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(v[0], v[-1])
    ax.set_aspect("equal")
    ax.set_xlabel("u [mm] (along x+ at all angles 0)")
    ax.set_ylabel("v [mm] (along y+ at all angles 0)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.format_coord = lambda x, y: point_text(pose, x, y)
    return notes


def _zone_boundaries(ax, pose: Pose, data: dict):
    """Smooth zone boundaries: for every pair of zones that touch on the face,
    the bisector |q - G_a| = |q - G_b| drawn where one of the two is nearest."""
    node = data["node"]
    u, v = data["u"], data["v"]
    U, V = np.meshgrid(u, v)
    keys, inv = np.unique(node.reshape(-1, 3), axis=0, return_inverse=True)
    inv = inv.reshape(node.shape[:2])
    if len(keys) < 2:
        return
    pairs = set()
    for a, b in ((inv[:, 1:], inv[:, :-1]), (inv[1:, :], inv[:-1, :])):
        diff = a != b
        pairs |= {tuple(sorted(p)) for p in zip(a[diff].tolist(), b[diff].tolist())}
    if len(pairs) > 400:
        return
    P = pose.distance * pose.arm + U[..., None] * pose.ex + V[..., None] * pose.ey
    kout = P / np.linalg.norm(P, axis=-1, keepdims=True)
    q_phi = (pose.k * (kout - np.asarray(dif.BEAM))) @ pose.S
    for ia, ib in pairs:
        Ga, Gb = pose.UB @ keys[ia], pose.UB @ keys[ib]
        g = np.linalg.norm(q_phi - Ga, axis=-1) - np.linalg.norm(q_phi - Gb, axis=-1)
        near = (inv == ia) | (inv == ib)
        # grow the region by a pixel so the line reaches the region's edge
        grown = near.copy()
        grown[1:, :] |= near[:-1, :]
        grown[:-1, :] |= near[1:, :]
        grown[:, 1:] |= near[:, :-1]
        grown[:, :-1] |= near[:, 1:]
        ax.contour(u, v, np.where(grown, g, np.nan), levels=[0], colors="k", linewidths=1.1)
