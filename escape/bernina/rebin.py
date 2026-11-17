"""Rebinning area-detector pixels into another space (2theta, chi, q, hkl, ...).

Binning is linear: with ``A`` the (bins x pixels) matrix of the geometry,
``sums = A @ data``. Two cases:

- **fixed geometry** (:class:`FixedBinner`): ``A`` is built once - from one
  bin per pixel, from coordinates on a :class:`Grid`, from pyFAI's
  pixel-splitting matrix (:meth:`FixedBinner.from_pyfai`), or for a
  diffractometer pose into a 3D q/hkl grid (:meth:`FixedBinner.from_pose`)
  - and applied to whole stacks as they come from the detector.
- **changing geometry** (:class:`StreamBinner`): every shot has its own
  angles. Pixel positions are fixed on the detector face, so one fused
  kernel goes pixel -> q -> voxel -> sum per shot, without temporaries. For
  slowly changing angles the voxel indices are reused while the geometry
  stays within a tolerance, and with ``group=True`` such shots are summed
  first and binned once.

Normalization follows pyFAI: the result is ``sum(signal) / sum(weight *
solid angle * polarization)`` per bin, so the mean is corrected and empty
bins are NaN. NaN pixels (gaps and masked pixels of corrected JUNGFRAU
data) are left out of both sums, per frame.

Backends: ``"numba"`` (default when installed: parallel CPU kernels),
``"numpy"`` (no compiled code), ``"cupy"`` (GPU; the same array code as
``"numpy"`` with cupy arrays and cupyx sparse matrices). dask: see
:meth:`FixedBinner.apply` / :meth:`FixedBinner.mean_dask` (one result per
shot) and :meth:`StreamBinner.reduce` (many shots into one grid).

Frame: SwissFEL/Bernina lab, z+ beam, y+ up, x+ left; the X-rays are
polarized horizontally (x). Lengths in mm, q in 1/Angstrom (2 pi
convention, as escape.bernina.diffraction).
"""
from __future__ import annotations

import numpy as np

try:
    import numba as nb
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

BEAM = (0.0, 0.0, 1.0)
# fastmath without 'nnan'/'ninf': the kernels test x == x to skip NaN pixels
_FM = {"nsz", "arcp", "contract", "afn", "reassoc"}
_NONE_F = np.zeros(0, np.float32)
_NONE_F2 = np.zeros((0, 0), np.float32)


def _backend(backend):
    if backend in (None, "auto"):
        return "numba" if HAVE_NUMBA else "numpy"
    if backend == "numba" and not HAVE_NUMBA:
        raise ImportError("backend 'numba' needs numba")
    if backend not in ("numba", "numpy", "cupy"):
        raise ValueError(f"unknown backend {backend!r}")
    return backend


def _xp(backend):
    if backend == "cupy":
        import cupy
        return cupy
    return np


def _to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# ====================================================================== grids

class Grid:
    """A regular grid of voxels (any number of dimensions).

    ``lo``: lower edge, ``step``: voxel size, ``n``: voxels per axis. Build
    it from a voxel size (:meth:`from_voxel`, :meth:`covering`), from a
    number of bins (:meth:`from_bins`), or as xrayutilities' gridders do,
    with the first and last voxel *centres* (:meth:`from_centres`).
    """

    def __init__(self, lo, step, n):
        n = np.atleast_1d(np.asarray(n, np.int64))
        self.lo = np.broadcast_to(np.asarray(lo, float), n.shape).copy()
        self.step = np.broadcast_to(np.asarray(step, float), n.shape).copy()
        self.n = n
        if np.any(self.n < 1) or np.any(self.step <= 0):
            raise ValueError("a grid needs n >= 1 and step > 0 on every axis")

    @classmethod
    def from_voxel(cls, lo, hi, voxel):
        lo, hi = np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float))
        voxel = np.broadcast_to(np.asarray(voxel, float), lo.shape)
        n = np.maximum(1, np.ceil((hi - lo) / voxel - 1e-9)).astype(np.int64)
        return cls(lo, voxel, n)

    @classmethod
    def from_bins(cls, lo, hi, n):
        lo, hi = np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float))
        n = np.broadcast_to(np.asarray(n, np.int64), lo.shape)
        return cls(lo, (hi - lo) / n, n)

    @classmethod
    def from_centres(cls, first, last, n):
        """xrayutilities' convention: n points from ``first`` to ``last``."""
        first, last = np.atleast_1d(np.asarray(first, float)), np.atleast_1d(np.asarray(last, float))
        n = np.broadcast_to(np.asarray(n, np.int64), first.shape)
        step = (last - first) / np.maximum(n - 1, 1)
        return cls(first - step / 2, step, n)

    @classmethod
    def covering(cls, points, voxel, pad=0.5, align=True):
        """The voxel grid covering ``points`` (..., ndim) plus ``pad`` voxels.

        With ``align`` the edges sit on multiples of the voxel size, so grids
        of different runs with the same voxel size coincide and add up.
        """
        pts = np.asarray(points, float).reshape(-1, np.shape(points)[-1])
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        voxel = np.broadcast_to(np.asarray(voxel, float), pts.shape[1:])
        lo, hi = pts.min(0) - pad * voxel, pts.max(0) + pad * voxel
        if align:
            lo, hi = np.floor(lo / voxel) * voxel, np.ceil(hi / voxel) * voxel
        return cls.from_voxel(lo, hi, voxel)

    ndim = property(lambda self: self.n.size)
    shape = property(lambda self: tuple(int(x) for x in self.n))
    size = property(lambda self: int(np.prod(self.n)))
    hi = property(lambda self: self.lo + self.n * self.step)

    @property
    def edges(self):
        return [lo + np.arange(n + 1) * s for lo, s, n in zip(self.lo, self.step, self.n)]

    @property
    def centres(self):
        """Voxel centres per axis (xrayutilities' ``xaxis``, ``yaxis``, ...)."""
        return [lo + (np.arange(n) + 0.5) * s for lo, s, n in zip(self.lo, self.step, self.n)]

    def index(self, coords):
        """Flat voxel index of points ``coords`` (..., ndim); -1 outside or NaN."""
        c = np.asarray(coords, float)
        if self.ndim == 1 and (c.ndim == 0 or c.shape[-1] != 1):
            c = c[..., None]
        with np.errstate(invalid="ignore"):
            i = np.floor((c - self.lo) / self.step)
            ok = np.all((i >= 0) & (i < self.n), axis=-1)
        i = np.where(ok[..., None], i, 0).astype(np.int64)
        flat = np.ravel_multi_index(np.moveaxis(i, -1, 0), self.shape)
        return np.where(ok, flat, -1)

    def affine(self, R=None):
        """(M, off) with grid coordinates g = M x + off for x = R q (3D)."""
        R = np.eye(self.ndim) if R is None else np.asarray(R, float)
        return R / self.step[:, None], -self.lo / self.step

    def __repr__(self):
        return f"Grid(lo={self.lo.tolist()}, step={self.step.tolist()}, n={self.n.tolist()})"


# ====================================================================== detector and geometry

def detector_pixels(width, height, pixel):
    """Face coordinates (u, v) [mm] of the pixel centres of a flat detector
    centred on the arm, in the order of an image (rows along v, row 0 at
    the bottom as in escape.bernina.detectors; u fastest): (npix, 2)."""
    pu, pv = np.broadcast_to(np.asarray(pixel, float), (2,))
    nu, nv = int(round(width / pu)), int(round(height / pv))
    u = (np.arange(nu) + 0.5) * pu - width / 2
    v = (np.arange(nv) + 0.5) * pv - height / 2
    U, V = np.meshgrid(u, v)
    return np.stack([U.ravel(), V.ravel()], -1)


def jungfrau_pixels(name):
    """(uv, image shape) of an assembled JUNGFRAU image (gap pixels
    included; they are NaN in corrected data and are skipped)."""
    from .detectors import JF_MODULE, JF_PIXEL, JUNGFRAU_LAYOUTS
    ox, oy = (np.asarray(o) for o in JUNGFRAU_LAYOUTS[name])
    nx, ny = int(ox.max() + JF_MODULE[0]), int(oy.max() + JF_MODULE[1])
    return detector_pixels(nx * JF_PIXEL, ny * JF_PIXEL, JF_PIXEL), (ny, nx)


def shot_geometry(angles, sample=None, space="hkl"):
    """(arm, ex, ey, R) of one shot from diffractometer ``angles`` (dict, as
    escape.bernina.diffraction). ``space``: "hkl" (needs ``sample``), "q"
    (sample frame q, the phi frame) or "lab" (lab q)."""
    from . import diffraction as dif
    arm, ex, ey = dif._detector_frame(angles)
    if space == "lab":
        return arm, ex, ey, np.eye(3)
    S = dif.GEOMETRY.sample_matrix(dif._complete(angles))
    if space == "q":
        return arm, ex, ey, S.T
    if space == "hkl":
        if sample is None:
            raise ValueError("space='hkl' needs a sample (its UB)")
        return arm, ex, ey, np.linalg.solve(sample.UB, S.T)
    raise ValueError(f"unknown space {space!r}")


# ---------------------------------------------------------------- pyFAI PONI <-> detector arm

def _poni_matrix(rot1, rot2, rot3):
    """Columns: the lab directions of pyFAI's detector axis 1 (rows), axis 2
    (columns) and of the normal from the sample (the arm), for pyFAI's
    rotations (pyFAI's calc_pos_zyx; its x, y, z are SwissFEL's x, y, z)."""
    c1, s1, c2, s2, c3, s3 = np.cos(rot1), np.sin(rot1), np.cos(rot2), np.sin(rot2), np.cos(rot3), np.sin(rot3)
    return np.array([[c2 * s3, c1 * c3 + s1 * s2 * s3, c3 * s1 - c1 * s2 * s3],
                     [c2 * c3, c3 * s1 * s2 - c1 * s3, -(c1 * c3 * s2 + s1 * s3)],
                     [s2, -c2 * s1, c1 * c2]])


def _orientation_mirror(orientation):
    """pyFAI's detector orientation (1-4; 0 = 3): (flip rows, flip columns)
    and the lab mirror it implies. pyFAI counts a flipped axis from its far
    edge, rotates as for orientation 3, and then mirrors lab y (rows) and/or
    lab x (columns) - checked against pyFAI's position_array."""
    if orientation not in (0, 1, 2, 3, 4):
        raise ValueError(f"unknown orientation {orientation}")
    f1, f2 = orientation in (1, 2), orientation in (1, 4)
    return f1, f2, np.array([-1.0 if f2 else 1.0, -1.0 if f1 else 1.0, 1.0])


def poni_to_frame(dist, poni1, poni2, rot1=0.0, rot2=0.0, rot3=0.0, orientation=3):
    """pyFAI geometry (m, rad) -> (distance [mm], arm, ex, ey, foot).

    ``ex`` runs along the image columns (pyFAI axis 2, our u), ``ey`` along
    the rows (axis 1, v); ``foot`` = (poni1, poni2) in mm is where the arm
    meets the face, measured from the corner of the first pixel (for a
    flipped axis, of the last one) - the face coordinates of each pixel come
    from :func:`poni_pixels`. With orientation 1, 2 or 4 the frame is
    mirrored (ex x ey points against the arm)."""
    M = _poni_matrix(rot1, rot2, rot3) * _orientation_mirror(orientation)[2][:, None]
    return dist * 1e3, M[:, 2].copy(), M[:, 1].copy(), M[:, 0].copy(), (poni1 * 1e3, poni2 * 1e3)


def frame_to_poni(distance, arm, ex, ey, foot, orientation=3):
    """Inverse of :func:`poni_to_frame`: dict(dist, poni1, poni2, rot1, rot2,
    rot3) in m/rad. ``foot`` (mm): where the arm meets the face, from the
    first pixel's corner along (rows, columns); a face centred on the arm,
    as :func:`detector_pixels`, has foot = (height / 2, width / 2).
    ``orientation``: pyFAI's detector orientation to express it in."""
    S = _orientation_mirror(orientation)[2]
    arm, ex, ey = (S * np.asarray(a, float) for a in (arm, ex, ey))
    if np.linalg.det(np.stack([ey, ex, arm], 1)) > 0:
        raise ValueError("this frame is mirrored for orientation %d: ex x ey must point along the arm "
                         "(orientation 3); use orientation 2 or 4 for a mirrored one" % orientation)
    rot2 = np.arcsin(np.clip(ey[2], -1, 1))
    rot1 = np.arctan2(-ex[2], arm[2])
    rot3 = np.arctan2(ey[0], ey[1])
    M = _poni_matrix(rot1, rot2, rot3)
    if not np.allclose(M, np.stack([ey, ex, arm], 1), atol=1e-9):
        raise ValueError("arm, ex, ey are not an orthonormal detector frame")
    return dict(dist=distance * 1e-3, poni1=foot[0] * 1e-3, poni2=foot[1] * 1e-3,
                rot1=float(rot1), rot2=float(rot2), rot3=float(rot3))


def poni_pixels(shape, pixel1, pixel2, poni1, poni2, orientation=3):
    """Face coordinates (u, v) [mm] of every pixel of an image ``shape``
    (rows, columns), in image order, for pyFAI pixel sizes and PONI (m).
    ``orientation`` as pyFAI's detectors (3, the default; 1 both axes, 2
    the rows, 4 the columns counted from the far edge; the matching mirror is
    in :func:`poni_to_frame`'s frame)."""
    n1, n2 = shape
    f1, f2, _ = _orientation_mirror(orientation)

    def axis(n, pixel, poni, flip):            # a flipped axis counts from its far edge
        idx = np.arange(n, dtype=float)
        return (((n - 1 - idx) if flip else idx) + 0.5 - poni / pixel) * pixel * 1e3
    v = axis(n1, pixel1, poni1, f1)
    u = axis(n2, pixel2, poni2, f2)
    U, V = np.meshgrid(u, v)
    return np.stack([U.ravel(), V.ravel()], -1)


def read_poni(path):
    """A pyFAI .poni file (versions 1-3) as a dict: dist, poni1, poni2,
    rot1-3 (m, rad), wavelength (m or None), pixel1, pixel2 (m), shape,
    orientation. No pyFAI needed."""
    import json
    raw = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        raw[key.strip().lower()] = val.strip()
    out = {k: float(raw[k]) for k in ("distance", "poni1", "poni2", "rot1", "rot2", "rot3") if k in raw}
    out["dist"] = out.pop("distance")
    out["wavelength"] = float(raw["wavelength"]) if "wavelength" in raw else None
    cfg = json.loads(raw["detector_config"]) if "detector_config" in raw else {}
    out["pixel1"] = float(cfg.get("pixel1", raw.get("pixelsize1", "nan")))
    out["pixel2"] = float(cfg.get("pixel2", raw.get("pixelsize2", "nan")))
    shape = cfg.get("max_shape")
    out["shape"] = tuple(shape) if shape else None
    out["orientation"] = int(cfg.get("orientation", 3) or 3)
    out["detector"] = raw.get("detector")
    return out


def poni_geometry(poni):
    """Everything the binners need from a pyFAI calibration: a .poni path, a
    dict as :func:`read_poni`, or a pyFAI AzimuthalIntegrator. Returns
    dict(uv, shape, distance, arm, ex, ey, k) (mm, 1/Angstrom; k None
    without a wavelength)."""
    if hasattr(poni, "getPyFAI"):                       # an AzimuthalIntegrator / Geometry
        det = poni.detector
        p = dict(dist=poni.dist, poni1=poni.poni1, poni2=poni.poni2, rot1=poni.rot1, rot2=poni.rot2,
                 rot3=poni.rot3, wavelength=poni.wavelength, pixel1=det.pixel1, pixel2=det.pixel2,
                 shape=tuple(det.max_shape), orientation=int(getattr(det, "orientation", 3) or 3))
    elif isinstance(poni, dict):
        p = poni
    else:
        p = read_poni(poni)
    if p.get("shape") is None:
        raise ValueError("the PONI has no detector shape (max_shape)")
    distance, arm, ex, ey, _ = poni_to_frame(p["dist"], p["poni1"], p["poni2"], p["rot1"], p["rot2"], p["rot3"],
                                             p.get("orientation", 3))
    uv = poni_pixels(p["shape"], p["pixel1"], p["pixel2"], p["poni1"], p["poni2"], p.get("orientation", 3))
    k = 2 * np.pi / (p["wavelength"] * 1e10) if p.get("wavelength") else None
    return dict(uv=uv, shape=tuple(p["shape"]), distance=distance, arm=arm, ex=ex, ey=ey, k=k)


def pixel_coordinates(uv, distance, arm, ex, ey, k=None, beam=BEAM):
    """2theta and chi [deg] of every pixel (chi as pyFAI: atan2(y, x) of the
    scattered direction, so 0 along x+, 90 up), and abs(Q) [1/Angstrom] if
    ``k`` is given: the coordinates for azimuthal integration at any pose
    (:meth:`FixedBinner.from_coords`)."""
    uv = np.asarray(uv, float)
    P = distance * np.asarray(arm, float) + uv[:, :1] * np.asarray(ex, float) + uv[:, 1:] * np.asarray(ey, float)
    n = P / np.linalg.norm(P, axis=1, keepdims=True)
    cos2t = np.clip(n @ np.asarray(beam, float), -1, 1)
    out = dict(tth=np.degrees(np.arccos(cos2t)), chi=np.degrees(np.arctan2(n[:, 1], n[:, 0])))
    if k is not None:
        out["q"] = 2 * k * np.sin(np.arccos(cos2t) / 2)
    return out


def pixel_corrections(uv, distance, arm, ex, ey, solid_angle=True, polarization=1.0):
    """Per-pixel solid angle (d/r)^3 (1 at the arm) times the polarization
    factor; ``polarization``: degree of horizontal (x) polarization as
    pyFAI's ``polarization_factor`` (1 SwissFEL, None: no correction)."""
    uv = np.asarray(uv, float)
    P = distance * np.asarray(arm) + uv[:, :1] * np.asarray(ex) + uv[:, 1:] * np.asarray(ey)
    r = np.linalg.norm(P, axis=1)
    c = (distance / r) ** 3 if solid_angle else np.ones(len(uv))
    if polarization is not None:
        nx, ny = P[:, 0] / r, P[:, 1] / r
        c = c * (0.5 * (1 + polarization) * (1 - nx ** 2) + 0.5 * (1 - polarization) * (1 - ny ** 2))
    return c


def _subpixels(split, pixel):
    su, sv = np.broadcast_to(np.asarray(split, int), (2,))
    if su * sv == 1:
        return [(0.0, 0.0)]
    if pixel is None:
        raise ValueError("pixel splitting needs the pixel size")
    pu, pv = np.broadcast_to(np.asarray(pixel, float), (2,))
    return [(((a + 0.5) / su - 0.5) * pu, ((b + 0.5) / sv - 0.5) * pv) for b in range(sv) for a in range(su)]


# ====================================================================== numba kernels

if HAVE_NUMBA:
    _jit = dict(fastmath=_FM, cache=True, nogil=True)

    @nb.njit(parallel=True, **_jit)
    def _pose_index(u_, v_, du, dv, dist, arm, ex, ey, k, beam, M, off, n, sa, use_pol, pol, index, corr):
        """Voxel of every (sub)pixel for one pose, and its correction factor.

        (u + du, v + dv) [mm] -> lab point d*arm + u*ex + v*ey -> unit kout
        -> q_lab = k (kout - beam) -> g = M q_lab + off (grid units) -> flat
        voxel, -1 outside. Branchless where it matters so the loop vectorizes.
        """
        n1, n2 = n[1], n[2]
        f0, f1, f2 = float(n[0]), float(n[1]), float(n[2])
        a0, a1, a2 = dist * arm[0], dist * arm[1], dist * arm[2]
        want = corr.size > 0
        hp, hm = 0.5 * (1 + pol), 0.5 * (1 - pol)
        for p in nb.prange(u_.size):
            u = u_[p] + du
            v = v_[p] + dv
            x = a0 + u * ex[0] + v * ey[0]
            y = a1 + u * ex[1] + v * ey[1]
            z = a2 + u * ex[2] + v * ey[2]
            rr = np.sqrt(x * x + y * y + z * z)
            r = k / rr
            qx = x * r - k * beam[0]
            qy = y * r - k * beam[1]
            qz = z * r - k * beam[2]
            g0 = M[0, 0] * qx + M[0, 1] * qy + M[0, 2] * qz + off[0]
            g1 = M[1, 0] * qx + M[1, 1] * qy + M[1, 2] * qz + off[1]
            g2 = M[2, 0] * qx + M[2, 1] * qy + M[2, 2] * qz + off[2]
            ok = (g0 >= 0) & (g0 < f0) & (g1 >= 0) & (g1 < f1) & (g2 >= 0) & (g2 < f2)
            i0 = np.int64(min(max(g0, 0.0), f0 - 1))
            i1 = np.int64(min(max(g1, 0.0), f1 - 1))
            i2 = np.int64(min(max(g2, 0.0), f2 - 1))
            index[p] = (i0 * n1 + i1) * n2 + i2 if ok else -1
            if want:
                c = 1.0
                if sa:
                    t = dist / rr
                    c = t * t * t
                if use_pol:
                    nx = x / rr
                    ny = y / rr
                    c *= hp * (1 - nx * nx) + hm * (1 - ny * ny)
                corr[p] = c

    @nb.njit(**_jit)
    def _accumulate(index, corr, frame, weight, count, inv, sig, norm):
        """sig[i] += x / nsub, norm[i] += weight * corr * count / nsub over
        the (sub)pixels; NaN pixels skipped. ``corr`` (0, 0) and ``count``
        (0,) mean all ones."""
        hc = corr.shape[0] > 0
        hn = count.shape[0] > 0
        for j in range(index.shape[0]):
            for p in range(index.shape[1]):
                i = index[j, p]
                x = frame[p]
                if i >= 0 and x == x:
                    c = weight[p] * inv
                    if hc:
                        c *= corr[j, p]
                    if hn:
                        c *= count[p]
                    sig[i] += x * inv
                    norm[i] += c

    @nb.njit(parallel=True, **_jit)
    def _accumulate_par(index, corr, frame, weight, count, inv, sig, norm, nthreads):
        """As _accumulate with private grids per thread: pays only for small grids."""
        hc = corr.shape[0] > 0
        hn = count.shape[0] > 0
        nbins = sig.size
        ps = np.zeros((nthreads, nbins))
        pn = np.zeros((nthreads, nbins))
        npx = index.shape[1]
        for t in nb.prange(nthreads):
            for j in range(index.shape[0]):
                for p in range(t * npx // nthreads, (t + 1) * npx // nthreads):
                    i = index[j, p]
                    x = frame[p]
                    if i >= 0 and x == x:
                        c = weight[p] * inv
                        if hc:
                            c *= corr[j, p]
                        if hn:
                            c *= count[p]
                        ps[t, i] += x * inv
                        pn[t, i] += c
        for b in nb.prange(nbins):
            for t in range(nthreads):
                sig[b] += ps[t, b]
                norm[b] += pn[t, b]

    @nb.njit(parallel=True, **_jit)
    def _group_add(buf, count, frame):
        for p in nb.prange(frame.size):
            x = frame[p]
            if x == x:
                buf[p] += x
                count[p] += 1

    # Fixed geometry: the normalization without NaN is computed once
    # (FixedBinner.norm); the kernels only collect what NaN pixels take away
    # from it ("lost"), so a frame's normalization costs writes only for its
    # NaN pixels instead of one per pixel.

    @nb.njit(parallel=True, **_jit)
    def _fixed_scatter(index, corr, stack, sig, lost):
        """One voxel per pixel (weight 1), parallel over frames."""
        hc = corr.size > 0
        for f in nb.prange(stack.shape[0]):
            fr = stack[f]
            s = sig[f]
            o = lost[f]
            for p in range(fr.size):
                i = index[p]
                x = fr[p]
                if i >= 0:
                    if x == x:
                        s[i] += x
                    else:
                        o[i] += corr[p] if hc else 1.0

    @nb.njit(parallel=True, **_jit)
    def _fixed_scatter_chunks(index, corr, frame, sig, lost, nthreads):
        """One frame, one voxel per pixel: pixels in nthreads chunks with private outputs."""
        hc = corr.size > 0
        nbins = sig.size
        ps = np.zeros((nthreads, nbins))
        pl = np.zeros((nthreads, nbins))
        npx = frame.size
        for t in nb.prange(nthreads):
            for p in range(t * npx // nthreads, (t + 1) * npx // nthreads):
                i = index[p]
                x = frame[p]
                if i >= 0:
                    if x == x:
                        ps[t, i] += x
                    else:
                        pl[t, i] += corr[p] if hc else 1.0
        for b in nb.prange(nbins):
            a = 0.0
            m = 0.0
            for t in range(nthreads):
                a += ps[t, b]
                m += pl[t, b]
            sig[b] = a
            lost[b] = m

    @nb.njit(parallel=True, **_jit)
    def _fixed_csc(colptr, rows, data, corr, stack, sig, lost):
        """Pixel-major sparse matrix (pixel splitting), parallel over frames."""
        hc = corr.size > 0
        for f in nb.prange(stack.shape[0]):
            fr = stack[f]
            s = sig[f]
            o = lost[f]
            for p in range(fr.size):
                x = fr[p]
                if x == x:
                    for j in range(colptr[p], colptr[p + 1]):
                        s[rows[j]] += data[j] * x
                else:
                    c = corr[p] if hc else 1.0
                    for j in range(colptr[p], colptr[p + 1]):
                        o[rows[j]] += data[j] * c

    @nb.njit(parallel=True, **_jit)
    def _fixed_csc_chunks(colptr, rows, data, corr, frame, sig, lost, nthreads):
        """One frame: pixels in nthreads chunks with private outputs."""
        hc = corr.size > 0
        nbins = sig.size
        ps = np.zeros((nthreads, nbins))
        pl = np.zeros((nthreads, nbins))
        npx = frame.size
        for t in nb.prange(nthreads):
            for p in range(t * npx // nthreads, (t + 1) * npx // nthreads):
                x = frame[p]
                if x == x:
                    for j in range(colptr[p], colptr[p + 1]):
                        ps[t, rows[j]] += data[j] * x
                else:
                    c = corr[p] if hc else 1.0
                    for j in range(colptr[p], colptr[p + 1]):
                        pl[t, rows[j]] += data[j] * c
        for b in nb.prange(nbins):
            a = 0.0
            m = 0.0
            for t in range(nthreads):
                a += ps[t, b]
                m += pl[t, b]
            sig[b] = a
            lost[b] = m


def _pose_index_xp(xp, u, v, du, dv, dist, arm, ex, ey, k, beam, M, off, n, sa, pol, want):
    """Array version of _pose_index (numpy or cupy)."""
    uu, vv = u + du, v + dv
    x = dist * arm[0] + uu * ex[0] + vv * ey[0]
    y = dist * arm[1] + uu * ex[1] + vv * ey[1]
    z = dist * arm[2] + uu * ex[2] + vv * ey[2]
    rr = xp.sqrt(x * x + y * y + z * z)
    q = [x * (k / rr) - k * beam[0], y * (k / rr) - k * beam[1], z * (k / rr) - k * beam[2]]
    g = [xp.floor(M[i, 0] * q[0] + M[i, 1] * q[1] + M[i, 2] * q[2] + off[i]) for i in range(3)]
    ok = (g[0] >= 0) & (g[0] < n[0]) & (g[1] >= 0) & (g[1] < n[1]) & (g[2] >= 0) & (g[2] < n[2])
    g = [xp.where(ok, gi, 0).astype(xp.int64) for gi in g]
    idx = xp.where(ok, (g[0] * int(n[1]) + g[1]) * int(n[2]) + g[2], -1).astype(xp.int32)
    if not want:
        return idx, None
    c = (dist / rr) ** 3 if sa else xp.ones_like(rr)
    if pol is not None:
        c = c * (0.5 * (1 + pol) * (1 - (x / rr) ** 2) + 0.5 * (1 - pol) * (1 - (y / rr) ** 2))
    return idx, c.astype(xp.float32)


# ====================================================================== fixed geometry

class FixedBinner:
    """A fixed (bins x pixels) matrix with per-pixel corrections.

    ``bin(frames)`` gives the sums of signal and of normalization per bin,
    ``mean(frames)`` their ratio shaped as ``out_shape``. Frames may be one
    image or a stack (nframes, ...) as it comes from the detector.
    """

    def __init__(self, data, indices, indptr, npix, shape=None, corrections=None,
                 out_shape=None, axes=None, backend="auto"):
        import scipy.sparse as sp
        self.npix = int(npix)
        self.shape = tuple(shape) if shape is not None else (self.npix,)
        self.A = sp.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32),
                                np.asarray(indptr, np.int64)), shape=(len(indptr) - 1, self.npix))
        self.A.sum_duplicates()
        self.nbins = self.A.shape[0]
        self.out_shape = tuple(out_shape) if out_shape is not None else (self.nbins,)
        self.axes = axes
        self.corrections = None if corrections is None else np.ascontiguousarray(
            np.asarray(corrections, np.float32).ravel())
        self.backend = _backend(backend)
        c = np.ones(self.npix, np.float32) if self.corrections is None else self.corrections
        self.norm = np.asarray(self.A @ c.astype(np.float64)).ravel()   # without NaN pixels
        self.pixel_index = None       # set when every pixel goes to one bin with weight 1
        self._csc = None
        self._dev = None

    # ---- constructors
    @classmethod
    def from_index(cls, index, nbins, shape=None, corrections=None, **kw):
        """One bin per pixel (-1: not used)."""
        index = np.asarray(index).ravel()
        use = np.flatnonzero(index >= 0)
        order = use[np.argsort(index[use], kind="stable")]
        indptr = np.zeros(nbins + 1, np.int64)
        np.cumsum(np.bincount(index[order], minlength=nbins), out=indptr[1:])
        fb = cls(np.ones(order.size, np.float32), order, indptr, index.size, shape, corrections, **kw)
        fb.pixel_index = np.ascontiguousarray(index, np.int32)
        return fb

    @classmethod
    def from_coords(cls, coords, grid: Grid, shape=None, corrections=None, **kw):
        """Pixel coordinates (npix, ndim), e.g. 2theta/chi/abs(Q), on a Grid."""
        return cls.from_index(grid.index(coords), grid.size, shape, corrections,
                              out_shape=grid.shape, axes=grid.centres, **kw)

    @classmethod
    def from_pyfai(cls, ai, npt, shape, unit="2th_deg", split="bbox", mask=None,
                   solid_angle=True, polarization=1.0, **kw):
        """pyFAI's sparse matrix for ``ai`` (an AzimuthalIntegrator):
        ``npt`` an int (1D) or (radial, azimuthal) (2D), ``split`` "no",
        "bbox" or "full". Corrections from pyFAI as well (polarization:
        pyFAI's polarization_factor, None: none)."""
        eng = ai.setup_sparse_integrator(shape, npt, mask=mask, unit=unit, split=split, algo="CSR", scale=False)
        corr = np.ones(shape, np.float32)
        if solid_angle:
            corr = corr * ai.solidAngleArray(shape)
        if polarization is not None:
            corr = corr * np.asarray(ai.polarization(shape, factor=polarization))
        out = tuple(np.atleast_1d(npt))
        axes = None
        try:
            axes = [np.asarray(eng.bin_centers)] if len(out) == 1 else [np.asarray(eng.bin_centers0),
                                                                       np.asarray(eng.bin_centers1)]
        except AttributeError:
            pass
        return cls(eng.data, eng.indices, eng.indptr, int(np.prod(shape)), shape, corr,
                   out_shape=out, axes=axes, **kw)

    @classmethod
    def from_pose(cls, uv, k, grid: Grid, distance, arm, ex, ey, R=None, shape=None, pixel=None,
                  split=1, solid_angle=True, polarization=1.0, beam=BEAM, **kw):
        """One fixed diffractometer pose into a 3D grid (q or hkl via ``R``,
        see :func:`shot_geometry`), with ``split`` x ``split`` subpixels."""
        uv = np.asarray(uv, float)
        subs = _subpixels(split, pixel)
        M, off = grid.affine(R)
        rows, cols, corr = [], [], np.zeros(len(uv))
        for du, dv in subs:
            idx, c = _pose_index_xp(np, uv[:, 0], uv[:, 1], du, dv, float(distance), np.asarray(arm, float),
                                    np.asarray(ex, float), np.asarray(ey, float), float(k), np.asarray(beam, float),
                                    M, off, grid.n, solid_angle, polarization, True)
            corr += c / len(subs)
            ok = idx >= 0
            rows.append(idx[ok])
            cols.append(np.flatnonzero(ok))
        import scipy.sparse as sp
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        A = sp.csr_matrix((np.full(rows.size, 1.0 / len(subs), np.float32), (rows, cols)), shape=(grid.size, len(uv)))
        fb = cls(A.data, A.indices, A.indptr, len(uv), shape, corr, out_shape=grid.shape, axes=grid.centres, **kw)
        if len(subs) == 1:
            fb.pixel_index = np.full(len(uv), -1, np.int32)
            fb.pixel_index[cols] = rows
        return fb

    # ---- application
    @property
    def csc(self):
        if self._csc is None:
            m = self.A.tocsc()
            m.sort_indices()
            self._csc = (m.indptr.astype(np.int64), m.indices.astype(np.int32), m.data.astype(np.float32))
        return self._csc

    def _stack(self, frames, xp=np):
        single = frames.ndim == 1 or tuple(frames.shape) == self.shape
        n = 1 if single else frames.shape[0]
        return single, xp.ascontiguousarray(frames.reshape(n, self.npix), dtype=xp.float32)

    def bin(self, frames, threads=None):
        """(sums of signal, sums of normalization) per bin: each (nbins,)
        for one frame, (nframes, nbins) for a stack."""
        if self.backend == "numba":
            single, stack = self._stack(np.asarray(frames))
            corr = _NONE_F if self.corrections is None else self.corrections
            threads = threads or nb.get_num_threads()
            sig = np.zeros((stack.shape[0], self.nbins))
            lost = np.zeros_like(sig)
            if self.pixel_index is not None and stack.shape[0] >= threads:
                _fixed_scatter(self.pixel_index, corr, stack, sig, lost)
            elif stack.shape[0] >= threads:
                _fixed_csc(*self.csc, corr, stack, sig, lost)
            elif self.pixel_index is not None:
                for f in range(stack.shape[0]):
                    _fixed_scatter_chunks(self.pixel_index, corr, stack[f], sig[f], lost[f], threads)
            else:
                for f in range(stack.shape[0]):
                    _fixed_csc_chunks(*self.csc, corr, stack[f], sig[f], lost[f], threads)
            np.subtract(self.norm, lost, out=lost)
            return (sig[0], lost[0]) if single else (sig, lost)
        xp = _xp(self.backend)
        if self._dev is None:
            if self.backend == "cupy":
                import cupyx.scipy.sparse as csp
                A = csp.csr_matrix(self.A)
            else:
                A = self.A
            c = None if self.corrections is None else xp.asarray(self.corrections)
            self._dev = (A, c)
        A, c = self._dev
        single, stack = self._stack(xp.asarray(frames), xp)
        valid = xp.isfinite(stack)
        x = xp.where(valid, stack, 0)
        w = valid.astype(xp.float32) if c is None else valid * c[None, :]
        sig, norm = xp.asarray(A @ x.T).T, xp.asarray(A @ w.T).T
        return (sig[0], norm[0]) if single else (sig, norm)

    def mean(self, frames, threads=None):
        """Corrected mean per bin, shaped ``(..., *out_shape)``; NaN where empty."""
        sig, norm = self.bin(frames, threads)
        xp = _xp(self.backend)
        # empty: nothing, or only the rounding residue of norm - lost, relative to the bin's own norm
        empty = norm <= 1e-7 * xp.asarray(self.norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = sig / norm
        m[empty] = xp.nan
        return m.reshape(m.shape[:-1] + self.out_shape)

    # ---- dask / escape
    def mean_dask(self, stack):
        """Per-frame means of a dask stack ``(nframes, *shape)``, lazily."""
        import dask.array as da
        stack = da.asarray(stack)
        nd = stack.ndim - 1
        stack = stack.rechunk({i: -1 for i in range(1, stack.ndim)})
        out_nd = len(self.out_shape)
        kw = {}
        if out_nd < nd:
            kw["drop_axis"] = list(range(1 + out_nd, 1 + nd))
        elif out_nd > nd:
            kw["new_axis"] = list(range(1 + nd, 1 + out_nd))
        return stack.map_blocks(lambda b: _to_numpy(self.mean(b)).reshape((len(b),) + self.out_shape),
                                chunks=(stack.chunks[0],) + tuple((s,) for s in self.out_shape),
                                dtype=float, **kw)

    def apply(self, array):
        """An escape.Array of images -> an escape.Array of per-shot means
        (same pulse index and scan). An output with more dimensions than
        the image (3D from 2D) comes flat: reshape to ``out_shape``."""
        nd = array.data.ndim - 1
        out = self.out_shape if len(self.out_shape) <= nd else (self.nbins,)
        drop = list(range(1 + len(out), 1 + nd)) or None
        return array.map_index_blocks(lambda b: _to_numpy(self.mean(b)).reshape((len(b),) + out),
                                      drop_axis=drop, new_element_size=out, dtype=float)


# ====================================================================== changing geometry

def scan_grid(uv, k, poses, voxel, pad=0.5, stride=8, beam=BEAM):
    """The voxel grid (:meth:`Grid.covering`) holding every pixel of every
    pose: ``poses`` an iterable of (distance, arm, ex, ey, R). Uses every
    ``stride``-th pixel plus the detector's edge."""
    uv = np.asarray(uv, float)
    u, v = uv[:, 0], uv[:, 1]
    edge = (u <= u.min()) | (u >= u.max()) | (v <= v.min()) | (v >= v.max())
    sel = uv[(np.arange(len(uv)) % stride == 0) | edge]
    pts = []
    for dist, arm, ex, ey, R in poses:
        P = dist * np.asarray(arm) + sel[:, :1] * np.asarray(ex) + sel[:, 1:] * np.asarray(ey)
        q = k * (P / np.linalg.norm(P, axis=1, keepdims=True) - np.asarray(beam))
        pts.append(q @ np.asarray(R, float).T)
    return Grid.covering(np.concatenate(pts), voxel, pad)


class StreamBinner:
    """Accumulate shots with their own geometry into a 3D :class:`Grid`.

    ``uv``: the pixels' face coordinates (npix, 2) in the order of the
    flattened image (:func:`detector_pixels`, :func:`jungfrau_pixels`).
    Per shot (:meth:`add`): the detector frame (``distance``, ``arm``,
    ``ex``, ``ey``) and ``R`` taking lab q to the output space
    (:func:`shot_geometry`, or :meth:`add_angles` with diffractometer angles).

    Options: ``solid_angle`` / ``polarization`` corrections (see
    :func:`pixel_corrections`), ``weight`` per pixel (0 masks), ``split``
    subpixels per pixel and axis (needs ``pixel``), ``tolerance`` (voxels,
    at the detector's corners and centre) within which a shot reuses the
    previous voxel indices, ``group`` to sum those shots first and bin once
    (geometry of the first shot of a group).
    """

    def __init__(self, uv, k, grid: Grid, *, pixel=None, weight=None, solid_angle=True, polarization=1.0,
                 split=1, tolerance=0.0, group=False, backend="auto", threads=None, beam=BEAM):
        if grid.ndim != 3:
            raise ValueError("StreamBinner bins into a 3D grid")
        self.backend = _backend(backend)
        self.xp = xp = _xp(self.backend)
        uv = np.asarray(uv, float)
        self.uv = uv
        self.npix = len(uv)
        self.k, self.grid, self.beam = float(k), grid, np.asarray(beam, float)
        self.solid_angle, self.polarization = bool(solid_angle), polarization
        self.want_corr = self.solid_angle or polarization is not None
        self.subs = _subpixels(split, pixel)
        self.tolerance, self.group = float(tolerance), bool(group)
        self.threads = threads or (nb.get_num_threads() if HAVE_NUMBA else 1)
        self._u = xp.asarray(np.ascontiguousarray(uv[:, 0]))
        self._v = xp.asarray(np.ascontiguousarray(uv[:, 1]))
        self.weight = xp.asarray(np.ones(self.npix, np.float32) if weight is None
                                 else np.ascontiguousarray(np.asarray(weight, np.float32).ravel()))
        nsub = len(self.subs)
        self.index = xp.empty((nsub, self.npix), xp.int32)
        self.corr = xp.empty((nsub, self.npix), xp.float32) if self.want_corr else None
        u, v = uv[:, 0], uv[:, 1]
        self._probes = np.array([[u.min(), v.min()], [u.min(), v.max()], [u.max(), v.min()],
                                 [u.max(), v.max()], [0.0, 0.0]], float)
        self.reset()

    def reset(self):
        xp = self.xp
        self.sig = xp.zeros(self.grid.size)
        self.norm = xp.zeros(self.grid.size)
        self._last = None
        self._buf = None
        self._count = None
        self._nbuf = 0
        self.stats = dict(shots=0, index_builds=0, bin_passes=0)

    def fresh(self):
        """An empty binner with the same configuration (shares pixel arrays)."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.index = self.xp.empty_like(self.index)
        new.corr = None if self.corr is None else self.xp.empty_like(self.corr)
        new.reset()
        return new

    # ---- geometry
    def _probe(self, geo):
        dist, arm, ex, ey, M, off = geo
        P = dist * arm + self._probes[:, :1] * ex + self._probes[:, 1:] * ey
        q = self.k * (P / np.linalg.norm(P, axis=1, keepdims=True) - self.beam)
        return q @ M.T + off

    def _build_index(self, geo):
        dist, arm, ex, ey, M, off = geo
        pol = self.polarization
        for j, (du, dv) in enumerate(self.subs):
            if self.backend == "numba":
                _pose_index(self._u, self._v, du, dv, dist, arm, ex, ey, self.k, self.beam, M, off, self.grid.n,
                            self.solid_angle, pol is not None, 0.0 if pol is None else float(pol),
                            self.index[j], _NONE_F if self.corr is None else self.corr[j])
            else:
                idx, c = _pose_index_xp(self.xp, self._u, self._v, du, dv, dist, arm, ex, ey, self.k, self.beam,
                                        M, off, self.grid.n, self.solid_angle, pol, self.want_corr)
                self.index[j] = idx
                if c is not None:
                    self.corr[j] = c
        self._last = geo
        if self.tolerance > 0:
            self._last_probe = self._probe(geo)
        self.stats["index_builds"] += 1

    def _bin(self, frame, count=None):
        inv = 1.0 / len(self.subs)
        if self.backend == "numba":
            corr = _NONE_F2 if self.corr is None else self.corr
            cnt = _NONE_F if count is None else count
            if self.threads > 1 and self.grid.size * self.threads * 4 < self.npix * len(self.subs):
                _accumulate_par(self.index, corr, frame, self.weight, cnt, inv, self.sig, self.norm, self.threads)
            else:
                _accumulate(self.index, corr, frame, self.weight, cnt, inv, self.sig, self.norm)
        else:
            xp = self.xp
            valid = xp.isfinite(frame)
            x = xp.where(valid, frame, 0)
            w = self.weight * valid if count is None else self.weight * count
            for j in range(len(self.subs)):
                idx = self.index[j]
                use = idx >= 0
                c = w if self.corr is None else w * self.corr[j]
                self.sig += xp.bincount(idx[use], weights=(x * inv)[use], minlength=self.grid.size)
                self.norm += xp.bincount(idx[use], weights=(c * inv)[use], minlength=self.grid.size)
        self.stats["bin_passes"] += 1

    def flush(self):
        """Bin the summed shots of an open group."""
        if self._nbuf:
            self._bin(self._buf, self._count)
            self._buf[:] = 0
            self._count[:] = 0
            self._nbuf = 0

    # ---- input
    def add(self, frame, distance, arm, ex, ey, R=None):
        """One shot: ``frame`` the image (flattened in the order of ``uv``)."""
        xp = self.xp
        frame = xp.ascontiguousarray(xp.asarray(frame).reshape(-1), dtype=xp.float32)
        M, off = self.grid.affine(R)
        geo = (float(distance), np.asarray(arm, float), np.asarray(ex, float), np.asarray(ey, float), M, off)
        self.stats["shots"] += 1
        if self._last is None or self.tolerance <= 0 \
                or np.abs(self._probe(geo) - self._last_probe).max() > self.tolerance:
            self.flush()
            self._build_index(geo)
        if self.group:
            if self._buf is None:
                self._buf = xp.zeros(self.npix, xp.float32)
                self._count = xp.zeros(self.npix, xp.float32)
            if self.backend == "numba":
                _group_add(self._buf, self._count, frame)
            else:
                valid = xp.isfinite(frame)
                self._buf += xp.where(valid, frame, 0)
                self._count += valid
            self._nbuf += 1
        else:
            self._bin(frame)

    def add_angles(self, frame, angles, distance, sample=None, space="hkl"):
        """One shot at diffractometer ``angles`` (see :func:`shot_geometry`)."""
        self.add(frame, distance, *shot_geometry(angles, sample, space))

    def __iadd__(self, other):
        """Merge another binner's sums (same grid)."""
        self.flush()
        other.flush()
        self.sig += other.sig
        self.norm += other.norm
        for key in self.stats:
            self.stats[key] += other.stats[key]
        return self

    # ---- many shots: numpy stacks, dask, escape
    def reduce(self, frames, distance, arm, ex, ey, R=None, split_every=8, **compute_kw):
        """Add many shots at once. ``frames`` (nframes, ...) numpy or dask;
        each geometry argument either one value for all shots or one per
        shot (``arm`` (n, 3), ``R`` (n, 3, 3), ...). A dask stack is binned
        chunk by chunk in parallel (each chunk with a fresh binner) and the
        partial grids are summed in a tree; ``compute_kw`` go to dask's
        compute (e.g. ``scheduler``). Returns self."""
        n = len(frames)
        geo = _per_shot(n, distance, arm, ex, ey, R)
        try:
            import dask.array as da
            is_dask = isinstance(frames, da.Array)
        except ImportError:  # pragma: no cover
            is_dask = False
        if not is_dask:
            for i in range(n):
                self.add(frames[i], *(g[i] for g in geo))
            return self
        import dask
        frames = frames.reshape(n, -1).rechunk({1: -1})
        template = self.fresh()
        starts = np.concatenate([[0], np.cumsum(frames.chunks[0])])
        parts = []
        for b, blk in enumerate(frames.to_delayed().ravel()):
            sl = slice(starts[b], starts[b + 1])
            part = dask.delayed(_partial)(template, blk, [g[sl] for g in geo])
            parts.append(da.from_delayed(part, (2, self.grid.size), float))
        sums = da.stack(parts).sum(axis=0, split_every=split_every).compute(**compute_kw)
        self.flush()
        self.sig += self.xp.asarray(sums[0])
        self.norm += self.xp.asarray(sums[1])
        self.stats["shots"] += n
        return self

    def reduce_escape(self, frames, angles, distance, sample=None, space="hkl", **compute_kw):
        """Add the shots of an escape.Array of images. ``angles``: {name:
        escape.Array (per pulse) or number}; arrays are matched to the
        frames by pulse ID first. Repeated angle values are converted once."""
        import escape
        names = [n for n, a in angles.items() if hasattr(a, "index")]
        matched = escape.match_arrays(frames, *(angles[n] for n in names)) if names else (frames,)
        frames = matched[0]
        vals = {n: np.asarray(_compute(a.data)).ravel() for n, a in zip(names, matched[1:])}
        fixed = {n: a for n, a in angles.items() if n not in vals}
        cache, geos = {}, []
        for i in range(len(frames.index)):
            key = tuple(float(vals[n][i]) for n in names)
            if key not in cache:
                cache[key] = shot_geometry({**fixed, **dict(zip(names, key))}, sample, space)
            geos.append(cache[key])
        arm, ex, ey, R = (np.array([g[j] for g in geos]) for j in range(4))
        data = frames.data
        if not hasattr(data, "to_delayed"):
            data = np.asarray(data)
        return self.reduce(data, distance, arm, ex, ey, R, **compute_kw)

    # ---- output
    def sums(self):
        """(signal, normalization) sums per voxel, shaped as the grid, numpy."""
        self.flush()
        return _to_numpy(self.sig).reshape(self.grid.shape), _to_numpy(self.norm).reshape(self.grid.shape)

    def result(self):
        """Corrected mean per voxel (NaN where empty), shaped as the grid."""
        sig, norm = self.sums()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(norm > 0, sig / np.where(norm > 0, norm, 1), np.nan)


def _compute(a):
    return a.compute() if hasattr(a, "compute") else a


def _per_shot(n, distance, arm, ex, ey, R):
    R = np.eye(3) if R is None else R
    out = []
    for val, shp in ((distance, ()), (arm, (3,)), (ex, (3,)), (ey, (3,)), (R, (3, 3))):
        a = np.asarray(val, float)
        out.append(np.broadcast_to(a, (n,) + shp) if a.shape == shp else a)
    return out


def _partial(template, block, geo):
    b = template.fresh()
    block = np.asarray(block)
    for i in range(len(block)):
        b.add(block[i], *(g[i] for g in geo))
    sig, norm = b.sums()
    return np.stack([sig.ravel(), norm.ravel()])
