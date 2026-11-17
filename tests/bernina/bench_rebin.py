"""Benchmark: rebinning area-detector frames, escape.bernina.rebin vs other
packages and plain numpy/scipy. Not a test (no assertions); run as a script:

    python tests/bernina/bench_rebin.py [--frames 32] [--shots 200] [--parts ABC] [--json out.json]

A  fixed geometry (2theta 1D, 2theta-chi 2D) on a stack of frames, with
   solid angle and polarization: pyFAI per frame vs FixedBinner (pyFAI's own
   matrix), numpy, the array backend (= the GPU code path, on CPU) and dask
B  geometry changing every shot into a 3D q grid: xrayutilities
   (Ang2Q.area + Gridder3D / FuzzyGridder3D), numpy + silx, StreamBinner
   (exact per shot; corrections, subpixel splitting, array backend)
C  a stream of shots with slowly changing angles: exact per shot vs reusing
   voxel indices within a tolerance vs grouping shots (sum, then bin once);
   accuracy against exact on smooth synthetic frames; dask vs serial

Detector: Bernina's JF01T03 (1.5M) as an assembled 1614 x 1030 image.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings

import numpy as np

from escape.bernina import rebin

warnings.filterwarnings("ignore")
RES = {}
NY, NX, PIX = 1614, 1030, 0.075
EN = 9000.0
K = 2 * np.pi / (12398.42 / EN)


def timeit(fn, repeat=3, number=1):
    fn()                                          # warm-up (jit, caches)
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(number):
            fn()
        ts.append((time.perf_counter() - t) / number)
    return min(ts)


def report(part, name, seconds, frames, note=""):
    ms = 1e3 * seconds / frames
    RES.setdefault(part, []).append(dict(method=name, ms_per_frame=ms, frames_per_s=1e3 / ms, note=note))
    print(f"  {name:<56s} {ms:9.2f} ms/frame {1e3 / ms:8.1f} fr/s  {note}", flush=True)


def rot(axis, deg):
    a = np.radians(deg) * (-1 if axis[1] == "-" else 1)
    c, s = np.cos(a), np.sin(a)
    i = "xyz".index(axis[0])
    j, k = (i + 1) % 3, (i + 2) % 3
    R = np.eye(3)
    R[j, j], R[j, k], R[k, j], R[k, k] = c, -s, s, c
    return R


def det_frame(delta):
    R = rot("x-", delta)
    return R[:, 2], R[:, 0], R[:, 1]


def frames_with_gaps(n, npix, rng):
    f = rng.poisson(5, (n, npix)).astype(np.float32)
    f[:, ::50] = np.nan                           # ~2% NaN pixels, as gaps/masks
    return f


# ====================================================================== A
def part_a(nframes):
    import dask.array as da
    import pyFAI
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    print(f"\nA  fixed geometry, {NY}x{NX} = {NY * NX / 1e6:.2f} Mpx, stack of {nframes}, "
          "solid angle + polarization, 2% NaN pixels")
    shape = (NY, NX)
    det = pyFAI.detectors.Detector(PIX * 1e-3, PIX * 1e-3, max_shape=shape)
    ai = AzimuthalIntegrator(dist=0.12, poni1=0.03, poni2=0.04, detector=det, wavelength=12398.42 / EN * 1e-10)
    rng = np.random.default_rng(0)
    stack = frames_with_gaps(nframes, NY * NX, rng).reshape((nframes,) + shape)
    corr = ai.solidAngleArray(shape) * ai.polarization(shape, factor=1.0)
    tth = ai.center_array(shape, unit="2th_deg").ravel()
    chi = ai.center_array(shape, unit="chi_deg").ravel()

    for label, npt, grid, coords in (
        ("1D 2theta, 1000 bins", 1000, rebin.Grid.from_bins(tth.min(), tth.max() * (1 + 1e-9), 1000), tth),
        ("2D 2theta x chi, 500 x 360", (500, 360),
         rebin.Grid.from_bins((tth.min(), -180), (tth.max() * (1 + 1e-9), 180), (500, 360)),
         np.stack([tth, chi], -1)),
    ):
        part = "A " + label
        print(f" {label}")
        is2d = isinstance(npt, tuple)
        kw = dict(unit="2th_deg", correctSolidAngle=True, polarization_factor=1.0)

        def pyfai(m):
            if is2d:
                return lambda: [ai.integrate2d_ng(d, npt[0], npt[1], method=m, **kw) for d in stack]
            return lambda: [ai.integrate1d_ng(d, npt, method=m, **kw) for d in stack]
        for split in ("no", "bbox"):
            report(part, f"pyFAI {split}/csr/cython (per frame)", timeit(pyfai((split, "csr", "cython")), repeat=2),
                   nframes)
        flat = stack.reshape(nframes, -1)
        idx = grid.index(coords)
        use = idx >= 0
        cflat = corr.ravel()

        def np_bincount():
            for f in flat:
                ok = use & np.isfinite(f)
                np.bincount(idx[ok], f[ok], grid.size) / np.bincount(idx[ok], cflat[ok], grid.size)
        report(part, "numpy bincount (per frame, NaN + corrections)", timeit(np_bincount, repeat=2), nframes)

        fbs = {"no": rebin.FixedBinner.from_coords(coords, grid, shape, corr),
               "bbox": rebin.FixedBinner.from_pyfai(ai, npt, shape, split="bbox")}
        for split, fb in fbs.items():
            report(part, f"escape FixedBinner.mean, {split} split, single frames",
                   timeit(lambda: [fb.mean(f) for f in stack]), nframes)
            report(part, f"escape FixedBinner.mean, {split} split, whole stack", timeit(lambda: fb.mean(stack)),
                   nframes)
        fbn = rebin.FixedBinner.from_pyfai(ai, npt, shape, split="bbox", backend="numpy")
        report(part, "escape array backend (GPU code path) on CPU, bbox, stack", timeit(lambda: fbn.mean(stack)),
               nframes, "scipy.sparse")
        dstack = da.from_array(stack, chunks=(8,) + shape)
        report(part, "escape FixedBinner.mean_dask, bbox, chunks of 8",
               timeit(lambda: fbs["bbox"].mean_dask(dstack).compute(scheduler="threads"), repeat=2), nframes)
        r = (ai.integrate2d_ng(stack[0], npt[0], npt[1], method=("bbox", "csr", "cython"), **kw).intensity.T
             if is2d else ai.integrate1d_ng(stack[0], npt, method=("bbox", "csr", "cython"), **kw).intensity)
        ours = fbs["bbox"].mean(stack[0])
        ok = np.isfinite(ours) & (r != 0)
        print(f"   max rel. diff to pyFAI bbox/csr: {np.max(np.abs(ours[ok] - r[ok]) / np.abs(r[ok])):.1e}")


# ====================================================================== B
def setup_3d(n=(100, 100, 100), eta=(10.0, 12.0), delta=30.0, dist=500.0):
    uv = rebin.detector_pixels(NX * PIX, NY * PIX, PIX)
    arm, ex, ey = det_frame(delta)
    poses = [(dist, arm, ex, ey, rot("y+", e).T) for e in eta]
    g = rebin.scan_grid(uv, K, poses, voxel=1e-3)              # extent of the scan
    grid = rebin.Grid.from_bins(g.lo, g.hi, n)
    return uv, dist, (arm, ex, ey), grid, delta


def part_b(nshots):
    import xrayutilities as xu
    from silx.math import Histogramnd
    uv, dist, (arm, ex, ey), grid, delta = setup_3d()
    lo, hi, n = grid.lo, grid.hi, grid.shape
    print(f"\nB  geometry changing per shot, 3D grid {n}, {len(uv) / 1e6:.2f} Mpx, {nshots} shots")
    part = "B per-shot conversion, 3D q grid 100^3"
    rng = np.random.default_rng(0)
    frames = rng.poisson(5, (8, len(uv))).astype(np.float32)
    etas = np.linspace(10, 12, nshots)

    qc = xu.experiment.QConversion(["y+"], ["x-"], [0, 0, 1], en=EN)
    qc.init_area("y+", "x+", cch1=NY / 2 - 0.5, cch2=NX / 2 - 0.5, Nch1=NY, Nch2=NX,
                 distance=dist, pwidth1=PIX, pwidth2=PIX)
    c = grid.centres
    for name, G, extra in (("Gridder3D", xu.Gridder3D, {}),
                           ("FuzzyGridder3D", xu.FuzzyGridder3D, dict(width=tuple(grid.step)))):
        g = G(*n)
        g.KeepData(True)
        g.dataRange(c[0][0], c[0][-1], c[1][0], c[1][-1], c[2][0], c[2][-1], fixed=True)

        def xu_run():
            g.Clear()
            for i, e in enumerate(etas):
                qx, qy, qz = qc.area(e, delta)
                g(qx, qy, qz, frames[i % 8].reshape(NY, NX), **extra)
        report(part, f"xrayutilities Ang2Q.area + {name}", timeit(xu_run, repeat=1), nshots)
    report(part, "  of which xrayutilities Ang2Q.area alone",
           timeit(lambda: [qc.area(e, delta) for e in etas], repeat=1), nshots)

    P = dist * arm + uv[:, :1] * ex + uv[:, 1:] * ey
    q_lab = K * (P / np.linalg.norm(P, axis=1, keepdims=True) - np.array([0, 0, 1.0]))

    def silx_run():
        h = None
        for i, e in enumerate(etas):
            smp = q_lab @ rot("y+", e)
            if h is None:
                h = Histogramnd(smp, np.stack([lo, hi], -1), n, weights=frames[i % 8])
            else:
                h.accumulate(smp, weights=frames[i % 8])
    report(part, "numpy rotation (q_lab cached) + silx Histogramnd", timeit(silx_run, repeat=1), nshots)

    for label, kw in (("no corrections", dict(solid_angle=False, polarization=None)),
                      ("solid angle + polarization", {}),
                      ("+ 2x2 subpixel splitting", dict(split=2, pixel=PIX)),
                      ("array backend (GPU code path) on CPU", dict(backend="numpy"))):
        def fused():
            sb = rebin.StreamBinner(uv, K, grid, **kw)
            for i, e in enumerate(etas):
                sb.add(frames[i % 8], dist, arm, ex, ey, rot("y+", e).T)
        report(part, f"escape StreamBinner exact, {label}", timeit(fused, repeat=1 if "backend" in kw else 2),
               nshots)
    sb = rebin.StreamBinner(uv, K, grid)
    geo = (dist, arm, ex, ey) + grid.affine(rot("y+", 11).T)
    report(part, "  of which voxel index + corrections alone", timeit(lambda: sb._build_index(geo), number=20), 1)
    f0 = np.ascontiguousarray(frames[0])
    report(part, "  of which accumulate alone", timeit(lambda: sb._bin(f0), number=20), 1)


# ====================================================================== C
STRATEGIES = (("exact per shot", {}),
              ("reuse index, tol 0.1 voxel", dict(tolerance=0.1)),
              ("group shots, tol 0.1 voxel", dict(tolerance=0.1, group=True)),
              ("group shots, tol 0.5 voxel", dict(tolerance=0.5, group=True)))


def part_c(nshots):
    import dask.array as da
    uv, dist, (arm, ex, ey), grid, delta = setup_3d()
    print(f"\nC  stream, {nshots} shots, slowly changing eta, 3D grid {grid.shape}, corrections on, 2% NaN pixels")
    rng = np.random.default_rng(1)
    noise = frames_with_gaps(8, len(uv), rng)
    probe = rebin.StreamBinner(uv, K, grid)
    g0 = probe._probe((dist, arm, ex, ey) + grid.affine(rot("y+", 11).T))
    g1 = probe._probe((dist, arm, ex, ey) + grid.affine(rot("y+", 11.01).T))
    per_deg = np.abs(g1 - g0).max() / 0.01
    print(f"   eta moves the fastest pixel by one voxel in {1 / per_deg:.4f} deg")
    P = dist * arm + uv[:, :1] * ex + uv[:, 1:] * ey
    q_lab = K * (P / np.linalg.norm(P, axis=1, keepdims=True) - np.array([0, 0, 1.0]))
    q0 = q_lab[len(q_lab) // 2 + NX // 2] @ rot("y+", 11)
    sig = 3 * np.min(grid.step)
    corr = rebin.pixel_corrections(uv, dist, arm, ex, ey)

    def peak(eta):        # a smooth intensity in the sample frame, as measured (times corrections)
        d = q_lab @ rot("y+", eta) - q0
        return ((1 + 100 * np.exp(-np.sum(d * d, 1) / (2 * sig ** 2))) * corr).astype(np.float32)

    scenarios = {
        "fixed + encoder jitter 1e-4 deg": lambda i: 11 + rng.normal(0, 1e-4),
        "drift 1e-4 deg/shot": lambda i: 11 + 1e-4 * i,
        "scan 2e-3 deg/shot": lambda i: 10.8 + 2e-3 * i,
    }
    for sname, eta_of in scenarios.items():
        part = "C stream: " + sname
        etas = np.array([eta_of(i) for i in range(nshots)])
        print(f" {sname}  (eta range {np.ptp(etas):.4f} deg = {np.ptp(etas) * per_deg:.2f} voxels)")
        acc = [rebin.StreamBinner(uv, K, grid, **kw) for _, kw in STRATEGIES]
        for e in etas:
            f = peak(e)
            for sb in acc:
                sb.add(f, dist, arm, ex, ey, rot("y+", e).T)
        exact = acc[0].result()
        for (label, kw), sb_acc in zip(STRATEGIES, acc):
            res = sb_acc.result()
            ok = np.isfinite(exact) & np.isfinite(res)
            err = float(np.max(np.abs(res[ok] - exact[ok])) / 100)

            def run():
                sb = rebin.StreamBinner(uv, K, grid, **kw)
                for i, e in enumerate(etas):
                    sb.add(noise[i % 8], dist, arm, ex, ey, rot("y+", e).T)
                sb.flush()
            st = sb_acc.stats
            report(part, label, timeit(run, repeat=1), nshots,
                   f"index builds {st['index_builds']}, bin passes {st['bin_passes']}, max err {err:.1e} of peak")

    part = "C dask"
    print(" dask vs serial, scan 2e-3 deg/shot")
    etas = 10.8 + 2e-3 * np.arange(nshots)
    R = np.array([rot("y+", e).T for e in etas])
    stack = np.stack([noise[i % 8] for i in range(nshots)])
    dstack = da.from_array(stack, chunks=(25, -1))
    for label, kw in (("exact", {}), ("group, tol 0.1 voxel", dict(tolerance=0.1, group=True))):
        report(part, f"serial reduce, {label}",
               timeit(lambda: rebin.StreamBinner(uv, K, grid, **kw).reduce(stack, dist, arm, ex, ey, R), repeat=1),
               nshots)
        report(part, f"dask reduce (chunks of 25, threads), {label}",
               timeit(lambda: rebin.StreamBinner(uv, K, grid, **kw).reduce(dstack, dist, arm, ex, ey, R,
                                                                          scheduler="threads"), repeat=1), nshots)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--parts", default="ABC")
    ap.add_argument("--json")
    a = ap.parse_args()
    import numba
    print(platform.processor() or platform.machine(), "numba threads", numba.get_num_threads())
    if "A" in a.parts:
        part_a(a.frames)
    if "B" in a.parts:
        part_b(min(a.shots, 50))
    if "C" in a.parts:
        part_c(a.shots)
    if a.json:
        json.dump(RES, open(a.json, "w"), indent=1)
