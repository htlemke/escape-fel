"""DAP's radial integration (sf-daq/dap, dap/algos/radprof.py) against
escape.bernina.rebin and a faster pure-numpy version DAP could adopt.

Runs DAP's own code from a checkout, unchanged:

    python tests/bernina/bench_dap_radprof.py --dap ~/mypy/dap [--detector JF07T32]

A DAP worker processes one frame at a time on one pinned core, so the
comparison is per frame and single-threaded (numba threads = 1); the
multithreaded escape number is shown for reference only.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import types
from pathlib import Path

import numpy as np


def load_dap_radprof(dap_root):
    """dap/algos/radprof.py as a module, without dap/algos/__init__ (which
    needs jungfrau_utils, zmq and the compiled peakfinder)."""
    algos = Path(dap_root) / "dap" / "algos"
    pkg = types.ModuleType("dap_algos")
    pkg.__path__ = [str(algos)]
    sys.modules["dap_algos"] = pkg
    return importlib.import_module("dap_algos.radprof")


# ---------------------------------------------------------------- a numpy version DAP could adopt

class RadialProfileNumpy:
    """Same output as dap's calc_radial_integration, numpy only.

    - prepared once per (shape, centre, mask object): no per-frame hashing
      of the 1.6-18 MB mask (dap's npmemo keys on mask.tobytes() every frame)
    - masked pixels go to an extra trash bin instead of a boolean-mask copy
    - NaN pixels (thresholded, or NaN in the image) are left out of the sum
      AND the count of their ring, instead of turning the ring NaN
    """

    def __init__(self, shape, x0, y0, keep_pixels=None):
        y, x = np.indices(shape)
        rad = np.sqrt((x - x0) ** 2 + (y - y0) ** 2).astype(int).ravel()
        self.nbins = int(rad.max()) + 1
        if keep_pixels is not None:
            rad = np.where(np.asarray(keep_pixels).ravel(), rad, self.nbins)    # trash bin
        self.rad = rad.astype(np.intp)
        self.count = np.bincount(self.rad, minlength=self.nbins + 1)[: self.nbins].astype(float)
        self.r_min = int(self.rad.min())

    def __call__(self, image):
        d = image.ravel()
        s = np.bincount(self.rad, d, self.nbins + 1)[: self.nbins]
        nan = np.isnan(d)
        if nan.any():                       # only frames with NaN pay for the NaN handling
            s = np.bincount(self.rad[~nan], d[~nan], self.nbins + 1)[: self.nbins]
            n = self.count - np.bincount(self.rad[nan], minlength=self.nbins + 1)[: self.nbins]
        else:
            n = self.count
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / n


# ---------------------------------------------------------------- data

def jungfrau_like(detector, rng):
    """An assembled image size, a pixel mask (gaps between modules and 0.5%
    bad pixels masked) and photon-like frames (most pixels ~0, a ring)."""
    from escape.bernina.rebin import jungfrau_pixels
    uv, shape = jungfrau_pixels(detector)
    from escape.bernina.detectors import JF_PIXEL, jungfrau_modules
    w, h, modules = jungfrau_modules(detector)
    u, v = uv[:, 0].reshape(shape), uv[:, 1].reshape(shape)
    on = np.zeros(shape, bool)
    for u0, v0, u1, v1 in modules:
        on |= (u >= u0) & (u < u1) & (v >= v0) & (v < v1)
    mask = on & (rng.random(shape) > 0.005)
    return shape, mask


def frames(shape, n, x0, y0, rng):
    y, x = np.indices(shape)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    ring = 2.0 * np.exp(-((r - 0.3 * max(shape)) / 20) ** 2)
    noise = rng.normal(0, 0.3, (n,) + shape)            # keV-like noise around 0
    photons = rng.poisson(ring + 0.02, (n,) + shape) * 9.0
    return (noise + photons).astype(np.float32)


def timeit(fn, n=5):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dap", required=True, help="path of a sf-daq/dap checkout")
    ap.add_argument("--detector", default="JF01T03")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    rp = load_dap_radprof(a.dap)
    import numba
    from escape.bernina import rebin

    rng = np.random.default_rng(0)
    shape, mask = jungfrau_like(a.detector, rng)
    x0, y0 = shape[1] * 0.45, shape[0] * 0.55
    imgs = frames(shape, a.frames, x0, y0, rng)
    print(f"{a.detector}: {shape[0]} x {shape[1]} = {np.prod(shape) / 1e6:.2f} Mpx, "
          f"{mask.mean():.1%} unmasked, {a.frames} frames")

    base = dict(do_radial_integration=1, beam_center_x=x0, beam_center_y=y0)

    def dap(img, cfg):
        res = dict(cfg)
        rp.calc_radial_integration(res, img, mask)
        return np.asarray(res["radint_I"]), res["radint_q"]

    def dap_minmax_fixed(img, cfg):
        """dap's calc_radial_integration with min(rad)/max(rad) (python
        builtins, looping over ~1e6-1e7 numpy elements) as rad.min()/rad.max()."""
        res = dict(cfg)
        rad_, norm_ = rp.prepare_radial_profile(img.shape, x0, y0, mask)
        r_min_, r_max_ = rad_.min(), rad_.max() + 1
        data = rp.calc_apply_threshold(res, img, value=np.nan, copy=True)
        prof = rp.radial_profile(data, rad_, norm_, mask)
        res["radint_I"] = prof[r_min_:]
        res["radint_q"] = [r_min_, r_max_]
        return res["radint_I"], res["radint_q"]

    # --- equal results without threshold
    I_dap, (r_min, r_max) = dap(imgs[0], base)
    n_out = len(I_dap)                 # up to the largest unmasked radius
    ours_np = RadialProfileNumpy(shape, x0, y0, mask)
    I_np = ours_np(imgs[0])[r_min:r_min + n_out]
    rad = np.sqrt(np.add.outer((np.arange(shape[0]) - y0) ** 2, (np.arange(shape[1]) - x0) ** 2)).astype(int)
    fb = rebin.FixedBinner.from_index(np.where(mask, rad, -1).ravel(), int(rad.max()) + 1, shape=shape)
    I_rb = fb.mean(imgs[0])[r_min:r_min + n_out]
    I_fx = dap_minmax_fixed(imgs[0], base)[0]
    for name, I in (("dap, min/max fixed", I_fx), ("numpy version", I_np), ("escape FixedBinner", I_rb)):
        ok = np.isfinite(I_dap)
        print(f"  {name} vs dap, no threshold: max rel. diff {np.max(np.abs(I[ok] - I_dap[ok]) / np.abs(I_dap[ok])):.1e},"
              f" same NaN bins: {np.array_equal(np.isnan(I), np.isnan(I_dap))}")

    # --- threshold: dap turns thresholded pixels into NaN before bincount
    thr = dict(base, apply_threshold=1, threshold_min=2.0)
    I_dap_t, _ = dap(imgs[0], thr)
    img_t = imgs[0].copy()
    img_t[img_t < 2.0] = np.nan
    I_np_t = ours_np(img_t)[r_min:r_min + n_out]
    print(f"  threshold_min=2: dap rings that are NaN: {np.isnan(I_dap_t).mean():.1%}; "
          f"NaN-aware version: {np.isnan(I_np_t).mean():.1%} (rings with no pixel left)")
    thr_max = dict(base, apply_threshold=1, threshold_max=50.0)       # cut only hot pixels
    hot = imgs[0].copy()
    hot.ravel()[rng.choice(hot.size, 20, replace=False)] = 1e4
    I_dap_h, _ = dap(hot, thr_max)
    print(f"  threshold_max=50 with 20 hot pixels: dap rings that are NaN: {int(np.isnan(I_dap_h).sum())}")

    # --- timing, one core
    numba.set_num_threads(1)
    print("  per frame, one core:")
    t_dap = timeit(lambda: [dap(f, base) for f in imgs]) / len(imgs)
    print(f"    dap calc_radial_integration                   {t_dap:7.2f} ms")
    key = timeit(lambda: [rp.prepare_radial_profile(shape, x0, y0, mask) for _ in imgs]) / len(imgs)
    print(f"      of which npmemo lookup (mask.tobytes() key)  {key:7.2f} ms")
    t_fx = timeit(lambda: [dap_minmax_fixed(f, base) for f in imgs]) / len(imgs)
    print(f"    dap with min/max -> rad.min()/rad.max()       {t_fx:7.2f} ms   {t_dap / t_fx:4.1f}x")
    t_dap_t = timeit(lambda: [dap(f, thr) for f in imgs]) / len(imgs)
    print(f"    dap with threshold                            {t_dap_t:7.2f} ms")
    t_np = timeit(lambda: [ours_np(f) for f in imgs]) / len(imgs)
    print(f"    numpy version (no NaN in frame)               {t_np:7.2f} ms   {t_dap / t_np:4.1f}x")
    t_np_t = timeit(lambda: [ours_np(np.where(f < 2.0, np.nan, f)) for f in imgs]) / len(imgs)
    print(f"    numpy version with threshold (NaN-aware)      {t_np_t:7.2f} ms   {t_dap_t / t_np_t:4.1f}x")
    t_rb = timeit(lambda: [fb.mean(f) for f in imgs]) / len(imgs)
    print(f"    escape FixedBinner (numba, 1 thread)          {t_rb:7.2f} ms   {t_dap / t_rb:4.1f}x")
    t_rb_t = timeit(lambda: [fb.mean(np.where(f < 2.0, np.nan, f)) for f in imgs]) / len(imgs)
    print(f"    escape FixedBinner with threshold             {t_rb_t:7.2f} ms   {t_dap_t / t_rb_t:4.1f}x")
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    t_rb8 = timeit(lambda: [fb.mean(f) for f in imgs]) / len(imgs)
    print(f"    escape FixedBinner, {numba.get_num_threads()} threads (reference)       {t_rb8:7.2f} ms")


if __name__ == "__main__":
    main()
