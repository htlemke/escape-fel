# Rebinning Detector Images: 2θ, q and hkl

`escape.bernina.rebin` maps area-detector pixels into another space and sums
them there: 2θ/χ profiles, |Q|, or 3D reciprocal-space maps in sample-frame
q or hkl. It does what pyFAI's integrators and xrayutilities' gridders do,
with the work arranged for many shots: what does not change between shots is
computed once, and what does is computed in one compiled pass per shot.

```{warning}
**New - for testing.** Checked against pyFAI (1D and 2D, no/bbox/full
splitting, solid angle and polarization) and xrayutilities (q conversion and
`Gridder3D`) in `tests/bernina/test_rebin.py`; the GPU (cupy) backend has
not been run on a GPU yet.
```

Install with `pip install "escape-fel[rebin]"` (adds numba; without it the
same code runs on numpy/scipy, much slower). pyFAI, xrayutilities, dask and
cupy are only needed for the functions that use them.

Every result is pyFAI's normalized mean per bin,
`sum(signal) / sum(weight * solid angle * polarization)`. NaN pixels - the
gaps and masked pixels of corrected JUNGFRAU images - are left out of both
sums, frame by frame, and empty bins come out as NaN.

## Fixed geometry: `FixedBinner`

If neither the detector nor the energy moves and the output is a lab
quantity (2θ, χ, |Q|), every pixel always ends up in the same bins. The
binner keeps that sparse matrix and applies it to whole stacks:

```python
from escape.bernina import rebin

# pyFAI's own matrix (pixel splitting and corrections as in pyFAI) ...
fb = rebin.FixedBinner.from_pyfai(ai, 1000, shape, unit="2th_deg", split="bbox")
# ... or your own coordinates per pixel on a Grid
fb = rebin.FixedBinner.from_coords(two_theta.ravel(), rebin.Grid.from_bins(5, 60, 1000), shape, corrections)

profile = fb.mean(image)          # one frame
profiles = fb.mean(stack)         # (nframes, ...) as it comes from the detector
lazy = fb.apply(images)           # escape.Array of images -> escape.Array of profiles
```

`from_pose` does the same for one fixed diffractometer pose into a 3D q or
hkl grid, for example a pump-probe delay scan at fixed angles.

### pyFAI calibrations (PONI) and the diffractometer arm

A pyFAI `.poni` file and the arm frame used here (`distance`, `arm`, `ex`,
`ey`) describe the same detector. The conversion goes both ways and does not
need pyFAI:

```python
g = rebin.poni_geometry("calib.poni")    # or an AzimuthalIntegrator: uv, shape, distance, arm, ex, ey, k
c = rebin.pixel_coordinates(g["uv"], g["distance"], g["arm"], g["ex"], g["ey"], g["k"])   # tth, chi, q
fb = rebin.FixedBinner.from_coords(c["q"], rebin.Grid.from_voxel(0.5, 5.0, 0.005), g["shape"],
                                   rebin.pixel_corrections(g["uv"], g["distance"], g["arm"], g["ex"], g["ey"]))

poni = rebin.frame_to_poni(distance, arm, ex, ey, foot=(height / 2, width / 2))   # the arm at a pose -> PONI
```

pyFAI's x, y, z are SwissFEL's: z along the beam, y up, x left. Detector
`orientation` 1, 2 and 4 are supported as well. pyFAI counts a flipped axis
from its far edge and mirrors the lab x and/or y axis, so those frames come
out mirrored. Pixel positions, 2θ, χ and q agree with pyFAI's
`position_array`/`center_array` to 1e-6 for tilted detectors in every
orientation (`tests/bernina/test_rebin.py`).

## Changing geometry: `StreamBinner`

When the angles change from shot to shot (rocking scans, RSMs, drifting
motors) each shot needs its own conversion. The pixels' positions on the
detector face do not change, so a single kernel computes pixel → q → voxel →
sum per shot without building any large intermediate arrays:

```python
uv, shape = rebin.jungfrau_pixels("JF01T03")          # pixel positions, image order
grid = rebin.scan_grid(uv, k, poses, voxel=2e-3)      # voxel size, extent of the scan
sb = rebin.StreamBinner(uv, k, grid, pixel=0.075)

for image, angles in shots:                           # online, one shot at a time
    sb.add_angles(image, angles, distance, sample, space="hkl")
rsm = sb.result()                                     # mean per voxel; axes: grid.centres
```

The grid follows xrayutilities' gridders: `Grid.from_voxel` (voxel size),
`Grid.from_bins`, `Grid.from_centres` (xrayutilities' first/last centre
convention), `Grid.covering` and `scan_grid` (extent from the data, edges on
multiples of the voxel size so grids from different runs add up).

### Slowly changing angles

Angles that drift or scan slowly change each pixel's voxel only now and
then. Two options use this:

- `tolerance=0.1`: reuse the voxel indices until the geometry has moved by
  0.1 voxel (checked at the detector's corners and centre).
- `group=True`: within that tolerance, also add up the images and bin the
  sum once (binning is linear). The group uses its first shot's geometry.

On a 4-core laptop with a 1.5M JUNGFRAU and a 100³ grid, an exact
conversion takes about 7-10 ms per shot. With grouping it takes about 1 ms
per shot, about the cost of adding the image to a buffer. The error stays
below 1% of the peak for a 0.1 voxel tolerance.
`tests/bernina/bench_rebin.py` measures this on your own machine.

### Many shots: dask and escape

```python
sb.reduce(stack, distance, arm, ex, ey, R)                  # numpy or dask stack, per-shot geometry
sb.reduce_escape(images, {"eta": eta, "delta": 30.0, "mu": 0}, distance, sample)   # matched by pulse ID
```

With a dask stack, each chunk is binned by its own binner and the partial
grids are summed in a tree, so the reduction also runs on a dask cluster.

## Options

| Option | Meaning |
|---|---|
| `solid_angle=True` | (d/r)³, 1 on the detector arm (as pyFAI) |
| `polarization=1.0` | degree of horizontal polarization (pyFAI's `polarization_factor`); `None` turns it off |
| `weight` | per-pixel weight; 0 masks a pixel |
| `split=2, pixel=0.075` | 2×2 subpixels per pixel: pixel splitting for voxels smaller than a pixel's footprint |
| `backend` | `"numba"` (default), `"numpy"`, `"cupy"` (GPU) |

## Compared with the DAP radial profile

sf-daq's online pipeline (`sf-daq/dap`, `dap/algos/radprof.py`) integrates
each frame in rings of one pixel around a beam centre given in pixels. A
`FixedBinner` built from the same rings reproduces its `radint_I` exactly.
`tests/bernina/bench_dap_radprof.py --dap <checkout>` runs DAP's own code next
to escape. It shows two things in DAP (checked against the 2026-10 version):

- `min(rad)` / `max(rad)` are the Python built-ins, which loop over every
  pixel in Python: about 140 ms of the 150 ms per 1.5 Mpx frame, and about
  1.5 s per frame for the 16M. `rad.min()` / `rad.max()` remove that cost.
- With `apply_threshold`, thresholded pixels are set to NaN before
  `np.bincount`. Every ring that holds one such pixel becomes NaN; with
  `threshold_min` below the photon energy that is the whole profile.
