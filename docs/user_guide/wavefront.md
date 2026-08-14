# X-ray Wavefront Sensing

`escape.wavefront` is a compact, teaching-oriented implementation of
**single-grating (Talbot) X-ray wavefront sensing** for FEL beams, together
with the free-space propagation needed to **back-propagate a measured beam to
any plane** — the focus, a sample, or an optic.

The method follows Liu, Seaberg *et al.* \[[1](#references), [2](#references)\];
the implementation is distilled and streamlined from M. Seaberg's reference code
([`lcls_beamline_toolbox`](https://github.com/mseaberg/lcls_beamline_toolbox)
and [`wfs_interface`](https://github.com/mseaberg/wfs_interface)), rewritten
around `escape` / dask conventions and plain NumPy so the whole pipeline can be
read end to end. See the [References](#references) below for the primary papers
and the underlying algorithms.

```{note}
This submodule is optional and imported explicitly:

    from escape import wavefront as wf

It depends on `scipy` (integration) and, for the full dynamic range, on
`scikit-image` (2-D phase unwrap — a graceful no-op if absent).
```

---

## The physics, in one page

A **2-D grating** (a diamond π-phase checkerboard in real instruments) is placed
in the beam.  A short distance `zT` downstream — a *fractional Talbot distance* —
the grating casts a high-contrast intensity **self-image** (a grid of spots) on
a detector of pixel size `dx`.

The self-image is a sensitive ruler for the wavefront:

* a local wavefront **slope** deflects the rays and **shifts** the local fringe,
* a local wavefront **curvature** magnifies or shrinks the fringe **period**.

Recovering those distortions is a **Fourier-fringe (Takeda) demodulation**:

```{mermaid}
flowchart LR
    A[Talbot image] --> B[FFT]
    B --> C["band-pass the +x and +y<br/>first-order peaks"]
    C --> D["cross-phase vs a<br/>reference image"]
    D --> E["phase gradients<br/>d&phi;/dx, d&phi;/dy"]
    E --> F["weighted<br/>least-squares<br/>integration"]
    F --> G["wavefront &phi;(x,y)"]
    G --> H["curvature &rarr; focus distance<br/>+ complex field"]
    H --> I["back-propagate<br/>to any plane"]
```

Two facts do most of the work:

1. **The shear relation.** After band-passing one first order and removing its
   carrier, the demodulated phase (measured *against a reference exposure*) is

   ```
   res = f_carrier · λ · zT · dφ/dx
   ```

   so the beam phase gradient is `dφ/dx = res / (f_carrier · λ · zT)`.  The
   division by the measured carrier frequency `f_carrier` is why a reference
   image matters: it fixes the carrier and keeps the absolute wavefront —
   defocus included — from being subtracted away.

2. **Curvature → focus.** A spherical wavefront has phase `φ = π r² / (λ R)`.
   Fitting the measured gradients to `dφ/dx = 2π x / (λ R)` gives the radius of
   curvature `R`; the focus sits `−R` away from the detector.

---

## Quick start: simulate → reconstruct → back-propagate

```python
import numpy as np
from escape import wavefront as wf
from escape.wavefront import talbot as t

# --- geometry -----------------------------------------------------------
shape  = (512, 512)
dx     = 0.6e-6                         # detector pixel (m)
period = 6e-6                           # grating period (m)
lam    = wf.wavelength_from_energy(9500.0)          # 9.5 keV -> m
zT     = wf.talbot_distance(period, lam, fraction=0.5)   # half-Talbot plane

# --- a reference (flat wavefront) exposure ------------------------------
x, y = t._coords(shape, dx)
beam = np.exp(-(x**2 + y**2) / (2 * (80e-6)**2))    # Gaussian illumination
reference = wf.simulate_talbot_image(shape, dx, period, lam, zT,
                                     incident_amplitude=beam)

# --- a measurement: a beam diverging as if from a source 8 m upstream ---
phi_in = wf.parabolic_phase(shape, dx, radius=8.0, lambda0=lam)
image  = wf.simulate_talbot_image(shape, dx, period, lam, zT,
                                  incident_phase=phi_in,
                                  incident_amplitude=beam)

# --- reconstruct --------------------------------------------------------
w = wf.reconstruct_wavefront(image, reference, dx=dx,
                             lambda0=lam, zT=zT, period=period)

print(w.radius_of_curvature())   # ~ 8.28 m  (= 8 m source + zT)
print(w.distance_to_focus())     # signed distance to the focus
print(w.rms(remove_sphere=True)) # residual aberration (rad)
```

`w` is a {class}`~escape.wavefront.Wavefront`: it carries the reconstructed
`phase`, the beam `amplitude`, the coordinate meshes, and the measured
`grad_x`/`grad_y`.  Its `field` property is the complex field
`amplitude · exp(i·phase)` at the detector.

### Gratings for teaching vs. reality

Real hard-X-ray sensors use an etched **π-phase checkerboard** for efficiency,
but its fractional-Talbot revivals are highly non-linear and only reconstruct
cleanly with careful per-plane calibration.  For learning and for verifying a
pipeline, prefer the smooth **mesh** grating (`grating="mesh"`, the default in
`simulate_talbot_image`) — its self-image is a faithful, magnifying replica and
reconstructs to ~0.1 % on defocus out of the box.  Pass `grating="checkerboard"`
to experiment with the realistic case.

---

## Back-propagation over a whole run (escape + dask)

The reconstructed field can be propagated to any plane with the
angular-spectrum method.  {func}`~escape.wavefront.propagate_array` does this
**lazily for a full stack of per-pulse fields**, plugging directly into
`escape`'s {meth}`~escape.Array.map_index_blocks`:

```python
import escape, dask.array as da

# a stack of reconstructed complex fields, one per FEL pulse
fields = escape.Array(
    data=da.stack([...]).rechunk((chunk, *shape)),
    index=pulse_ids,
)

# walk each field upstream to the focus (negative z = upstream)
focus = wf.propagate_array(fields, dx=dx, z=w.distance_to_focus(),
                           lambda0=lam)

# still lazy; compute intensity per pulse on demand
intensity = focus.map_index_blocks(lambda b: np.abs(b) ** 2, dtype="f4")
mean_focus = intensity.mean(axis=0).compute()
```

Positive `z` propagates downstream (toward the detector / forward), negative `z`
upstream (toward the focus / backward).  Because the transform is FFT-based over
the last two axes, an entire dask chunk of events is propagated in one
vectorised call — no per-event Python loop.

---

## Why the propagator is pure NumPy/dask (numba & memory leaks)

The propagation is entirely FFT-bound, and `numpy.fft` / `dask.array.fft`
already dispatch to a fast, **memory-clean** C library.  Adding a `numba` kernel
would buy nothing and would risk exactly the kind of leak seen when the
`jungfrau_utils` numba gain/pedestal corrections are called repeatedly from
inside dask workers.

If you *do* need a genuinely element-wise numba kernel (not an FFT) inside a
dask graph, these rules keep it from leaking:

1. **Compile once, at module level** — never decorate a closure inside the
   mapped function (that recompiles and caches a new artifact every task).
2. **Keep the kernel single-threaded** — no `@njit(parallel=True)`, `prange`, or
   `@guvectorize(target='parallel')` inside a dask graph.  Numba's threading
   layer spins up a pool per process that is never released between tasks and
   fights the scheduler.  Let *dask* parallelise over chunks; use
   `@njit(cache=True, fastmath=True)` with a plain `range`.  If unavoidable, pin
   `NUMBA_NUM_THREADS=1` / `numba.config.THREADING_LAYER = 'workqueue'`.
3. **Feed a contiguous copy and write into a pre-allocated `out`** —
   `np.ascontiguousarray(block)`; returning views of numba-managed buffers can
   keep chunks alive.
4. **Wrap the compiled call in a thin Python function** and hand *that* to
   `map_index_blocks`, so the artifact is shared, not rebuilt.

A full worked sketch is in the source of
`escape.wavefront.propagation` (see the notes block at the end of the file).

---

## Accuracy and dynamic range

Validated on simulated data (mesh grating, half-Talbot plane):

| Quantity                         | Result                                   |
|----------------------------------|------------------------------------------|
| Defocus / focus distance         | ~0.1 % over `R` from ~20 m down to ~1.5 m |
| Astigmatism / coma / trefoil     | shape correlation > 0.997                |
| Focal-spot recovery (back-prop)  | 37 µm beam → 4.7 µm focus, 470× gain     |

The unwrapped **dynamic range** is set by the fringe carrier: the cross-phase
wraps once the fringe shift exceeds one period.  2-D phase unwrapping (via
`scikit-image`, enabled by default) extends this by many wraps; without it the
usable focus distance is capped at tens of metres.  Beyond the range, focus
estimates degrade gracefully into noise — a real sensor limitation, not a bug.

---

## API

See {doc}`../api/wavefront` for the full reference.  The essentials:

| Function | Purpose |
|----------|---------|
| {func}`~escape.wavefront.wavelength_from_energy` | eV → m |
| {func}`~escape.wavefront.talbot_distance` | fractional Talbot distance |
| {func}`~escape.wavefront.mesh_grating` / {func}`~escape.wavefront.checkerboard_grating` | grating transmission |
| {func}`~escape.wavefront.simulate_talbot_image` | forward simulation |
| {func}`~escape.wavefront.fourier_fringe_gradients` | demodulate → gradients |
| {func}`~escape.wavefront.integrate_gradients` | gradients → wavefront |
| {func}`~escape.wavefront.reconstruct_wavefront` | full pipeline → {class}`~escape.wavefront.Wavefront` |
| {func}`~escape.wavefront.angular_spectrum_propagate` | propagate one field |
| {func}`~escape.wavefront.propagate_array` | propagate a lazy escape Array of fields |

---

## References

**Primary method (X-ray FEL Talbot wavefront sensing)**

1. Y. Liu, M. Seaberg, Y. Feng, K. Li, Y. Ding, G. Marcus, D. Fritz, X. Shi,
   W. Grizolli, L. Assoufid, P. Walter & A. Sakdinawat,
   "X-ray free-electron laser wavefront sensing using the fractional Talbot
   effect", *Journal of Synchrotron Radiation* **27**(2), 254–261 (2020).
   doi:[10.1107/S1600577519017107](https://doi.org/10.1107/S1600577519017107)

2. Y. Liu, M. Seaberg, D. Zhu, J. Krzywinski, F. Seiboth, C. Hardin, D. Cocco,
   A. Aquila, B. Nagler, H. J. Lee, S. Boutet & Y. Feng,
   "High-accuracy wavefront sensing for x-ray free electron lasers",
   *Optica* **5**(8), 967–975 (2018).
   doi:[10.1364/OPTICA.5.000967](https://doi.org/10.1364/OPTICA.5.000967)

**Reference software** (the code this module distils, by M. Seaberg, SLAC)

3. `lcls_beamline_toolbox` — <https://github.com/mseaberg/lcls_beamline_toolbox>
   (Talbot reconstruction in `utility/Talbot_functions_beta.py`; wave propagation
   in `xraywavetrace/beam.py`).
4. `wfs_interface` — <https://github.com/mseaberg/wfs_interface>.

**Underlying algorithms**

5. M. Takeda, H. Ina & S. Kobayashi, "Fourier-transform method of fringe-pattern
   analysis for computer-based topography and interferometry", *Journal of the
   Optical Society of America* **72**(1), 156–160 (1982).
   doi:[10.1364/JOSA.72.000156](https://doi.org/10.1364/JOSA.72.000156)
   *(Fourier-fringe demodulation.)*

6. R. T. Frankot & R. Chellappa, "A method for enforcing integrability in shape
   from shading algorithms", *IEEE Transactions on Pattern Analysis and Machine
   Intelligence* **10**(4), 439–451 (1988).
   doi:[10.1109/34.3909](https://doi.org/10.1109/34.3909)
   *(Least-squares gradient integration.)*

7. M. A. Herráez, D. R. Burton, M. J. Lalor & M. A. Gdeisat, "Fast two-dimensional
   phase-unwrapping algorithm based on sorting by reliability following a
   noncontinuous path", *Applied Optics* **41**(35), 7437–7444 (2002).
   doi:[10.1364/AO.41.007437](https://doi.org/10.1364/AO.41.007437)
   *(The 2-D unwrap used via* `scikit-image`*.)*

8. J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Company
   (2005), ch. 3–4. *(Angular-spectrum / Fresnel propagation.)*
