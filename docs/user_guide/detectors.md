# Detector Images: Droplets and Gain Switching

`escape.utilities_detectors` holds processing steps for pixel-detector image
stacks (e.g. Jungfrau). The droplet tools are built on
{meth}`~escape.Array.map_index_blocks` and run lazily, one chunk of frames at
a time, so stacks larger than memory are fine.

```{warning}
**Experimental — for testing.** These tools are new and have not yet been
validated against a broad range of real detector data. Check the results
(e.g. against a simple threshold analysis) before relying on them.
```

## Droplets — photon counting

A single photon's charge often spreads over a few neighbouring pixels.
`find_droplets` thresholds each frame, joins neighbouring above-threshold
pixels into "droplets" (4-connected within a frame, never across frames),
sums each droplet's intensity and then, depending on `mode`:

| `mode` | Output per event | Use |
|---|---|---|
| `"image"` | same-shape frame, each droplet's intensity on its centroid pixel | drop-in for the raw frames, cleaner intensity histograms |
| `"sparse"` | `(max_droplets, 3)` rows of `(row, col, intensity)` | big data reduction for low-occupancy frames |
| `"photon_count"` | histogram of droplets with 1, 2, 3, … photons (last bin = overflow) | pile-up statistics |

```python
from escape.utilities_detectors import find_droplets

clean = find_droplets(images, threshold=15.0)                 # lazy escape.Array
clean.scan.nanmean()                                          # use like the raw frames

events = find_droplets(images, threshold=15.0, mode="sparse",
                       max_droplets=200, chunk_size=50)
```

`chunk_size` sets how many frames are processed per task, the main knob for
memory and parallelism.

### Photon statistics

With `mode="photon_count"` (needs `photon_value`, the ADU of one photon), the
summed histogram can be fitted with a zero-truncated Poisson distribution;
zero-photon droplets cannot exist, so a plain Poisson fit would be biased:

```python
from escape.utilities_detectors import fit_photon_number_distribution

hist = find_droplets(images, threshold=15.0, mode="photon_count",
                     photon_value=105.0, max_photons=8)
totals = hist.compute().data.sum(axis=0)       # sum over all events
fit = fit_photon_number_distribution(totals)
fit.k, fit.reduced_chi2                        # photon rate, fit quality (≈1 = Poissonian)
```

If the last (overflow) bin is not small, raise `max_photons` before trusting
the fit.

To try it without beamline data, generate a synthetic stack with
{func}`escape.storage.example_data.make_detector_photon_stack`.

## Dynamic gain-switching pedestal offsets

`escape.utilities_detectors.dynamic_pedestal` estimates, per pixel, the
pedestal offset a gain-switching detector settles to right after a
*dynamic* gain switch, which can differ from the pedestal measured in forced
gain. Such an offset shows up as a step in the corrected intensity at the
switching threshold. The offsets are found by requiring the corrected
intensity to be continuous across each gain transition, compared against an
independent intensity monitor.

```python
from escape.utilities_detectors import (
    decode_jungfrau_raw, find_dynamic_pedestal_offsets, apply_calibration,
)

# NumPy arrays: frames = raw 16-bit Jungfrau words (keep_raw_data), shape (n_events, *pixels)
raw, gain_idx = decode_jungfrau_raw(frames)
offsets, counts = find_dynamic_pedestal_offsets(monitor, raw, gain_idx, pedestal, gain)
corrected = apply_calibration(raw, gain_idx, pedestal, gain, offset=offsets)
```

```{warning}
**Beta**, not yet validated against real detector data: treat the recovered
offsets as a hypothesis to check, not a drop-in calibration. Process an ROI
or tile rather than a full multi-megapixel detector in one call.
```

See {doc}`../api/detectors` for all functions.
