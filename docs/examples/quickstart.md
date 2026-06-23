# Quick Start

This page walks through the most common operations using synthetic data that is
included with the package.

## Creating Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt
import escape
from escape.storage.example_data import make_scan, make_pump_probe_scan

# Ten-step delay scan, 500 events per step
sig = make_scan(
    n_steps=10,
    n_events_per_step=500,
    scan_par_name="delay_ps",
    scan_par_values=np.linspace(-1, 4, 10),
    name="bragg_intensity",
    seed=0,
)

print(sig)
print(sig.scan)
```

## Per-Step Statistics and Plotting

```python
# Per-step statistics
means  = sig.scan.nanmean()
stds   = sig.scan.nanstd()
counts = sig.scan.count()

# Built-in scan plot (median ± σ/√N)
fig, ax = plt.subplots()
sig.scan.plot(axis=ax, fmt="o-")
ax.set_xlabel("delay / ps")
ax.set_ylabel("Bragg intensity (a.u.)")
plt.tight_layout()
plt.show()
```

## Boolean Filtering

```python
# Keep only events with signal above a threshold
strong = sig.filter(0.7, 1.5)   # keep 0.7 ≤ sig ≤ 1.5

print(f"Before filter: {len(sig)} events")
print(f"After  filter: {len(strong)} events")

# Re-plot on the filtered data
fig, ax = plt.subplots()
strong.scan.plot(axis=ax, fmt="s-", label="filtered")
sig.scan.plot(axis=ax, fmt="o--", label="all")
ax.legend()
plt.show()
```

## Normalisation with Index Alignment

```python
sig, i0, pump_on, delay = make_pump_probe_scan(n_steps=10, seed=1)

# Division auto-aligns on common pulse IDs
sig_norm = sig / i0

# Separate pump-on and pump-off shots
sig_on  = sig_norm[~pump_on]   # laser ON
sig_off = sig_norm[pump_on]    # laser OFF (reference)

# Per-step pump/probe ratio
ratio = sig_on.scan / sig_off.scan.nanmean(axis=0)

fig, ax = plt.subplots()
ratio.scan.plot(axis=ax, fmt="o-")
ax.axhline(1.0, ls="--", color="k")
ax.set_xlabel("delay / ps")
ax.set_ylabel("relative signal")
plt.tight_layout()
plt.show()
```

## Applying a Custom Function with `map_index_blocks`

```python
from escape.storage.example_data import make_image_scan

imgs = make_image_scan(n_steps=3, n_events_per_step=50, seed=2)

# Sum a region of interest per event
roi = imgs[:, 25:45, 25:45]
roi_sum = roi.nansum(axis=(1, 2))

print(roi_sum.shape)   # (150,)

# Or apply a custom function to each chunk:
def hot_pixel_mask(block, threshold=200):
    out = block.copy()
    out[out > threshold] = np.nan
    return out

imgs_clean = imgs.map_index_blocks(hot_pixel_mask, 200)
mean_img = imgs_clean.scan[0].mean(axis=0).compute()

fig, ax = plt.subplots()
ax.imshow(mean_img, cmap="viridis")
ax.set_title("Mean image (step 0, hot-pixel masked)")
plt.tight_layout()
plt.show()
```

## Storing Results to HDF5

```python
import h5py

with h5py.File("quickstart_results.h5", "w") as fh:
    sig.store(fh, "signal")
    i0.store(fh,  "i0")
    ratio.store(fh, "ratio")

# Reload
with h5py.File("quickstart_results.h5", "r") as fh:
    ratio_loaded = escape.Array.load_from_h5(fh, "ratio")

print(ratio_loaded.scan)
```
