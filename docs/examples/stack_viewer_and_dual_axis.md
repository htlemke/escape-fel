# Interactive Viewers and Dual-Axis Plots

This example covers two of the plotting utilities in `escape.utilities`
(built on `escape.plot_utilities`): {class}`~escape.plot_utilities.StackViewer`,
a slider-driven viewer for 3-D image stacks with live ROI selection, and the
dual/secondary-axis helpers for scans whose parameter isn't perfectly
monotonic (e.g. a "sawtooth" delay scan re-measured forward and backward to
average out drift).

## Interactive Image-Stack Viewer

```python
import matplotlib.pyplot as plt
from escape.storage.example_data import make_image_scan
from escape.utilities import StackViewer

# A scan with a 2-D image (a shifting Bragg peak) per step
imgs = make_image_scan(n_steps=8, n_events_per_step=100, seed=0)

viewer = StackViewer(imgs, rois=True)
viewer  # displays as an ipywidgets VBox in a Jupyter cell
```

In a notebook with the `ipympl` (`%matplotlib widget`) backend enabled, this
displays a slider stepping through the scan: moving it first shows a cheap
preview (mean of a handful of events) and, once the slider settles, replaces
it with the full per-step mean computed in a background thread — so scrubbing
through many steps stays responsive even for large, lazily-loaded (dask-backed)
data. With `rois=True`, an "Add ROI" panel appears next to the image; each
rectangular region gets its own tab with a cropped preview, and a trend plot
tracks that region's mean value across every step you've visited.

## Per-Cell Figure Naming with `nfigure` / `nsubplots`

`StackViewer` (and the other widgets on this page) create their figures via
{func}`~escape.plot_utilities.nfigure`/{func}`~escape.plot_utilities.nsubplots`
rather than plain `plt.figure`/`plt.subplots`. Called with no `num`, they
derive a name from the currently executing notebook cell, so re-running a
cell replaces *that cell's* figure/widget instead of leaving stale ones
around, while different cells naturally get different figures instead of
overwriting each other:

```python
from escape.utilities import nfigure, nsubplots

fig = nfigure()                          # named after this cell
fig, axes = nsubplots(1, 2, figsize=(8, 3))  # same, for a subplot grid
```

Pass `num=` explicitly (as before) if you want to manage the name yourself,
e.g. to intentionally reuse one figure across several cells.

## Dual/Secondary Axis for Non-Monotonic Scans

A scan that is re-measured forward and backward to average out slow drift
produces a delay array that isn't globally monotonic — matplotlib's built-in
`secondary_xaxis` can't label it directly, since it assumes an invertible
(monotonic) relationship. {func}`~escape.plot_utilities.add_step_secondary_axis`
handles this by detecting the scan's monotonic segments and adding a step-index
axis on top, with the reversal point ("turning point") labelled directly on
the curve:

```{eval-rst}
.. plot::

   import matplotlib
   matplotlib.use("Agg")
   import numpy as np
   import matplotlib.pyplot as plt
   from escape.plot_utilities import add_step_secondary_axis

   # Forward scan, then a repeat scan back down -- not globally monotonic
   delay_fwd = np.linspace(-2, 6, 12)
   delay_ps = np.concatenate([delay_fwd, delay_fwd[::-1]])

   rng = np.random.default_rng(0)
   signal = 1.0 - 0.4 * np.exp(-((delay_ps - 2.0) ** 2) / (2 * 1.5 ** 2))
   signal += 0.02 * rng.standard_normal(signal.shape)

   fig, ax = plt.subplots(figsize=(7, 4))
   ax.plot(delay_ps, signal, "o-", color="0.15", markersize=4)
   ax.set_xlabel("delay / ps")
   ax.set_ylabel("signal (a.u.)")

   add_step_secondary_axis(ax, delay_ps, signal)
   plt.tight_layout()
```

For the fully general case — an arbitrary secondary array that isn't simply
the step index (e.g. a timetool-measured actual delay paired with the
nominal one) — use {func}`~escape.plot_utilities.dual_x_axis_plot` or
{func}`~escape.plot_utilities.add_crossing_secondary_axis` directly; both
work the same way but locate ticks by interpolating crossings of the paired
arrays instead of assuming the secondary quantity is a plain index.
