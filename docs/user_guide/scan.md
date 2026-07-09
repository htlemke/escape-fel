# Scan Steps

A {class}`~escape.Scan` object partitions the event stream of an Array into
sequential **scan steps** — for example, one step per delay value in a
pump-probe scan.  It is always accessible via `array.scan`.

## What Is a Scan?

Internally, the Scan stores:

* `step_lengths` — a list `[n0, n1, n2, ...]` where `ni` is the number of
  events in step `i`.
* `parameter` — a dict mapping parameter names to their per-step values, e.g.
  `{"delay_ps": {"values": [-1.0, 0.0, 0.5, 1.0, ...]}}`.

```python
from escape.storage.example_data import make_scan

sig = make_scan(n_steps=10, n_events_per_step=500, scan_par_name="delay_ps",
                scan_par_values=list(range(-5, 5)))

print(sig.scan)
# Scan over 10 steps
# Parameters delay_ps

print(sig.scan.step_lengths)   # [500, 500, 500, ...]
```

## Parameter Table

{attr}`~escape.Scan.par_steps` returns a `pandas.DataFrame` with one row per
step:

```python
print(sig.scan.par_steps)
#    delay_ps  step_length
# 0      -5.0          500
# 1      -4.0          500
# ...
```

## Accessing Individual Steps

Subscript the scan to retrieve a step as an ordinary Array:

```python
step0 = sig.scan[0]       # first step
step5 = sig.scan[5]       # sixth step
last  = sig.scan[-1]      # last step

# Slices return an Array covering those steps:
first3 = sig.scan[0:3]    # steps 0, 1, 2 concatenated
```

Each returned Array has `step_lengths=[n]` and the parameter values for that
step, so it behaves exactly like a single-step scan.

## Per-Step Statistics

All statistics methods on `Scan` iterate over steps and return a list — one
value per step:

```python
means  = sig.scan.nanmean()   # list of per-step means
stds   = sig.scan.nanstd()    # list of per-step std devs
meds   = sig.scan.nanmedian() # list of per-step medians
counts = sig.scan.count()     # list of event counts per step
```

For multi-dimensional data the `axis` argument is forwarded:

```python
# Mean image per step for a 3-D (events, rows, cols) array
step_mean_imgs = img_arr.scan.mean(axis=0)   # list of 2-D arrays
```

Combined statistics:

```python
med, mad = sig.scan.median_and_mad()
```

### Inline Plotting of Statistics

Every per-step statistics method accepts a `plot` keyword that immediately
visualises the result without a separate `.plot()` call.  Scalar-per-step
results are drawn as a line; 1-D-array-per-step results (e.g. waveforms) are
rendered as a 2-D heat-map.

| `plot=` value | Effect |
|---|---|
| `True` | use the current Matplotlib axes (`plt.gca()`) |
| a `Figure` | open a new subplot inside that figure |
| an `Axes` | draw directly on that axes object |

```python
import matplotlib.pyplot as plt
from escape.storage.example_data import make_scan

sig = make_scan(n_steps=12, scan_par_name="delay_ps",
                scan_par_values=list(range(-5, 7)), seed=0)

# Quick plot on the current axes:
sig.scan.nanmean(plot=True)

# Plot on a specific axes and customise style:
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
sig.scan.nanmean(plot=axes[0], plot_kws={"marker": "o", "color": "steelblue"})
sig.scan.nanstd(plot=axes[1],  plot_kws={"color": "coral"})
plt.tight_layout()
```

```{eval-rst}
.. plot::

   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from escape.storage.example_data import make_scan

   sig = make_scan(n_steps=12, scan_par_name="delay_ps",
                   scan_par_values=list(range(-5, 7)), seed=0)

   fig, axes = plt.subplots(1, 2, figsize=(8, 3))
   sig.scan.nanmean(plot=axes[0], plot_kws={"marker": "o", "color": "steelblue"})
   axes[0].set_title("nanmean per step")
   sig.scan.nanstd(plot=axes[1], plot_kws={"color": "coral"})
   axes[1].set_title("nanstd per step")
   plt.tight_layout()
```

The `plot_kws` dict is forwarded directly to the underlying Matplotlib call, so
any valid `Axes.plot` keyword works.  For 2-D results, `plot_kws` is forwarded
to {func}`~escape.utilities.plot2D`, and the special key `"colorbar"` (bool,
default `True`) controls whether a colourbar is added.

## Plotting a Scan

`scan.plot()` produces an errorbar plot with the scan parameter on the x-axis
and the per-step median (with 1σ confidence of the mean) on the y-axis:

```python
import matplotlib.pyplot as plt

sig.scan.plot()
plt.xlabel("delay / ps")
plt.ylabel("signal (a.u.)")
plt.tight_layout()
plt.show()
```

## Step Histogram

`scan.hist()` plots a 2-D colour map of per-step value histograms — useful for
visualising shot-to-shot fluctuations along a scan:

```python
x, bins, hdata = sig.scan.hist(bins=50, normalize_to="max")
```

`hist()` always produces a plot, so it computes dask-backed data automatically;
for large arrays it warns before triggering that computation. Call
`.compute()` beforehand yourself if you'd rather control when that happens.

## Scan Arithmetic

Binary operators applied between a Scan and a scalar (or a list with the same
length as the scan) perform per-step operations:

```python
# Subtract the per-step mean from each step's data
step_means = sig.scan.nanmean()
corrected  = sig.scan - step_means   # escape.Array
```

## Number of Events Per Step

```python
sig.scan.count()           # list of ints
```

## Merging Scans from Multiple Runs

{meth}`~escape.Scan.merge_scans` combines data from several scans at the same
parameter points, pooling events together:

```python
sig_run1 = make_scan(n_steps=5, scan_par_name="angle")
sig_run2 = make_scan(n_steps=5, scan_par_name="angle")

merged = sig_run1.scan.merge_scans(sig_run2.scan,
                                   roundto_interval=0.01,
                                   par_name="angle")
```
