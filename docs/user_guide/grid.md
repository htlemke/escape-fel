# Grid — Multi-Dimensional Scans

When a scan sweeps over more than one parameter simultaneously (e.g. a 2-D
motor scan, or a delay × fluence matrix), the steps can be organised into an
N-D {class}`~escape.storage.storage.Grid`.

## What Is a Grid?

A Grid maps each scan step to a position in an N-D array of shape
`(n0, n1, ...)`.  It is attached to `array.scan.grid` (and also accessible
directly as `array.grid`).

The Grid stores:

* `shape` — dimensions of the grid, e.g. `(10, 8)` for a 10 × 8 scan.
* `positions` — lists of axis values, e.g. motor positions for each axis.
* `dimension_names` — labels for each axis.

## Creating a Grid

Grids are constructed automatically by the SwissFEL parser when it detects a
multi-dimensional scan pattern.

For quick testing and documentation examples use
{func}`~escape.storage.example_data.make_grid_scan`:

```python
from escape.storage.example_data import make_grid_scan

sig = make_grid_scan(
    shape=(5, 8),                          # 5 rows × 8 columns
    dim_names=("delay_ps", "motor_mm"),
    dim_ranges=((-0.5, 2.0), (0.0, 4.0)),
    n_events_per_step=200,
    seed=0,
)
print(sig.grid.shape)   # [5, 8]
```

For manual construction, pass `grid_specs` to {class}`~escape.Array`:

```python
import numpy as np
import escape, itertools

# 3×4 grid scan: steps 0..11 mapped to a 3×4 matrix
n_per_step = 200
x_vals = np.array([0.0, 1.0, 2.0])
y_vals = np.array([0.0, 0.5, 1.0, 1.5])
grid_indices = [{"grid_index": list(idx)} for idx in itertools.product(range(3), range(4))]
n_steps = len(grid_indices)

data  = np.random.randn(n_steps * n_per_step)
index = np.arange(len(data))

arr = escape.Array(
    data=data, index=index,
    step_lengths=[n_per_step] * n_steps,
    parameter={"scan_step_info": {"values": grid_indices}},
    grid_specs={
        "shape": [3, 4],
        "positions": [x_vals, y_vals],
        "grid_dimension_names": ["x_mm", "y_mm"],
    },
)
```

## Indexing a Grid

Subscript `array.grid` with N-D indices (one per axis) to retrieve the
corresponding scan steps as an Array:

```python
# Single step at grid position (1, 2)
step = arr.grid[1, 2]

# All steps along the first row:
row0 = arr.grid[0, :]

# A sub-region:
subgrid = arr.grid[1:3, 2:4]
```

Integer, slice, list, and array indices are supported.

## Grid Statistics

All per-step statistics from the Scan are also available on the Grid, with
results reshaped to the grid layout:

```python
grid_means = arr.grid.nanmean()    # numpy array of shape (3, 4)
grid_stds  = arr.grid.nanstd()     # shape (3, 4)
```

### Built-in 2-D Plotting

Pass `plot=True` to any grid statistic method to get an immediate 2-D colour
map.  The `plot` argument accepts three kinds of values:

| `plot=` value | Effect |
|---|---|
| `True` | draw on the current Matplotlib axes (`plt.gca()`) |
| a `Figure` | create a new subplot inside that figure |
| an `Axes` | draw on that specific axes |

```python
# Quickest option — current axes:
sig.grid.nanmean(plot=True)

# Specific axes:
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
sig.grid.nanmean(plot=ax, plot_kws={"cmap": "viridis"})

# Custom colourmap, fixed colour range, no colourbar:
sig.grid.nanmean(
    plot=True,
    plot_kws={"cmap": "plasma", "vmin": 0.0, "vmax": 1.0, "colorbar": False},
)

# Pass an Axes via plot_kws instead of via plot= (alternative syntax):
sig.grid.nanmean(plot=True, plot_kws={"axis": ax, "cmap": "magma"})
```

```{eval-rst}
.. plot::

   import matplotlib
   matplotlib.use("Agg")
   import matplotlib.pyplot as plt
   from escape.storage.example_data import make_grid_scan

   sig = make_grid_scan(shape=(5, 8), n_events_per_step=200, seed=0)

   fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
   sig.grid.nanmean(plot=axes[0], plot_kws={"cmap": "viridis"})
   axes[0].set_title("nanmean")
   sig.grid.nanstd(plot=axes[1],  plot_kws={"cmap": "plasma"})
   axes[1].set_title("nanstd")
   plt.tight_layout()
```

## Fill Count

Check how many grid positions are populated (useful for partially completed
scans):

```python
filled, total, pct = arr.grid.fill_count()
print(f"{filled}/{total} steps filled ({pct:.1f} %)")
```

## Combining Grids with `unravel_arrays`

{func}`~escape.storage.storage.unravel_arrays` creates a sorter Array that
spans the full Cartesian product of several 1-D scans:

```python
# sig_a has 10 steps (scan parameter A)
# sig_b has 8 steps (scan parameter B)
sorter = escape.storage.unravel_arrays(sig_a, sig_b)
# sorter.grid.shape == (10, 8)

# Categorise a third signal onto the 10×8 grid
sig_c_grid = sorter.categorize(sig_c)
print(sig_c_grid.grid.nanmean())   # shape (10, 8)
```

(binning-a-per-shot-channel)=
## Making an Axis out of a Per-Shot Channel

A scan axis does not have to be a motor. Any per-shot channel becomes one by
**binning** it: {meth}`~escape.Array.digitize` groups the shots into bins and
hands back an Array whose scan steps *are* those bins, and
{func}`~escape.storage.storage.unravel_scans` crosses that with the scan the
run already has.

The standard case is a jittering arrival time measured shot by shot with a
timetool. The laser–X-ray delay of each shot is only known after the fact, so
the run is an energy scan and the delay axis is made afterwards, as finely as
the statistics allow:

```python
import escape
from escape.storage.storage import unravel_scans

tt = d.bernina.tt_kb.edge_position_fs          # one arrival time per shot
mu = d.bernina.jet.det_jf.dap_roi1 / d.bernina.mon_mono.intensity

tt_bins = tt.digitize(bin_width=50, range=(-400, 400), unit="fs")
grid = unravel_scans(mu, tt_bins, categorize_target=mu)

grid.mesh.shape                 # (n_energies, n_bins)
grid.mesh.nanmean(plot=True)    # mu(E, arrival time) as a map
```

`bin_width=` is the bin **size** in the channel's own units. The bins sit on
the grid of multiples of that size — by default with the bin *centers* on the
multiples, so each bin is labelled by the {func}`escape.utilities.roundto`
value of the shots in it. The grid is fixed by the size alone, not by where
these particular shots happen to start, so two runs (or two cuts of one run)
binned the same way share their bins. `n_bins=` gives that many equal bins
instead, `range=` restricts what is covered (shots outside it are dropped), and
`align="edge"` puts the bin *edges*, rather than the centers, on the multiples.
{func}`~escape.storage.storage.bin_edges` builds such edges on their own.

When the per-shot channel is a timetool reading, the two steps are also
available as one call:

```python
grid = mu.tools.timetool_binning(
    tt, time_vec=None, time_bins=50.0,
    time_range=(-400, 400), cross_scan=True, unit="fs",
)
```

{meth}`~escape.storage.storage_tools.ArrayTools.timetool_binning` digitizes the
delay and crosses it with the run's own scan (`cross_scan=True`; without it the
delay bins *replace* that scan, which is what a pure delay scan wants). It can
also build the corrected delay for you, by adding each step's nominal delay to
the per-shot reading — `time_vec="auto"`. Do **not** use that here: on an
energy scan the "nominal delay" it would add is the mono setpoint.

Changing the bin size is the one knob worth turning: narrow bins resolve more
in time but put fewer shots in each one. `grid.mesh.count()` says how many
shots every cell actually got.

```{tip}
The axis is named after the channel that was binned (and labelled with the
`unit=` you give it), not after the `bin_center_...` parameter that carries it,
so `grid.mesh.select({"bernina.tt_kb.edge_position_fs": (-100, 100)})` and the
plot labels both read the way you would say it.
```

```{note}
A method over a grid of a few thousand cells reads every cell's events. If the
Array is still lazy, dask evaluates all cells in one pass — but computing the
per-shot Array **once** up front (`mu = (sig / i0).compute()`; a per-shot
channel is a few MB) is faster still, and pays off as soon as you try a second
bin size.
```
