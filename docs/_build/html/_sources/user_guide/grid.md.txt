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
multi-dimensional scan pattern.  For manual construction, pass `grid_specs` to
{class}`~escape.Array`:

```python
import numpy as np
import escape

# 3×4 grid scan: steps 0..11 mapped to a 3×4 matrix
n_steps = 12
n_per_step = 200
x_vals = np.array([0.0, 1.0, 2.0])    # rows
y_vals = np.array([0.0, 0.5, 1.0, 1.5])  # columns

# Build scan_step_info parameter: each step knows its grid index
import itertools
grid_indices = [{"grid_index": list(idx)} for idx in itertools.product(range(3), range(4))]

data  = np.random.randn(n_steps * n_per_step)
index = np.arange(len(data))

arr = escape.Array(
    data=data,
    index=index,
    step_lengths=[n_per_step] * n_steps,
    parameter={"scan_step_info": {"values": grid_indices}},
    grid_specs={
        "shape": [3, 4],
        "positions": [x_vals, y_vals],
        "grid_dimension_names": ["x_mm", "y_mm"],
    },
)
print(arr.grid)
# <Grid shape=(3, 4) dims=['x_mm', 'y_mm'] filled=12/12 (100.0%)>
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
map:

```python
arr.grid.nanmean(plot=True)
# displays a pcolormesh with x_mm and y_mm axes
```

Custom keyword arguments for the plot:

```python
arr.grid.nanmean(
    plot=True,
    plot_kws={"cmap": "viridis", "vmin": -1, "vmax": 1},
)
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
