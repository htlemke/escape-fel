# Array Sorting and Operations

One of the most powerful features of `escape` is that operations between
Arrays automatically **align by pulse ID** — you never need to manually find
matching indices before computing.

## Arithmetic Operators

All standard Python arithmetic operators are supported and operate element-wise
on the intersection of the two index arrays:

```python
import numpy as np
import escape

a = escape.Array(data=np.array([8, 4, 3, 7]), index=np.array([0, 1, 3, 4]))
b = escape.Array(data=np.array([2, 2, 2, 2]), index=np.array([0, 1, 2, 4]))

c = a - b
print(c.index)  # [0 1 4] — only the 3 common pulse IDs
print(c.data)   # [6 2 5]
```

The first operand determines **index order and step grouping** of the result:

```python
d = escape.Array(data=np.array([8, 4, 3, 7]), index=np.array([2, 3, 0, 1]))
e = escape.Array(data=np.array([2, 2, 2, 2]), index=np.array([0, 1, 2, 3]))

f = d - e          # order from d
g = -e + d         # order from e

print(f.index)     # [2 3 0 1]
print(g.index)     # [0 1 2 3]
```

Supported operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `&`, `|`, `^`,
`~`, and comparison operators `<`, `<=`, `==`, `!=`, `>=`, `>`.

## Normalisation Example

```python
from escape.storage.example_data import make_pump_probe_scan

sig, i0, pump_on, delay = make_pump_probe_scan(n_steps=15)

# normalise signal by I0 — index alignment is automatic
sig_norm = sig / i0
```

## Boolean Filtering

A boolean Array (produced by a comparison) can be used to select a subset of
events.  The scan grouping is preserved in the result — only steps and events
that pass the filter are kept:

```python
good_shots = (i0 > 0.8) & (i0 < 1.2)  # bool escape.Array
sig_filt = sig[good_shots]
print(sig_filt.shape)   # fewer events, same number of steps
```

The {func}`~escape.storage.storage.filter` function does the same for
value-range filters on 1-D arrays:

```python
sig_filt = sig.filter(0.8, 1.2)   # keep events where 0.8 ≤ sig ≤ 1.2
```

## Matching and Aligning Arrays

### `match_arrays`

{func}`escape.match_arrays` returns a tuple of Arrays restricted to their
common pulse IDs, all ordered as the first argument:

```python
sig_m, i0_m = escape.match_arrays(sig, i0)
# sig_m and i0_m contain only the common pulse IDs, in sig's order
```

This is equivalent to `sig[np.isin(sig.index, i0.index)]` but also handles
differing scan structures.

### `categorize`

`array.categorize(other)` **re-sorts and re-groups** `other` according to
`array`'s index ordering and scan step boundaries, returning a new Array with
`other`'s data values but `array`'s structure:

```python
# Create a new grouping (e.g. time bins of 1000 pulse IDs each)
time_bins = sig.get_index_array(N_index_aggregation=1000)

# Apply that grouping to signal and I0
sig_rebinned = time_bins.categorize(sig)
i0_rebinned  = time_bins.categorize(i0)
```

After categorization, `sig_rebinned` and `i0_rebinned` share the same step
structure as `time_bins`, so per-step statistics can be compared directly.

:::{note}
`a.categorize(b)` is equivalent to `escape.match_arrays(a, b)[1]`.
:::

## Digitize — Sorting by Value

{func}`~escape.storage.storage.digitize` (also available as `array.digitize()`)
**re-sorts** events into groups defined by value bins, converting the Array into
a new Array where each "step" corresponds to one bin.  This is the primary way
to do pump-probe timetool correction or to bin data by any derived quantity.

```python
import numpy as np

# Sort signal by time-tool corrected delay
t_bins = np.linspace(-0.5e-12, 3e-12, 50)
sig_sorted = delay.compute().digitize(t_bins)

# Now sig_sorted has 50 steps = delay bins
print(len(sig_sorted.scan))   # up to 49 non-empty bins
```

The resulting Array carries per-step metadata with the bin centre, left edge,
and right edge values.

## Re-Sorting with Another Array

Using another Array as an index subscript applies that Array's *boolean data*
as an event filter (when it contains booleans):

```python
pump_on_bool = pump_on.astype(bool)

# Only pump-ON events
sig_on  = sig[pump_on_bool]

# Only pump-OFF events (reference shots)
sig_off = sig[~pump_on_bool]
```

## Concatenating Arrays

{func}`escape.concatenate` joins a list of Arrays along the event axis,
preserving per-step metadata:

```python
combined = escape.concatenate([scan_run1, scan_run2])
print(combined.scan.par_steps)  # steps from both runs
```

## HDF5 Storage and Loading

For computationally expensive derived quantities, store the result to an HDF5
file so you can reload it without recomputing:

```python
import h5py

with h5py.File("results.h5", "w") as fh:
    sig_norm.store(fh, "sig_norm")

# Later:
with h5py.File("results.h5", "r") as fh:
    sig_norm = escape.Array.load_from_h5(fh, "sig_norm")
```

See [](dataset.md) for higher-level storage via {class}`~escape.DataSet`.
