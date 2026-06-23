# The Array Object

{class}`escape.Array` is the central object in `escape`.  It pairs a
multi-dimensional data array with a 1-D integer index (pulse IDs) and optional
scan metadata.

## Creating an Array

The minimal constructor arguments are `data` and `index`:

```python
import numpy as np
import escape

# 1-D scalar data — one float per pulse
sig = escape.Array(
    data=np.array([1.2, 0.9, 1.1, 0.8, 1.3]),
    index=np.array([100, 101, 102, 103, 104]),
    name="signal",
)
```

For higher-dimensional data (e.g. detector images) the first axis is always the
event axis:

```python
# 50 events, each a 512×512 image
imgs = escape.Array(
    data=np.random.rand(50, 512, 512).astype(np.float32),
    index=np.arange(50),
    name="detector",
)
```

### With scan metadata

Pass `step_lengths` (a list of per-step event counts) and `parameter` (a
dict of parameter values per step) to attach scan information:

```python
import numpy as np
import escape

delays_ps = np.linspace(-1, 5, 10)  # 10 delay steps
n_per_step = 500

data = np.random.randn(10 * n_per_step)
index = np.arange(len(data))

arr = escape.Array(
    data=data,
    index=index,
    step_lengths=[n_per_step] * 10,
    parameter={"delay_ps": {"values": list(delays_ps)}},
    name="bragg_intensity",
)
print(arr.scan)
# Scan over 10 steps
# Parameters delay_ps
```

### Using synthetic example data

The {mod}`escape.storage.example_data` module provides ready-made generators
for documentation and testing:

```python
from escape.storage.example_data import make_scan, make_pump_probe_scan

# simple 1-D scan
sig = make_scan(n_steps=10, n_events_per_step=500, scan_par_name="delay_ps")

# pump-probe dataset (returns 4 Arrays)
sig, i0, pump_on, delay = make_pump_probe_scan(n_steps=20)
```

## Key Properties

| Property | Description |
|----------|-------------|
| `array.data` | The underlying NumPy or dask array |
| `array.index` | 1-D NumPy array of integer pulse IDs |
| `array.shape` | Shape of `data` |
| `array.ndim` | Number of dimensions of `data` |
| `array.dtype` | Data dtype |
| `array.name` | Optional string label |
| `array.scan` | {class}`~escape.Scan` object with step metadata |
| `array.is_dask_array()` | `True` if data is backed by dask |

```python
print(sig.shape)   # (5000,)
print(sig.index[:5])  # [0 1 2 3 4]
print(sig.scan)    # Scan over 10 steps ...
```

## Dask-Backed Arrays

When parsing large raw data files, `escape` stores data as dask arrays —
no computation happens until you request results:

```python
arr.is_dask_array()   # True for lazily loaded data
```

To trigger computation and convert to NumPy:

```python
arr_np = arr.compute()
arr_np.is_dask_array()  # False
```

{func}`escape.compute` can materialise several Arrays in one pass, sharing the
dask scheduler for efficiency:

```python
sig_np, i0_np = escape.compute(sig, i0)
```

## Aggregation Methods

All standard NumPy-style reductions are available on Arrays.  When applied
without an axis argument they collapse the entire array; with `axis=0` they
reduce over the event axis.

```python
sig.mean()          # grand mean (scalar)
sig.nanmean()       # ignores NaN values
sig.nanstd()        # standard deviation ignoring NaN
sig.nanpercentile([25, 50, 75])  # quartiles
```

For multi-dimensional data (waveforms, images) pass an axis to reduce over
detector pixels while keeping the event axis:

```python
roi_sum = imgs[:, 100:200, 150:250].nansum(axis=(1, 2))
# roi_sum is an escape.Array with shape (N_events,)
```

## Slicing and Selection

Use standard Python slicing along the event axis:

```python
first_100 = arr[:100]       # first 100 events
every_other = arr[::2]      # every second event
```

Boolean arrays select a subset of events while preserving scan structure:

```python
good = sig > 0.5            # escape.Array of bool
sig_good = sig[good]        # only events where signal > 0.5
```

See [](array_operations.md) for advanced selection using another Array.
