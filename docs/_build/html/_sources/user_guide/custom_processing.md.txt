# Custom Processing

`escape` provides two complementary mechanisms for applying your own NumPy or
SciPy functions to event data while preserving index alignment and scan
structure.

---

## `map_index_blocks` — Block-Wise Processing

{meth}`~escape.Array.map_index_blocks` wraps **dask's `map_blocks`** and
applies a function to each chunk of the event axis independently.  This is the
right tool when:

* the function operates on a *batch* of events at once (a NumPy array), and
* you want the result to remain a lazy dask-backed escape Array.

### Basic usage

```python
import numpy as np
import escape
from escape.storage.example_data import make_image_scan

imgs = make_image_scan(n_steps=5, n_events_per_step=100)

# Apply a per-image threshold: set pixels below 4 keV to NaN
def threshold(block, thr):
    out = block.copy()
    out[out < thr] = np.nan
    return out

imgs_thres = imgs.map_index_blocks(threshold, 4.0)
# imgs_thres is still a lazy escape.Array
roi_sum = imgs_thres[:, 20:60, 30:80].nansum(axis=(1, 2))
```

### Changing the output shape

When the function changes the number of dimensions per event, pass
`new_element_size` (shape of the per-event output, excluding the event axis):

```python
# Reduce each image to a 1-D projection
def project(block):
    return block.sum(axis=1)   # (events, rows) → (events, cols)

proj = imgs.map_index_blocks(project, new_element_size=(imgs.shape[2],))
```

If the function produces a fixed-size output per event:

```python
from escape.swissfel import timetool

refstep = timetool.get_reference_function(width_px=30, reflen=150)

# Find step edge in each timetool projection row → (position, amplitude) pair
posamp = tt_proj.map_index_blocks(
    lambda block: np.array([timetool.find_signal(row, refstep) for row in block]),
    new_element_size=(2,),
    dtype=float,
)
tt_pos = posamp[:, 0]   # escape.Array of positions
tt_amp = posamp[:, 1]   # escape.Array of amplitudes
```

:::{note}
The function receives a **raw NumPy array** (one dask chunk), not an
`escape.Array`.  The first axis of the block is always the event axis.
:::

---

## `escaped` — Lifting Functions to Array Level

The {func}`~escape.storage.storage.escaped` decorator transforms a function
that works on **raw NumPy/dask data** into one that:

1. Detects any `escape.Array` arguments automatically.
2. Intersects their indices and aligns them.
3. Passes the aligned raw data arrays to the underlying function.
4. Wraps the outputs back as `escape.Array` objects with the correct indices
   and scan structure.

### Basic decoration

```python
import numpy as np
import escape

@escape.escaped
def normalise(signal, reference):
    return signal / reference

sig_norm = normalise(sig, i0)
# sig_norm is an escape.Array aligned on the common pulse IDs of sig and i0
```

Any mix of Array and scalar arguments works:

```python
@escape.escaped
def scale_and_shift(data, factor, offset):
    return data * factor + offset

result = scale_and_shift(sig, 2.0, -1.0)
```

### Multiple Array outputs

By default, `escaped` wraps *all* outputs that have the same length as the
index.  To wrap only specific outputs, pass `convertOutput2EscData`:

```python
@escape.escaped
def split(data):
    return data > 0, data[data > 0].mean()  # (bool Array, scalar)

mask, mean_val = split(sig)   # mask is an Array; mean_val is a Python float
```

### Choosing the "master" Array

When multiple Arrays are passed, the first one determines the output index
ordering.  You can override this with `escSorter`:

```python
result = normalise(sig, i0, escSorter=i0)   # use i0's ordering
```

### Wrapping existing NumPy functions

Any NumPy function can be lifted without subclassing:

```python
my_polyfit = escape.escaped(np.polyfit)

coeffs = my_polyfit(x_arr, y_arr, 1)   # linear fit per-pulse
```

---

## Comparison: When to Use Which

| Scenario | Use |
|----------|-----|
| Vectorised NumPy function over a *batch* of events (images, waveforms) | `map_index_blocks` |
| Per-event scalar computation (or loop over events) inside the function | `map_index_blocks` with inner loop |
| Function that mixes multiple escape Arrays and should auto-align | `escaped` decorator |
| Simple arithmetic / comparison between Arrays | Direct operators (`+`, `-`, `/`, …) |

---

## Writing a Per-Event Loop Inside `map_index_blocks`

If you need to process each event individually (e.g. call a fitting routine),
wrap the loop inside `map_index_blocks`:

```python
from scipy.optimize import curve_fit

def fit_peak_batch(block):
    """Fit a Gaussian to each row of block (events × pixels)."""
    x = np.arange(block.shape[1])
    results = np.zeros((len(block), 3))   # (amplitude, center, width)
    for i, row in enumerate(block):
        try:
            popt, _ = curve_fit(
                lambda x, a, mu, sig: a * np.exp(-(x - mu)**2 / (2 * sig**2)),
                x, row, p0=[row.max(), np.argmax(row), 5.0]
            )
            results[i] = popt
        except RuntimeError:
            results[i] = np.nan
    return results

fit_params = waveform_arr.map_index_blocks(fit_peak_batch, new_element_size=(3,), dtype=float)
amp   = fit_params[:, 0]   # escape.Array of peak amplitudes
center = fit_params[:, 1]  # escape.Array of peak positions
```
