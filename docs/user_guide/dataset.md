# DataSet — Managed Storage

{class}`~escape.DataSet` is a container that groups multiple named Arrays (and
other Python objects) and optionally backs them to an HDF5 or zarr file.  It is
the recommended way to organise all channels from a single experiment run.

## Creating a DataSet

### With a new result file

```python
import escape

ds = escape.DataSet.create_with_new_result_file("run0042_reduced.esc.h5")
```

An escape results file carries `.esc` plus a backend suffix, `.h5` or
`.zarr`. Missing parts are filled in for you, the same way when creating,
opening or loading (`DataSet(results_file=...)`,
`create_with_new_result_file`, `load_from_result_file`):

| given | file used | |
|---|---|---|
| `run`, `scan_0.5V` | `run.esc.h5`, `scan_0.5V.esc.h5` | silently (default backend: h5) |
| `run.h5`, `run.esc` | `run.esc.h5` | with a `UserWarning` |
| `run.zarr` | `run.esc.zarr` | with a `UserWarning` |
| `run.esc.h5`, `run.esc.zarr` | unchanged | |

Only a trailing `.esc`/`.h5`/`.zarr` counts as a suffix, so a dot inside the
name (`scan_0.5V`) is kept. `ds.results_filepath` holds the path that was
actually used, and {func}`~escape.storage.dataset.normalize_result_filepath`
applies the same rule on its own.

If the file already exists and `force_overwrite=False`,
`create_with_new_result_file` asks whether to overwrite it when someone can
answer (a terminal or a Jupyter notebook). Otherwise, or if the answer is not
`y`, it raises `FileExistsError` and leaves the file untouched. Before escape
0.3 it returned `None` instead.

### Without a file (in-memory only)

```python
ds = escape.DataSet()
```

## Appending Data

{meth}`~escape.DataSet.append` accepts `escape.Array` objects, plain NumPy/dask
arrays, or arbitrary Python objects:

```python
from escape.storage.example_data import make_pump_probe_scan

sig, i0, pump_on, delay = make_pump_probe_scan(n_steps=10)

ds.append(sig,     name="signal")
ds.append(i0,      name="i0")
ds.append(pump_on, name="pump_on")
ds.append(delay,   name="delay")
```

After appending, channels are accessible as attributes:

```python
print(ds.signal.shape)      # (5000,)
print(ds.i0.scan.count())   # [500, 500, ...]
```

Serialisation is handled automatically:

* `escape.Array` → stored in the HDF5 group as a series of chunked datasets.
* Arbitrary Python objects → pickled or hickled depending on the file backend.

### What data can be stored

`escape.Array` and `escape.ArrayTimestamps` accept numpy and dask arrays,
lazy callables, and plain sequences:

* lists, tuples and other sequences are converted to numpy arrays when the
  array is created (or when `.data` is assigned); a list of equal-length lists
  becomes a 2-D array, while ragged lists raise `ValueError`;
* numbers mixed with `None` (for example a value read before a PV was
  connected) become floats, with `None` stored as NaN;
* dask arrays and callables are left lazy: nothing is loaded or called until
  the data is used;
* `ArrayTimestamps` raises `ValueError` if `data` and `timestamps` differ in
  length.

Data that has no HDF5/zarr equivalent (strings, Python objects) makes
`store()` raise a `TypeError` naming the array and the results file. The check
happens before anything is written, and a store that fails part-way removes
what it had written, so the file is never left with a half-stored array.
Appending timestamps that partly overlap the ones already stored raises
`ValueError`. Store such values with `ds.append(obj, name=...)` instead, which
pickles them.

## Loading a Saved DataSet

```python
ds = escape.DataSet.load_from_result_file("run0042_reduced.esc.h5")
print(list(ds.datasets.keys()))
# ['signal', 'i0', 'pump_on', 'delay']
```

`escape.Array` channels are loaded as lazy dask-backed Arrays — no data is read
until you call `.compute()` or access a reduction.

Because of this, arrays read from the file **only while it is open**. Either
keep the DataSet open while you work:

```python
with escape.DataSet.load_from_result_file("run0042_reduced.esc.h5") as ds:
    ds.signal.scan.plot()
```

or load the arrays you need into memory before closing it:

```python
ds = escape.DataSet.load_from_result_file("run0042_reduced.esc.h5")
sig = ds.signal.materialize()   # Array and ArrayTimestamps both have this
ds.close()
sig.scan.plot()                 # still works
```

An array first touched after its file was closed raises a `RuntimeError`
that says so.

### Damaged files

A file written by escape <= 0.2.14 can contain an array whose store failed
half-way (for example `timestamps_0000` without `data_0000`). Reading that
array raises an error naming the file and the group.
{func}`~escape.storage.dataset.check_result_file` lists the state of every
array, and with `repair=True` removes the incomplete parts so the rest of the
file loads again. The data that was never written cannot be recovered.

```python
from escape.storage.dataset import check_result_file

check_result_file("run0042.esc.h5")               # {'chan': 'orphan_timestamps', ...}
check_result_file("run0042.esc.h5", repair=True)
```

## Computing and Storing Multiple Arrays Efficiently

For dask-backed Arrays you can compute them all in one scheduler pass:

```python
# Derived quantities (still lazy)
sig_norm = sig / i0

# Store a batch of arrays efficiently — all dask graphs are fused
escape.store([ds.datasets["signal"], ds.datasets["i0"]])
```

Or compute into memory:

```python
sig_np, i0_np = escape.compute(sig, i0)
```

## Storing Small Quantities in Bulk

{meth}`~escape.DataSet.store_datasets_max_element_size` stores all Arrays
whose per-event element size is below a threshold (in number of values) in one
efficient batch.  This is useful after loading raw data and attaching derived
quantities:

```python
# Store all scalar or small-array channels (skip large detector images)
ds.store_datasets_max_element_size(max_element_size=5000)
```

## Using DataSet as a Context Manager

```python
with escape.DataSet.create_with_new_result_file("output.esc.h5") as ds:
    ds.append(sig, name="signal")
    escape.store([ds.datasets["signal"]])
# file is closed automatically
```

## Merging Multiple DataSets

{func}`~escape.storage.dataset.merge_datasets` concatenates all common
channels:

```python
ds1 = escape.DataSet.load_from_result_file("run0001_reduced.esc.h5")
ds2 = escape.DataSet.load_from_result_file("run0002_reduced.esc.h5")

merged = escape.merge_datasets([ds1, ds2])
print(len(merged.signal))   # combined event count
```

## Converting Between File Formats

To convert a zarr dataset to HDF5 (for sharing or archiving):

```python
from escape.storage.dataset import convert_resultsfile

convert_resultsfile("run0042_reduced.esc.zarr", out_type="h5")
```
