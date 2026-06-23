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

The filename **must** carry the `.esc` suffix (and `.h5` or `.zarr` to select
the backend).

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

## Loading a Saved DataSet

```python
ds = escape.DataSet.load_from_result_file("run0042_reduced.esc.h5")
print(list(ds.datasets.keys()))
# ['signal', 'i0', 'pump_on', 'delay']
```

`escape.Array` channels are loaded as lazy dask-backed Arrays — no data is read
until you call `.compute()` or access a reduction.

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
