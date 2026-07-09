# Loading SwissFEL Data

The `escape.swissfel` module provides a high-level entry point for reading
scan data recorded at SwissFEL into escape `Array` objects.  The main function
is {func}`~escape.swissfel.load_dataset_from_scan`.

## Overview

At SwissFEL, each scan produces a JSON *metadata file* (the scan-info file)
that records which detector channels were active, the scan-parameter values
for every step, and references to the raw HDF5 data files.
`load_dataset_from_scan` locates this file by run number, parses it together
with all referenced data files, and returns a
{class}`~escape.storage.DataSet` whose attributes are the detector channels as
escape {class}`~escape.Array` objects.

## Basic Usage

### Load by run number

The most common call pattern is to supply a run number together with the
experiment pgroup:

```python
from escape.swissfel import load_dataset_from_scan

ds = load_dataset_from_scan(run_number=42, pgroup="p12345")
```

For tab-completion convenience, `run_number` (singular) accepts a single
integer.  `run_numbers` (plural) accepts a single integer or a list:

```python
# all three are equivalent:
ds = load_dataset_from_scan(run_number=42,         pgroup="p12345")
ds = load_dataset_from_scan(run_numbers=42,        pgroup="p12345")
ds = load_dataset_from_scan(run_numbers=[42],      pgroup="p12345")
```

### Load by experiment name

If your pgroup is not at hand you can pass the experiment name and the pgroup
is looked up automatically:

```python
ds = load_dataset_from_scan(run_number=42, exp_name="my_experiment",
                            instrument="bernina")
```

### Load a specific metadata file

```python
ds = load_dataset_from_scan(
    metadata_file="/sf/bernina/data/p12345/res/scan_info/run0042.json"
)
```

## Accessing Channels

Each parsed detector channel appears as an attribute of the returned
{class}`~escape.storage.DataSet`:

```python
sig = ds.SARES11_SPEC125      # escape.Array
print(sig.shape)              # (N_events, ...)
print(sig.scan.par_steps)
#    delay_ps  run_number  step_length
# 0      -1.0          42          500
# 1       0.0          42          500
# ...
```

## Run Number in Scan Parameters

Every array loaded by `load_dataset_from_scan` carries a ``run_number`` scan
parameter that records which run each step came from.  The value is extracted
from the ``run{NNNN}`` token embedded in the metadata file path (filename stem
or parent directory).

This column appears automatically in {attr}`~escape.Scan.par_steps`:

```python
print(sig.scan.par_steps[["delay_ps", "run_number"]])
#     delay_ps  run_number
# 0       -1.0          42
# 1        0.0          42
# ...
```

## Loading Multiple Runs

Pass a list to `run_numbers` to load several runs at once.  The arrays are
concatenated along the event axis, and the `run_number` column lets you
distinguish steps that share the same scan parameter values:

```python
ds = load_dataset_from_scan(run_numbers=[42, 43, 44], pgroup="p12345")

sig = ds.SARES11_SPEC125
print(sig.scan.par_steps[["delay_ps", "run_number"]].head(6))
#    delay_ps  run_number  step_length
# 0      -1.0          42          500
# 1       0.0          42          500
# 2      -1.0          43          498
# 3       0.0          43          501
# 4      -1.0          44          500
# 5       0.0          44          500

# Select only run 43:
mask = sig.scan.par_steps["run_number"] == 43
run43_steps = mask[mask].index.tolist()
sig_run43 = sig.scan[run43_steps]
```

## Saving Results to a File

To persist the parsed data for fast reloading, pass a result filename:

```python
ds = load_dataset_from_scan(
    run_numbers=[42, 43],
    pgroup="p12345",
    result_filename="auto",       # derived from the first metadata file
    results_directory="./results",
    result_type="zarr",           # or "h5"
)
```

On a subsequent call you can skip re-parsing by setting ``load_result_only``:

```python
ds = load_dataset_from_scan(
    result_filename="run0042",
    results_directory="./results",
    load_result_only=True,
)
```

## API Reference

```{eval-rst}
.. autofunction:: escape.swissfel.load_dataset_from_scan
```
