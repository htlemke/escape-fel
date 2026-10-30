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

## Loading Part of a Scan

`step_selection` loads only some scan steps, e.g. for a quick first look at
a long scan, or at the steps already finished in a running one:

```python
ds = load_dataset_from_scan(run_number=42, pgroup="p12345",
                            step_selection=slice(0, 10))   # first 10 steps
ds = load_dataset_from_scan(run_number=42, pgroup="p12345",
                            step_selection=[0, 5, 9])      # or a list of steps
```

## Faster Reloading: the Parse Cache

Most of the time of a first load goes into scanning the structure of the raw
HDF5 files. `escape` stores the result of that scan (not the data) and reuses
it on the next load of the same files. With the default
`checknstore_parsing_result="auto"` this happens without any setup:

1. a cache in the run's own `aux/` directory is used if it exists (one the
   DAQ can write during acquisition, see {doc}`../development/daq_live_parsing`);
2. otherwise a cache in the pgroup's work directory
   (`/das/work/pNNN/pNNNNN`) is created and used, if that directory is
   writable;
3. otherwise nothing is cached. Missing paths or permissions never raise.

A reload that finds the cache is typically about ten times faster (a 121-step,
166-channel run: 35 s → 3.5 s). `clear_parsing_result=True` rebuilds it;
`checknstore_parsing_result=False` switches caching off.

## Adding the Channel-Monitor Data

During a run, eco's namespace monitor records the beamline's EPICS channels
(motor positions, temperatures, settings, …) whenever their value changes,
in `aux/namespace_monitor.ixp.h5`. `merge_data_sources=True` adds those
channels to the loaded DataSet:

```python
ds = load_dataset_from_scan(run_number=42, pgroup="p12345", merge_data_sources=True)
# prints: merged <n> channel(s) from aux/namespace_monitor.ixp.h5 (<m> shadowed by this run's own Array data)
```

The monitor channels are {class}`~escape.ArrayTimestamps` (value vs. time,
not one value per pulse) and are only read when first used, so even files
with tens of thousands of channels load in a second or two. Plot one against
clock time with `.plot()`.

When a name exists in more than one source, the more precise one wins: the
run's own per-pulse data over the monitor data, and the monitor data over
the snapshot taken at run start. The file is written at the very end of a
run; if it is missing or unreadable, a note is printed and everything else
loads as usual.

## Reducing a Run While It Is Being Acquired

`escape.swissfel.live_reduce` watches a run that is still being recorded.
Every poll it loads the steps whose files are complete and appends their
small (scalar or short-array) channels to a results file, so the reduced data
can be opened during the run and is ready when it ends:

```bash
python -m escape.swissfel.live_reduce --run-number 42 --pgroup p12345 \
    --results-directory ./results --poll-interval 15
```

or from Python:

```python
from escape.swissfel.live_reduce import live_reduce_scan

live_reduce_scan(run_number=42, pgroup="p12345", results_directory="./results")
```

It stops after several polls in which neither a new step became complete nor
the scan grew (`--idle-polls-before-final-pass`, default 4), then makes one
final pass over the whole run. Channels larger than `--max-element-size`
(default 5000 values per event, i.e. images) are not stored. Read the results
back at any time with
{meth}`DataSet.load_from_result_file <escape.DataSet.load_from_result_file>`.

The final pass also adds the run's channel-monitor data (see above); pass
`--no-merge-data-sources` to skip that.

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
