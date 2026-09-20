# Integrating live parse-cache warming into the DAQ client

This page is for whoever maintains the DAQ/acquisition client (the process
that already writes `scan_info_rel.json`, `status.json`, etc. into a run's
`aux/` directory) — not for `escape-fel` users or maintainers. It describes
a small, optional addition to that client: launching an `escape-fel`
subprocess once per scan that keeps a parse-result cache warm in the run's
own `aux/` directory while acquisition is happening, so that analysis-side
`load_dataset_from_scan()` calls — during acquisition or well after it —
skip the expensive step of scanning every raw HDF5 file's structure.

## Why this exists

`load_dataset_from_scan()` figures out each channel's shape/dtype/chunking
by opening every referenced raw HDF5 file once. For a run with dozens of
steps and a few hundred channels, that scan is the majority of parse time.
`escape-fel` already supports caching that result to disk
(`checknstore_parsing_result`) so a *second* call against the same run is
fast — but until now, nothing populated that cache until an analysis user
happened to call `load_dataset_from_scan()` themselves.

The fix is to have the entity best positioned to do this — the DAQ client
itself, which already knows exactly when each step's files are finished
writing — populate that cache as the scan runs, in the same directory it
already writes everything else to. Analysis-side calls made afterward (or
even *during* acquisition) then pick it up automatically, for free.

## What to add, and where

Add one subprocess launch per scan, right where your client already knows
the scan's metadata file path (the same `scan_info_rel.json` your client
itself writes into `aux/`):

```python
import subprocess

subprocess.Popen(
    [
        python_executable,  # whichever interpreter has escape-fel installed
        "-m", "escape.swissfel.live_reduce",
        "--mode", "daq-cache",
        "--run-number", str(run_number),
        "--pgroup", pgroup,
        # --exp-name is an alternative to --pgroup if that's what you have
    ],
    # no stdin/stdout/stderr redirection needed unless you want to keep the
    # log around — see "Logging" below
)
```

Launch it once, as soon as the scan's metadata file exists (i.e. once your
client has created the run's `aux/scan_info_rel.json`), and don't wait on
it — it's meant to run in the background for the duration of the scan and
exit on its own once it decides acquisition has finished (see below). Don't
launch a second one for the same run while one is still running.

```{important}
Launch it as a genuine child process of your own acquisition client (a
plain `subprocess.Popen`/`fork`+`exec`, not via `su`/`sudo` to a different
user, and not from a *separate* script run by a different account). It
needs to inherit your client's own filesystem write access to `aux/` —
that directory is normally not writable by anyone else, including by an
analysis user's own account even when they're in the experiment's own
group. Verified directly during development: even from a DAQ-adjacent
node, writing to a run's `aux/` as a regular account failed with `Read-only
file system` at the GPFS mount, while the DAQ process's own writes into
that same directory obviously already succeed. If your client runs as a
privileged/service account to write `scan_info_rel.json` etc. in the first
place, a child process of it already has everything it needs, with no
extra setup.
```

## What it actually does

`daq_cache_writer()` (`escape/swissfel/live_reduce.py`):

1. Locates the run's `scan_info_rel.json` from `--run-number`/`--pgroup` (or
   `--exp-name`), the same way `load_dataset_from_scan()` does.
2. Polls it every `--poll-interval` seconds (default 10). Each poll checks
   how many *leading* scan steps have every one of their referenced HDF5
   files fully readable — not just present on disk (see `count_ready_steps`
   in the same module for why `Path.exists()` isn't enough: a file can
   exist, with a plausible size, for over a second before it's actually
   complete).
3. Whenever that count grows, it calls `load_dataset_from_scan()` restricted
   to the new, ready prefix, with `checknstore_parsing_result="same_directory"`
   — which is what actually writes/extends the cache file next to
   `scan_info_rel.json` (`aux/scan_info_rel.parse_result_v03.json` for the
   default `parse_version=3`, `..._v02.json`/`..._v01... ` for the older
   parsers). No results file is created and no reduced data is kept —
   the `DataSet` built along the way is discarded; only the cache file is
   the point.
4. Stops once it sees neither a newly-ready step nor growth in the total
   step count for `--idle-polls-before-final-pass` consecutive polls
   (default 4 — roughly 40s of genuine silence on both fronts at the
   default poll interval), then does one last unrestricted pass to catch
   whatever finished right at the end, and exits.

It never blocks your client and never raises past its own process boundary
in a way that could affect acquisition — every parse/cache-write attempt is
wrapped in a `try`/`except` that logs and continues (see "Logging" below
for what that looks like if `aux/` isn't writable, e.g. if this is ever
accidentally launched from the wrong account).

## What analysis users get automatically

Nothing needs to change on the analysis side. `load_dataset_from_scan()`'s
`checknstore_parsing_result` parameter now defaults to `"auto"`: for each
run it's asked to load, it checks whether a `"same_directory"` cache
already exists next to that run's `scan_info_rel.json`, and uses it
automatically if so — this is exactly the file this subprocess maintains.
If it's not there (this subprocess was never launched for that run, hasn't
had time to write anything yet, or couldn't write there), `"auto"` falls
back to a `"work_directory"` cache under the pgroup's work directory
(`/das/work/pNNN/pNNNNN/.escape_parse_result/`), which is created and
updated by the analysis-side call itself — but only if that directory
already exists and is writable for the current user. If it isn't (missing
path, no permission, scan not under a `pNNNNN` directory), nothing is
cached and the full scan runs, silently and without raising.

## Also relevant: waiting for data mid-acquisition

Separately from the cache, `load_dataset_from_scan(..., wait_for_data_files=True)`
now also uses the same real-readiness check (rather than the old
existence-only check) when asked to block until referenced files show up.
This isn't something the DAQ client needs to call — it's for analysis
users who want to start a `load_dataset_from_scan()` call before a run has
finished and have it wait for outstanding files properly. Mentioned here
only so the two pieces aren't confused: `wait_for_data_files` blocks one
call; `daq_cache_writer` is the long-lived background process described
above.

## Logging

The subprocess prints its progress (one line per poll, plus scan output
from each `load_dataset_from_scan()` call) to stdout/stderr. If you want it
kept, redirect both when launching, e.g.:

```python
log = open(f"/path/writable/by/your/client/run{run_number:04d}_cache_writer.log", "w")
subprocess.Popen([...], stdout=log, stderr=subprocess.STDOUT)
```

A permission or read-only-filesystem failure while writing the cache shows
up as a single `Cannot write parse result cache: ...` line per attempt, not
a crash — the process keeps polling and simply never manages to persist
anything. If you see that in the log, it means the process doesn't have
the write access described above; double check it's a genuine child of
your client's own process/account rather than a separately-launched script.

## Verifying it's working

While a scan with this enabled is running (or right after), on a node with
read access to the run's `aux/` directory:

```bash
ls -la aux/scan_info_rel.parse_result_v03.json   # or _v02/_v01 depending on parse_version
```

should exist and grow in size as more steps complete. From Python:

```python
from escape.swissfel.parse import _resolve_auto_parsing_cache
_resolve_auto_parsing_cache("auto", "aux/scan_info_rel.json", parse_version=3)
# -> "same_directory" once the cache exists, False if not (yet)
```

And a normal analysis load should print `Using aux/ parse-result cache for
...` (from `load_dataset_from_scan`, when `verbose` is truthy) instead of
scanning every file from scratch.

## Command-line reference

```
python -m escape.swissfel.live_reduce --mode daq-cache \
    --run-number <N> (--pgroup <p12345> | --exp-name <name>) \
    [--instrument bernina] \
    [--poll-interval 10] \
    [--idle-polls-before-final-pass 5] \
    [--max-polls <N>] \
    [--parse-version 3]
```

`--max-polls` is a hard safety cap independent of the idle-detection logic
above — set it if you want a guaranteed upper bound on how long this
process can run regardless of activity (e.g. for a scan type that should
never take longer than X polls).
