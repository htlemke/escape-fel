# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`escape` (PyPI package `escape-fel`) is a Python framework for **event-based
data analysis at free-electron laser (FEL) facilities**, developed for the
SwissFEL Bernina beamline. It is the successor to the older `ixppy` project.
Every measurement is one value (or image/waveform) per X-ray pulse; `escape`'s
job is to keep per-pulse data from different instruments correctly aligned by
pulse ID and to scale that alignment to dask-backed, out-of-core data.

## Commands

**Editable install for development:**
```bash
pip install -e .
```
Version is derived from git tags via `setuptools-scm` (see `pyproject.toml`);
there is no hardcoded version string to bump.

**Build documentation (Sphinx + MyST + nbsphinx):**
```bash
pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
```

**Formatting:** the repo is set up for `black` (see `.vscode/settings.json`);
there is no `pyproject.toml` `[tool.black]` section or pre-commit config, so
there's no enforced line length/config beyond black's defaults.

**Tests:** there is no real test suite / pytest configuration in this repo.
`test_esc.py` and `escape/storage/tests.py` are legacy dev scratch files, not
a maintained test suite — don't assume `pytest` will find meaningful coverage
here. If you add tests, `escape.storage.example_data` (see below) is the
intended way to generate synthetic input data rather than depending on real
beamline files.

**Ad-hoc sanity checks:** the bare system `python3` is typically missing
`scipy`/`dask`/`pyarrow` and other runtime deps `escape` needs at import
time — prefer whatever interpreter/environment you normally use for this
project's deps over the bare `python3` on `PATH` when running quick
`python3 -c "..."` checks against this repo. (If a facility- or
machine-specific interpreter path applies, it belongs in the
gitignored `CLAUDE.local.md`, not here.)

**Releases:** version numbers come entirely from git tags via
`setuptools-scm` — no manual bump anywhere in source. Only tags matching
`v*.*.*` trigger `.github/workflows/publish.yml`, which builds and publishes
to PyPI via Trusted Publishing. Full maintainer steps live at
`docs/development/releasing.md` (linked into the Sphinx site via a
deliberately unlisted `:hidden:` toctree entry). Pre-`v`-prefix tags in
history (`0.1.0`, `0.2.0`, `test1`, …) predate this convention and aren't
real release markers.

**As of 2026-08-26, pushing `main` to `origin` DOES publish to PyPI**, via
`.githooks/pre-push` (each clone must opt in once with
`git config core.hooksPath .githooks`): it auto-tags the next PATCH version
and pushes that tag too, unless the pushed commit is already tagged (the
signal for "I already picked a MINOR/MAJOR version manually") or
`SKIP_AUTOTAG=1` is set for that push. See `docs/development/releasing.md`
for the exact behavior before assuming a plain `git push` is release-free.

## Architecture

### Public API surface

`escape/__init__.py` defines what's actually stable/public:
`Array`, `ArrayTimestamps`, `Scan`, `ScanTimestamps`, `DataSet`,
`concatenate`, `store`, `compute`, `match_arrays`, `escaped`,
`unravel_scans` (+ `unravel_arrays` back-compat alias), `digitize`, `filter`,
plus the `escape.utilities` / `escape.utilities_detectors` subpackages. If a
class/function isn't re-exported here, treat it as internal unless a docs
page says otherwise.

`escape.wavefront` and `escape.exafs` are separate **opt-in** subpackages —
real, documented, and installable (`docs/api/wavefront.rst`,
`docs/api/exafs.rst`) but *not* imported by `escape/__init__.py`, so they
only load on explicit `from escape import wavefront` / `import escape.exafs`.
See the dedicated section below.

**`escape/stream/` is not part of the stable API** (not imported by
`escape/__init__.py`). It provides live, pulse-by-pulse data acquisition
(`Stream`/`EventWorker`, mirroring the `Array` API — see
`docs/user_guide/stream.md`), with pluggable event handlers: `LocalEventHandler`
(direct bsread, no extra deps — used by the offline
`escape/stream/example_local_stream.ipynb` demo), `EventHandler_SFEL` (bsread +
dispatcher), and the `psi-datahub`-backed `DataHubEventHandler` /
`DataHubLocalEventHandler` / `MultiSourceEventHandler` (`escape/stream/es_wrappers_datahub.py`,
optional — degrades gracefully if `psi-datahub` isn't installed). As of
2026-09-04 this absorbed the former `escape/stream_new/` work-in-progress
rewrite (now removed) — don't expect that module name to exist any more in
older notes/branches. Do not reference `escape.stream` from documentation,
examples, or new integrations outside this stream work unless explicitly
asked to.

### Core data model — `escape/storage/storage.py` (~3900 lines, the heart of the package)

- **`Array`** — pairs a data array (numpy, dask, or a lazy callable) with an
  `index` of integer pulse IDs (`index_dim=0`, i.e. the first axis is always
  the event axis). Optionally carries `step_lengths` + `parameter` scan
  metadata. Reduction/elementwise methods (`nanmean`, `nansum`, `isnan`, …)
  are generated programmatically from a delegate-method table rather than
  hand-written per method — look there before adding a new one manually.
  `array.scan` lazily builds a `Scan`; `array.grid` exposes `scan.grid`.
  `array.tools` (`ArrayTools`, in `storage_tools.py`) exposes detector/ROI
  convenience methods bound to that Array.
- **Index-aligned arithmetic is the central design idea.** Two Arrays
  combined with `+ - * / // % ** & | ^ ~` or comparisons intersect their
  pulse-ID indices and operate only on the common events — mismatched pulse
  IDs are silently dropped. **The first operand determines output index
  order and scan-step grouping**; `a - b` and `-b + a` can have differently
  ordered (though equal-valued-per-ID) results. This convention holds for
  operators, `escaped()`-decorated functions, and `Array.categorize()`.
- **`escaped(func, convertOutput2EscData="auto")`** — decorator that lifts a
  plain numpy/dask function to operate on `Array`s: finds Array arguments,
  computes their pulse-ID intersection (`match_indexes`), reorders every
  Array's raw data onto the common IDs (first Array is the ordering
  reference, override with `escSorter=`), calls the wrapped function on raw
  arrays, and rewraps any output whose length matches the common event count
  back into an `Array` (scan metadata carried forward via
  `get_scan_step_selections`). All of `Array`'s arithmetic dunder methods are
  themselves generated at import time by wrapping `operator.*` with
  `escaped()` — arithmetic and `escaped()` are the same alignment machinery.
- **`Scan`** — partitions an `Array` into sequential steps from
  `step_lengths` + per-step `parameter` values. `scan.par_steps` is a
  `pandas.DataFrame`, one row per step; per-step stats (`nanmean`, `median`,
  `weighted_stat`, …) iterate steps and return one value each; indexing/
  slicing a `Scan` returns the corresponding sub-`Array`.
- **`Grid`** (also in `storage.py`) — maps `Scan` steps onto an N-D
  coordinate grid (`shape`, `positions`, `dimension_names`); supports N-D
  `__getitem__`, reshapes per-step stats into N-D arrays, and reports
  fill/sparsity for partially-completed multidimensional scans. Built via
  `grid_specs` on `Array`/`Scan`, or automatically by
  `unravel_scans`/`unravel_arrays` (Cartesian product of several 1-D scans)
  or by the SwissFEL parser when it detects a multi-dimensional scan.
- **`map_index_blocks`** — applies a function chunk-by-chunk over the event
  axis of a dask-backed Array (wraps dask `map_blocks`); the function
  receives a raw numpy block, not an `Array`. This is the tool for detector
  frame-by-frame processing (thresholding, droplet-finding, per-event
  fitting) as opposed to `escaped()`, which is for auto-aligning multiple
  Arrays together.
- Other module-level functions worth knowing: `concatenate` (merge Arrays
  along the event axis), `digitize` (bin an Array's data/index into scan
  steps), `filter` (boolean/range row selection preserving scan structure),
  `compute`/`store`/`store_all` (dask materialization and HDF5 persistence).

### Timestamp-based variant

**`escape/storage/storage_timestamps.py`** — `ArrayTimestamps`/
`ScanTimestamps` mirror `Array`/`Scan` but group events by **timestamp
intervals** (`scan.timestamp_intervals`, a list of `[start, stop]` bounds)
instead of discrete pulse-ID matching/`step_lengths`. There is no
`escaped()`-style pulse-ID intersection here — selection is by timestamp
range. Has its own parallel HDF5 persistence classes.

### Storage / persistence

- **`escape/storage/dataset.py`** — `DataSet`: a named container of
  `Array`/`ArrayTimestamps`/arbitrary Python objects, optionally backed by a
  `.esc.h5` (h5py) or `.esc.zarr` (zarr) results file — the `.esc` suffix is
  required and selects the backend by the following extension. Non-Array
  values are pickled/hickled with an `esc_type` attribute for round-tripping.
  `merge_datasets()` concatenates same-named channels across DataSets;
  `convert_resultsfile()` round-trips between the h5 and zarr backends.
- **`escape/storage/source.py`** — `Source`: internal provenance/laziness
  descriptor for how an Array's data can be (re)computed — one of
  `"factory"` (callable + args), `"dataset"`, `"status"`,
  `"array_map_index_blocks"`. Not user-facing API.
- **`escape/storage/storage_h5.py`** — dead file (syntax error, unused
  `Store` class stub) and not imported anywhere in the package. Don't build
  on it; if HDF5 array persistence is needed, that's in `storage.py`
  (`ArrayH5Dataset`/`ArrayH5File`).
- **`escape/storage/example_data.py`** — synthetic FEL-like data generators
  (`make_scan`, `make_pump_probe_scan`, `make_image_scan`,
  `make_grid_scan`, `make_detector_photon_stack`, …), all seedable. This is
  the intended source of example/test data in docs and code — don't use real
  beamline files or the `stream` test helpers for that purpose (see
  "not part of the stable API" above).

### Detector-image processing — `escape/utilities_detectors/`

Built on top of `map_index_blocks`, mostly also reachable as
`array.tools.<method>`:
- `droplets.py` — connected-component / single-photon "droplet" clustering.
- `dynamic_pedestal.py` — per-pixel dynamic-gain pedestal calibration for
  gain-switching detectors (Jungfrau-family). **Beta, not yet validated
  against real detector data** — treat recovered offsets as a hypothesis to
  verify, not a drop-in calibration (see the warning in
  `docs/api/detectors.rst`).
- `masking.py` — polygon-ROI masking of image stacks.
- `thresholding.py` — elementwise pixel value thresholding/clipping.

### Scientific reduction modules — `escape/wavefront/` and `escape/exafs/`

Both are opt-in subpackages (not imported from `escape/__init__.py`, see
"Public API surface" above) that apply a domain-specific reduction on top of
the core `Array`/`map_index_blocks` machinery, following the same lazy/dask
pattern as `utilities_detectors/`.

- **`escape/wavefront/`** — Talbot X-ray wavefront sensing, distilled from
  M. Seaberg's `lcls_beamline_toolbox`/`wfs_interface`. `propagation.py` is a
  pure-NumPy-FFT angular-spectrum propagator (forward + backward);
  `propagate_array()` runs it lazily over a dask-backed `Array` of complex
  fields via `map_index_blocks`. `talbot.py` does the forward simulation,
  Takeda-style fringe demodulation (`fourier_fringe_gradients`, needs a
  reference image), weighted-least-squares gradient integration
  (`integrate_gradients`), and `reconstruct_wavefront()` → `Wavefront`
  (curvature/focus). **Non-obvious calibration facts**: the recovered
  gradient is *minus* the cross-phase term, `/ (f_carrier · λ · zT)`; use the
  smooth `mesh_grating` at `fraction=0.5` for calibration (faithful
  magnifying self-image) — the π `checkerboard` fractional-Talbot revivals
  are nonlinear and need per-plane calibration instead. Optional
  `scikit-image` (`pip install escape-fel[wavefront]`) extends unwrap dynamic
  range; degrades gracefully without it. Docs: `docs/user_guide/wavefront.md`,
  `docs/api/wavefront.rst`; example: `examples/wavefront_sensor.ipynb`.
- **`escape/exafs/`** — EXAFS reduction, mu → chi(k) → chi(R), ported from a
  standalone `exafs_toolkit`. Single-spectrum steps (`energy_k.py`,
  `preedge.py`, `background.py` — AUTOBK spline, `fourier.py`, `utils.py`,
  `io.py`, `plotting.py`) mirror IFEFFIT/Larch conventions. `batch.py` is the
  escape/dask integration layer — `reduce_array`/`ft_array` lazily reduce a
  `(n_events, n_energy)` `Array` via `map_index_blocks` (same pattern as
  `wavefront.propagate_array`), plus `optical_density`/`common_k_grid`/
  `reduce_spectrum` helpers. Stops at chi(R) — no shell fitting (that's
  Larch/Demeter/Artemis territory). Validated against the bundled
  `escape/exafs/data/cu_rt01.xmu` Cu-foil reference (APS 13ID, via
  xraylarch examples). Docs: `docs/user_guide/exafs.md`,
  `docs/api/exafs.rst`; examples: `examples/exafs_quickstart.ipynb`,
  `examples/exafs_understanding_the_steps.ipynb`,
  `examples/exafs_batch_escape.ipynb`.

### Beamline integration — `escape/swissfel/`

SwissFEL-specific glue that turns raw experiment data into `Array`/`DataSet`
objects. `load_dataset_from_scan()` is the main documented entry point:
locates a scan's JSON metadata file by run number/pgroup, parses referenced
HDF5 data files, and returns a `DataSet` with one `Array` per detector
channel (auto-attaching a `run_number` scan parameter). Other files:
`cluster.py` (SwissFEL scan/cluster JSON metadata parsing),
`detector.py` (Jungfrau gain/pedestal correction via `jungfrau_utils`),
`recspace_conv.py` (reciprocal-space conversion for diffraction geometry),
`timetool.py` (X-ray/laser timing-tool signal processing).

### `escape/parse/`

Legacy/compat parsing tied to the deprecated `ixppy` project and raw
low-level SwissFEL JSON/HDF5 ingestion (`ixppy_tmp.py` is explicitly marked
"work in progress"). `escape/swissfel/parse.py` builds on top of this rather
than replacing it.

### Plotting & misc top-level modules

`escape/utilities.py` holds general helpers used throughout `storage.py`
(plotting, stats, interactive ROI/step-viewer widgets, `DataSet`'s
`StructureGroup`). `escape/plot_utilities.py` is the larger widget/plotting
toolkit it builds on. `escape/plot2D_dev.py` and `escape/plot_stack_dev.py`
are standalone dev/scratch files, not imported anywhere in the package.
`escape/cell2function.py` is an AST-based helper (`NameScanner`) for
deriving function parameters from notebook-namespace variables.

## Project status / known pending work

- **conda-forge packaging is mid-flight.** A staged-recipes submission is
  open at `conda-forge/staged-recipes` PR #33847 (from fork
  `htlemke/staged-recipe-escale-fel`). It sat with red CI and zero reviewer
  activity for two months because `meta.yaml` pinned `version = "0.1.0"`,
  which was **never actually published to PyPI** (source URL 404'd), and
  because `escape/storage/source.py` used to import `datastorage`
  unconditionally at module level — unlike `dataset.py`/`parse.py`, which
  already guarded it — so `import escape` would also have failed the test
  step, since `datastorage` isn't packaged on conda-forge. Fixed 2026-08-26:
  the import was made lazy and `datastorage` moved to an optional `legacy`
  extra (released as v0.2.2), and `meta.yaml`/the fork were updated to point
  at that release with correct deps/sha256, plus a review-request comment
  was posted. Check the PR for current CI/review status before assuming
  it's still stuck.
- **`escape/stream/` was rewritten 2026-09-04** to fix a class of bugs where
  the event loop silently never started on channel registration (a
  `hasattr(self, "loopThread")`-vs-`None` guard in
  `EventWorker.stopEventLoop()` let an `AttributeError` in the debounced
  restart-timer thread get swallowed by the default thread excepthook,
  aborting `_do_restart()` before `startEventLoop()` ran — see
  `escape/stream/escape_stream.py`'s `EventWorker.stopEventLoop`, now an
  identity check). The rewrite (formerly prototyped as the now-removed
  `escape/stream_new/`) exposes `Stream` (`EscData` kept as a back-compat
  alias) mirroring the `Array` API for live data: `Stream.digitize(bins)`
  returns a `StreamBinning` whose `.categorize(other_stream)` re-groups a
  second stream by that binning; `Stream[mask_stream]` / `.filter(mask_stream)`
  emits only events where the mask stream is truthy at the same pulse ID.
  Validated against the offline synthetic test stream
  (`escape/stream/testStream.py` + `LocalEventHandler`,
  `escape/stream/example_local_stream.ipynb`) — still not part of the stable
  API (see above), so treat it as real but pre-release.
- **Naming history**: `map_index_blocks` was previously called
  `map_event_blocks` — if you see the old name in older notebooks/docs
  references, it's the same method.

## Conventions to preserve when editing

- Keep the "first operand wins" index-ordering convention consistent across
  any new operator, `escaped()`-wrapped function, or `categorize`-like method.
- New reduction/elementwise `Array` methods should go through the
  `_ARRAY_DELEGATE_METHODS` table pattern in `storage.py` rather than being
  hand-written, unless the method needs genuinely custom alignment logic.
- Anything meant to be public should be re-exported from `escape/__init__.py`
  (and, for docs, added under `docs/api/`) — don't assume a class is public
  just because it's importable from a submodule.
