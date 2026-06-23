# escape
## _Event Synchronous Categorisation And Processing Environment_

**escape** is a Python framework for event-based data analysis at free-electron laser (FEL) facilities. It pairs raw measurement arrays with pulse IDs and scan metadata, providing index-aligned arithmetic, scan-step aggregation, and scalable storage — designed for the data volumes and workflows typical at instruments like SwissFEL Bernina.

## Core concepts

- **`Array`** — wraps a data array together with a pulse-ID index and optional scan grouping. Index-aligned arithmetic between arrays is automatic: operations find common pulse IDs and align data before computing.
- **`Scan`** — partitions an Array into sequential steps with parameter metadata (e.g. delay values). Exposes per-step statistics (`nanmean`, `median`, `weighted_stat`, …) directly.
- **`Grid`** — maps scan steps onto an N-dimensional parameter grid, enabling multi-dimensional scans and 2D result arrays.
- **`DataSet`** — named collection of Arrays backed by HDF5 (`.esc.h5`) or zarr (`.esc.zarr`) files for persistent storage.
- **`escaped`** — decorator that lifts any NumPy/dask function to operate on Arrays with automatic index alignment.

## Quick example

```python
import escape as esc
import numpy as np

# Load a dataset
ds = esc.DataSet.load("run042.esc.h5")
sig  = ds["JF_signal"]
i0   = ds["I0_monitor"]
pump = ds["pump_on"]

# Normalize signal to I0 — pulse IDs aligned automatically
sig_norm = sig / i0

# Per-step median across the pump-probe scan
median_on  = sig_norm[pump].scan.nanmedian()
median_off = sig_norm[~pump].scan.nanmedian()
```

## Installation

**conda** (recommended — reuses existing environment packages):
```bash
conda install -c conda-forge escape-fel
```

**pip:**
```bash
pip install escape-fel
```

**Latest development version:**
```bash
pip install git+https://github.com/htlemke/escape-fel
```

## Documentation

Full documentation including user guide and API reference:
**[escape-fel.readthedocs.io](https://escape-fel.readthedocs.io)**

## Links

- [Documentation](https://escape-fel.readthedocs.io)
- [Source code](https://github.com/htlemke/escape-fel)
- [Issue tracker](https://github.com/htlemke/escape-fel/issues)
- [PyPI](https://pypi.org/project/escape-fel)
