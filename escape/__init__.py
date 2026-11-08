"""
Event Synchronous Categorisation And Processing Environment,
(escape)

a high level, object oriented framework which abstracs event processing to high level objects. This is the successor project of ixppy, which used lazy evaluation steps for large data volumes. Here dask ist used for such evaluation allowing to scale evaluation on clusters or multi-processor machines.

The public names below are loaded on first use (PEP 562), so that
``import escape.bernina`` (diffractometer and detector calculations, numpy
only) works without the data stack (dask, h5py, zarr, ...), e.g. in a
browser (Pyodide/xeus-python). ``escape.Array``, ``from escape import Array``
and ``from escape import *`` behave as before.
"""

import importlib

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("escape-fel")
except PackageNotFoundError:
    __version__ = "unknown"

# name -> module it comes from (loaded on first access)
_LAZY = {
    **{name: "escape.storage" for name in (
        "Array",
        "ArrayTimestamps",
        "Scan",
        "ScanTimestamps",
        "concatenate",
        "store",
        "compute",
        "match_arrays",
        "escaped",
        "unravel_scans",
        "unravel_arrays",  # backward-compat alias for unravel_scans
        "digitize",
        "filter",
    )},
    **{name: "escape.storage.dataset" for name in ("DataSet", "merge_datasets", "convert_resultsfile")},
}
_SUBMODULES = ("storage", "utilities", "utilities_detectors")

__all__ = list(_LAZY) + list(_SUBMODULES)

STORAGE_LOCK = None


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"escape.{name}")
    else:
        raise AttributeError(f"module 'escape' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
