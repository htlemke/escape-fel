"""
Event Synchronous Categorisation And Processing Environment,
(escape)

a high level, object oriented framework which abstracs event processing to high level objects. This is the successor project of ixppy, which used lazy evaluation steps for large data volumes. Here dask ist used for such evaluation allowing to scale evaluation on clusters or multi-processor machines.

"""

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("escape-fel")
except PackageNotFoundError:
    __version__ = "unknown"

import escape.storage as storage

from escape.storage import (
    Array,
    ArrayTimestamps,
    Scan,
    ScanTimestamps,
    concatenate,
    store,
    compute,
    match_arrays,
    escaped,
    unravel_scans,
    unravel_arrays,  # backward-compat alias for unravel_scans
    digitize,
    filter,
)

from escape.storage.dataset import DataSet, merge_datasets, convert_resultsfile

from . import utilities


STORAGE_LOCK = None
