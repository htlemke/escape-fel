"""
Event Synchronous Categorisation And Processing Environment,
(escape)

a high level, object oriented framework which abstracs event processing to high level objects. This is the successor project of ixppy, which used lazy evaluation steps for large data volumes. Here dask ist used for such evaluation allowing to scale evaluation on clusters or multi-processor machines.

"""

import escape.storage as storage

from escape.storage import (
    Array,
    Scan,
    concatenate,
    store,
    compute,
    match_arrays,
    escaped,
)

from escape.storage.dataset import DataSet, merge_datasets, convert_resultsfile

from . import utilities


STORAGE_LOCK = None
