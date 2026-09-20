from asyncio import run
import pickle
import re
import shutil
import time
from unicodedata import name

import numpy as np
from escape.storage.storage import concatenate, Array, Scan
from ..parse.swissfel import readScanEcoJson_v01, parseScanEco_v01
from .cluster import (
    parseScanEcoV01,
    parseScanEcoV02,
    parseScanEcoV03,
    h5_file_ready,
    _resolve_cache_path_v02,
    _resolve_cache_path_v03,
    _safe_exists,
    _usable_work_cache_parent,
)
from pathlib import Path
import json
import pathlib
import warnings
import logging
import escape
from copy import deepcopy as copy
from lazy_object_proxy import Proxy
import oschmod

import traceback

# from ipytree import Node

# from ipytree import Tree as Treejs

try:
    import bitshuffle.h5
except:
    print("Could not import bitshuffle.h5!")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm
import h5py
import zarr
from rich.tree import Tree

logger = logging.getLogger(__name__)


_RESULT_FILE_TYPE_SUFFIXES = {"h5": ".h5", "zarr": ".zarr"}


def _normalize_result_filepath(path, result_type):
    """Ensure *path* ends with the ``.esc.<result_type>`` suffix pair that
    :func:`~escape.storage.dataset.filespec_to_file` expects.

    A path with no recognized suffix at all (e.g. ``"run70"``) gets one
    added silently. A path with a suffix that doesn't match *result_type*
    (e.g. ``.h5`` while ``result_type="zarr"``, or a missing ``.esc``) has
    its trailing ``.esc``/``.h5``/``.zarr`` suffixes stripped and replaced,
    with a warning, since silently writing to a different path than the one
    given is otherwise easy to miss.
    """
    path = Path(path)
    type_suffix = _RESULT_FILE_TYPE_SUFFIXES[result_type]
    canonical_suffixes = [".esc", type_suffix]
    if path.suffixes[-2:] == canonical_suffixes:
        return path

    stem = path.name
    had_suffix = False
    while Path(stem).suffix in (".esc", ".h5", ".zarr"):
        had_suffix = True
        stem = Path(stem).stem

    new_path = path.with_name(stem + "".join(canonical_suffixes))
    if had_suffix:
        warnings.warn(
            f"result_file {path.name!r} does not have the expected "
            f"'.esc{type_suffix}' suffix for result_type={result_type!r} "
            f"— using {new_path.name!r} instead.",
            stacklevel=2,
        )
    return new_path


def _extract_run_number(metadata_file):
    """Return the run number embedded in a metadata file path, or None.

    Searches each path component from right to left for the pattern
    ``run{NNNN}`` (4 or more digits), e.g. ``run0042`` in either the
    filename stem or a parent directory such as ``raw/run0042/aux/``.
    """
    for part in reversed(Path(metadata_file).parts):
        m = re.search(r'run(\d{4,})', part)
        if m:
            return int(m.group(1))
    return None


read_scan_json = readScanEcoJson_v01
parse_scan = parseScanEcoV01
# parse_file = parseSFh5File_v01


def parse_run(
    runno, pgroup, instrument, json_dir="/sf/{instrument}/data/{pgroup}/res/scan_info/"
):
    files = Path(json_dir.format(instrument=instrument, pgroup=pgroup)).glob(
        f"run{runno:04d}*"
    )
    return parse_scan(next(files))


from escape.utilities import StructureGroup, dict2structure, name2pgroups
from escape.storage import DataSet

# class StructureGroup:
#     def __repr__(self):
#         s = object.__repr__(self)
#         s += "\n"
#         s += "items\n"
#         for k in self.__dict__.keys():
#             s += "    " + k + "\n"
#         return s

#     def get_structure_tree(self, base=None):
#         if not base:
#             base = Tree("")
#         for key, item in self.__dict__.items():
#             if hasattr(item, "get_structure_tree"):
#                 item.get_structure_tree(base=base.add(key))
#             else:
#                 base.add(key).add(str(item))
#         return base


# def dict2structure(t, base=None):
#     if not base:
#         base = StructureGroup()
#     for tt, tv in t.items():
#         p = tt.split(".")
#         tbase = base
#         for tp in p[:-1]:
#             if tp in tbase.__dict__.keys():
#                 if not isinstance(tbase.__dict__[tp], StructureGroup):
#                     tbase.__dict__[tp] = StructureGroup()
#             else:
#                 tbase.__dict__[tp] = StructureGroup()
#             tbase = tbase.__dict__[tp]
#         if hasattr(tbase, p[-1]):
#             if not isinstance(tbase.__dict__[p[-1]], StructureGroup):
#                 tbase.__dict__[p[-1]] = tv
#         else:
#             tbase.__dict__[p[-1]] = tv
#     return base


def interpret_raw_data_definition(
    metadata_file=None,
    run_numbers=None,
    pgroup=None,
    exp_name=None,
    instrument="bernina",
    search_path=[
        "{instrument:s}/data/{pgroup:s}/raw/run{run_number:04d}/aux/scan_info*.json",
        "{instrument:s}/data/{pgroup:s}/work/raw/run{run_number:04d}/aux/scan_info*.json",
        "{instrument:s}/data/{pgroup:s}/res/scan_info/run{run_number:04d}*.json",
    ],
    verbose=True,
):
    # format search paths
    if isinstance(search_path, str):
        search_path = [search_path]
    if isinstance(run_numbers, int):
        run_numbers = [run_numbers]

    if metadata_file:
        return [metadata_file]
    if run_numbers and exp_name and instrument:
        rpgs = name2pgroups(exp_name, beamline=instrument)
        pgroup = rpgs[0][1]
        if len(rpgs) > 1:
            print(f"Found multiple pgroups, {rpgs}, choosing {pgroup}")

    if run_numbers and pgroup and instrument:
        metadata_files = []
        for run_number in run_numbers:
            for tsp in search_path:
                tfiles = list(
                    Path("/sf").glob(
                        tsp.format(
                            instrument=instrument, pgroup=pgroup, run_number=run_number
                        )
                    )
                )
                if len(tfiles) < 1:
                    if verbose:
                        print(
                            f"No files found using {('/sf/' + tsp.format(instrument=instrument, pgroup=pgroup, run_number=run_number)):s}"
                        )
                    continue
                if 1 < len(tfiles):
                    print(
                        "WARNING:found more than one file matching raw data definition! Taking fist one."
                    )
                tfile = tfiles[0]
                break
            if verbose:
                print(f"Found metadatafile {tfile.as_posix()}")
            metadata_files.append(tfile.as_posix())
        return metadata_files


def _wait_for_data_files_on_disk(
    metadata_file,
    search_paths=["./", "./scan_data/", "../scan_data"],
    exclude_from_files=[],
    poll_interval=10,
    timeout=None,
    verbose=True,
):
    """Block until every raw data file referenced by a scan-info JSON exists.

    Repeatedly re-reads ``metadata_file`` and checks each file referenced in
    its ``scan_files`` against ``search_paths``, using the same path
    resolution rules as the parsers (e.g.
    :func:`~escape.swissfel.cluster.parseScanEcoV01`). Returns as soon as
    every file for every scan step is present on disk. This only gates
    *finishing* the dataset — the parsers themselves already tolerate
    parsing whatever files are available while some are still missing.

    Parameters
    ----------
    metadata_file : str or Path
        Path to the scan-info JSON file.
    search_paths : list of str, optional
        Local directories searched for raw HDF5 data files referenced by the
        metadata JSON, same semantics as ``load_dataset_from_scan``.
    exclude_from_files : list of str, optional
        Channel names or file patterns to skip, same as elsewhere.
    poll_interval : float, optional
        Seconds to sleep between filesystem re-checks. Defaults to ``10``.
    timeout : float or None, optional
        Give up and raise ``TimeoutError`` after this many seconds of
        waiting. ``None`` (default) waits indefinitely.
    verbose : bool, optional
        Print how many of the referenced files are present on each poll.

    Returns
    -------
    dict
        The scan-info JSON (``s``) as last read, once every referenced file
        is present.
    """
    metadata_file = Path(metadata_file)
    start = time.monotonic()
    run_root_directory = None
    if metadata_file.parent.stem == "aux":
        run_root_directory = metadata_file.parent.parent

    while True:
        s, scan_info_filepath = readScanEcoJson_v01(
            metadata_file, exclude_from_files=exclude_from_files
        )

        missing = []
        n_total = 0
        for files_step in s["scan_files"]:
            searchpaths = None
            for fina in files_step:
                n_total += 1
                fp = Path(fina)
                if (not fp.is_absolute()) and run_root_directory:
                    fp = run_root_directory / fp
                fn = Path(fp.name)
                if not searchpaths:
                    searchpaths = [fp.parent] + [
                        scan_info_filepath.parent / Path(tp.format(fp.parent.name))
                        for tp in search_paths
                    ]
                if not any(h5_file_ready(path / fn) for path in searchpaths):
                    missing.append(fp)

        if not missing:
            if verbose:
                print(
                    f"All {n_total} data file(s) referenced in {metadata_file.name} are present."
                )
            return s

        if verbose:
            print(
                f"Waiting for data files: {n_total - len(missing)}/{n_total} present "
                f"({len(missing)} missing) in {metadata_file.name} ..."
            )

        if timeout is not None and (time.monotonic() - start) > timeout:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for {len(missing)} data file(s) "
                f"referenced by {metadata_file} (still missing, e.g. "
                f"{[m.as_posix() for m in missing[:5]]}"
                f"{', ...' if len(missing) > 5 else ''})"
            )

        time.sleep(poll_interval)


def _with_scan_parameter(ar, make_parameter, defer=False):
    """Append the scan parameter ``make_parameter(ar)`` to *ar* and return it.

    With ``defer=True`` and a lazy ``Proxy`` array, the append is deferred
    to the array's first use instead of forcing the array to be built (and,
    for ``lazy_esc_array_parsing``, its files scanned) right here.
    """

    def apply(a):
        a.scan.append_parameter(make_parameter(a))
        return a

    if defer and type(ar) is Proxy:
        return Proxy(lambda: apply(ar.__wrapped__))
    return apply(ar)


def _resolve_auto_parsing_cache(checknstore_parsing_result, metadata_file, parse_version):
    """Resolve ``checknstore_parsing_result="auto"`` for one metadata file.

    A DAQ-side process can maintain a parse-result cache right next to the
    scan-info JSON (``checknstore_parsing_result="same_directory"``, i.e. in
    the run's ``aux/`` directory) throughout acquisition -- see
    ``escape.swissfel.live_reduce``'s ``daq_cache_writer`` for exactly that.
    When that cache already exists, later analysis-side calls should use it
    automatically rather than scanning from scratch. This only ever *reads*
    that location; it never tries to create it there, since ``aux/`` is
    normally only writable by the DAQ/acquisition process, not by analysis
    users. If it doesn't exist, a ``"work_directory"`` cache is used when
    the pgroup's work directory exists and is writable, else no caching.

    Any value other than the literal string ``"auto"`` passes through
    unchanged -- this is purely about resolving the new default.
    """
    if checknstore_parsing_result != "auto":
        return checknstore_parsing_result

    scan_info_filepath = Path(metadata_file)
    if parse_version == 1:
        cache_path = scan_info_filepath.with_suffix(".parse_result.json")
    elif parse_version == 2:
        cache_path = _resolve_cache_path_v02("same_directory", scan_info_filepath)
    else:
        cache_path = _resolve_cache_path_v03("same_directory", scan_info_filepath)

    if _safe_exists(cache_path):
        return "same_directory"
    # No DAQ-side cache: try a work-directory one so analysis-side re-loads
    # don't re-scan every file. Only if that directory already exists and is
    # writable -- otherwise silently no caching (never an error).
    return "work_directory" if _usable_work_cache_parent(scan_info_filepath) else False


def load_dataset_from_scan(
    metadata_file=None,
    run_number=None,
    run_numbers=None,
    pgroup=None,
    exp_name=None,
    instrument="bernina",
    results_directory="./",
    result_filename=None,
    result_type="zarr",
    result_file=None,
    load_result_only=False,
    clear_result_file=False,
    store_status=True,
    search_path=[
        "{instrument:s}/data/{pgroup:s}/raw/run{run_number:04d}/aux/scan_info*.json",
        "{instrument:s}/data/{pgroup:s}/work/raw/run{run_number:04d}/aux/scan_info*.json",
        "{instrument:s}/data/{pgroup:s}/res/scan_info/run{run_number:04d}*.json",
    ],
    search_paths=["./", "./scan_data/", "../scan_data"],
    memlimit_MB=100,
    createEscArrays=True,
    lazyEscArrays=True,
    lazy_esc_array_parsing=False,
    exclude_from_files=[],
    checknstore_parsing_result="auto",
    clear_parsing_result=False,
    analyze_namespace_info=False,
    name="delme",
    alias_mappings={},
    step_selection=slice(None),
    load_dap_data=False,
    raw_data_suffix="_rawdata",
    append_scan_parameter: {"par_name": {"values": list}} = None,
    verbose=0,
    perm_result_file='g+rw',
    parse_version=3,
    wait_for_data_files=False,
    wait_poll_interval=10,
    wait_timeout=None,
    merge_data_sources=False,
):
    """Load detector and scan-parameter data from one or more SwissFEL scan runs.

    Locates the scan metadata JSON file(s) for the requested run(s), parses all
    recorded detector channels into escape :class:`~escape.Array` objects, and
    returns them collected in a :class:`~escape.storage.DataSet`.  When multiple
    runs are loaded they are concatenated along the event axis so the resulting
    arrays span all runs seamlessly, with a ``run_number`` scan parameter added
    automatically to identify which step originated from which run.

    Parameters
    ----------
    metadata_file : str or Path, optional
        Direct path to a scan-info JSON file.  Takes precedence over
        ``run_number``/``run_numbers`` when provided.
    run_number : int, optional
        Single run number.  Convenience alias for ``run_numbers=[run_number]``
        that avoids wrapping a scalar in a list and aids tab-completion.
        Ignored when ``run_numbers`` is also given.
    run_numbers : int or list of int, optional
        One or more run numbers to load.  The corresponding metadata files are
        located automatically using ``search_path``.  A single integer is
        accepted as well as a list.
    pgroup : str, optional
        SwissFEL pgroup identifier, e.g. ``"p12345"``.  Required together with
        ``run_numbers`` unless ``exp_name`` is supplied instead.
    exp_name : str, optional
        Experiment name used to look up the pgroup automatically via
        :func:`~escape.utilities.name2pgroups`.  Alternative to ``pgroup``.
    instrument : str, optional
        Beamline name, e.g. ``"bernina"`` (default) or ``"alvra"``.
    results_directory : str or Path, optional
        Directory where the result file is written when ``result_filename`` is
        set.  Defaults to the current working directory.
    result_filename : str or None, optional
        Base name for the result file.  Pass ``"auto"`` to derive the name from
        the first metadata file.  ``None`` (default) skips result-file creation.
    result_type : {"zarr", "h5"}, optional
        Storage backend for the result file.  Defaults to ``"zarr"``.
    result_file : h5py.File or zarr.Group, optional
        An already-open result file to use directly, bypassing
        ``result_filename``/``results_directory``.
    load_result_only : bool, optional
        If ``True``, skip parsing and load directly from an existing result
        file at the path derived from ``results_directory`` and
        ``result_filename``.  Defaults to ``False``.
    clear_result_file : bool, optional
        Delete an existing result file before writing.  Defaults to ``False``.
    store_status : bool, optional
        Persist the run-status JSON as pickled objects inside the DataSet.
        Defaults to ``True``.
    search_path : list of str, optional
        Glob patterns (relative to ``/sf/``) used to find the metadata file for
        each run number.  The tokens ``{instrument}``, ``{pgroup}``, and
        ``{run_number}`` are substituted at runtime.  The first pattern that
        yields a match is used.
    search_paths : list of str, optional
        Local directories searched for raw HDF5 data files referenced by the
        metadata JSON.  Defaults to ``["./", "./scan_data/", "../scan_data"]``.
    memlimit_MB : int, optional
        Approximate per-step memory limit in MB for lazy loading.  Defaults to
        ``100``.
    createEscArrays : bool, optional
        Build escape :class:`~escape.Array` objects from the parsed data.
        Defaults to ``True``.
    lazyEscArrays : bool, optional
        Use dask-backed lazy arrays instead of loading data into memory
        immediately.  Defaults to ``True``: eager array construction is
        unchanged from the older parsers and does not benefit from the
        scanning speedup, so building lazily avoids paying that cost up
        front.  Pass ``False`` to compute arrays eagerly as before.
    lazy_esc_array_parsing : bool, optional
        Experimental, off by default. ``parse_version=3`` only. ``False`` (default): every new data file
        is opened up front to learn each channel's shape (the dominant cost
        of a cold load). ``True``: only the channel names are determined up
        front (from one representative file per file kind); a channel's
        shape, dtype, step lengths and data are determined -- by scanning
        that channel's file kind once -- the first time the array is
        touched (``.shape``, ``.data``, ``.scan``, ...). Kinds whose arrays
        are never used are never opened, and kinds without channels (e.g.
        PVDATA) are skipped. Assumes the channel set of each file kind is
        the same in every file of the run, and only stays lazy end to end
        without a results file (with one, aliased arrays are touched when
        the DataSet is created). See ``parseScanEcoV03``. Later intended
        to replace the camel-case ``lazyEscArrays``.
    exclude_from_files : list of str, optional
        Channel names or file patterns to skip during parsing.
    checknstore_parsing_result : bool or str, optional
        Cache the intermediate parsing result on disk and reuse it on
        subsequent calls with the same inputs. ``"auto"`` (default): for
        each run, check whether a ``"same_directory"`` cache (i.e. one
        living in the run's own ``aux/`` directory, next to its scan-info
        JSON) already exists -- e.g. one a DAQ-side process maintained
        throughout acquisition, see ``escape.swissfel.live_reduce``'s
        ``daq_cache_writer`` -- and use it automatically if so, without
        ever trying to create one there itself (``aux/`` is normally not
        writable by analysis users). If nothing is found there, falls back
        to a ``"work_directory"`` cache (created/updated as needed) when the
        pgroup's work directory exists and is writable, otherwise to no
        caching -- a missing path or missing permissions never raises. Pass ``"same_directory"``, ``"work_directory"``, or
        an explicit path to force a specific cache location and skip the
        ``aux/``-first check; ``False`` disables caching outright.
    clear_parsing_result : bool, optional
        Clear a cached parsing result before re-parsing.  Defaults to
        ``False``.
    name : str, optional
        Human-readable label attached to the returned DataSet.
    alias_mappings : dict, optional
        Mapping from raw channel names to short alias names, e.g.
        ``{"SARES11-SPEC125-M1.roi1_background_subtracted": "spec"}``.
        Aliases from the scan metadata are applied first; this dict takes
        precedence.
    step_selection : slice or list of int, optional
        Subset of scan steps to load.  Defaults to ``slice(None)`` (all steps).
    load_dap_data : bool, optional
        Additionally load DAP (data-analysis pipeline) result files from the
        run directory.  Defaults to ``False``.
    raw_data_suffix : str or None, optional
        When non-empty, a second parsing pass loads the corresponding raw-data
        files and attaches them as separate channels with this suffix appended
        to their names, e.g. ``"_rawdata"``.  Pass ``None`` or ``""`` to
        disable.  Defaults to ``"_rawdata"``.
    append_scan_parameter : dict, optional
        Extra scan parameter to append to every array after loading, in the
        form ``{"par_name": {"values": [v0, v1, ...]}}``.  The list must have
        one entry per scan step.
    verbose : int, optional
        Verbosity level.  ``0`` is silent; higher values print progress
        information.  Defaults to ``0``.
    perm_result_file : str or None, optional
        Permission string applied recursively to the result file after
        creation, e.g. ``"g+rw"`` (default).  ``None`` skips the chmod step.
    parse_version : {1, 2, 3}, optional
        Parser version to use (default ``3``):

        - ``1`` -- ``parseScanEcoV01``, the original eco-scan parser.
          Metadata scan via ``dask.compute(scheduler="processes")``;
          per-channel slices pre-computed and cached; keeps an HDF5 file
          handle per read; ``lazyEscArrays`` defaults to ``False`` (eager).
        - ``2`` -- ``parseScanEcoV02``, a clean rewrite. Each dataset wrapped
          in a picklable ``_H5ProxyV02`` fed straight to ``da.from_array``
          (file opened/closed per chunk read, no handle held open);
          metadata scan via ``ThreadPoolExecutor`` (h5py releases the GIL,
          so threads match process-pool speed without pickling overhead);
          simpler JSON cache (raw shape/dtype/chunk0, no slice lists,
          ``.parse_result_v02.json`` so caches never collide with v1).
        - ``3`` (default) -- ``parseScanEcoV03``, a drop-in v2 replacement
          that additionally prunes the per-file HDF5 discovery walk using
          "dead-end" knowledge shared between scanning workers (threads by
          default; ``use_processes=True`` switches to a process pool backed
          by a ``multiprocessing.Manager``) -- speeds up scans with many
          files of the same instrument configuration. ``lazyEscArrays``
          defaults to ``True`` here (dask-backed, built lazily), unlike
          v1/v2's ``False``.

        Pass ``1`` or ``2`` to fall back to an older parser if ``3`` ever
        misbehaves for a given beamline's file layout.
    wait_for_data_files : bool, optional
        If ``True``, block (per metadata file, polling every
        ``wait_poll_interval`` seconds) until every raw data file referenced
        by that run's scan-info JSON exists on disk — e.g. for a scan that
        is still acquiring — before parsing it for real and folding it into
        the returned :class:`~escape.storage.DataSet`.  Files already on
        disk may be parsed while waiting; only the final ``DataSet`` is
        gated on completeness.  Defaults to ``False`` (parse whatever is
        currently on disk, same as before this option existed).
    wait_poll_interval : float, optional
        Seconds between filesystem re-checks when ``wait_for_data_files`` is
        ``True``.  Defaults to ``10``.
    wait_timeout : float or None, optional
        Give up and raise ``TimeoutError`` after this many seconds of
        waiting per metadata file when ``wait_for_data_files`` is ``True``.
        ``None`` (default) waits indefinitely.
    merge_data_sources : bool, optional
        If ``True``, also fold in the run's CA/EPICS channel-monitor dump
        (path taken from ``scan_parameters["monitors"]`` in the scan-info
        JSON, same as how ``"status"`` is resolved just above -- this has
        already changed once in practice, from ``namespace_monitor.h5`` to
        ``namespace_monitor.ixp.h5``, so the filename is always read from
        the JSON rather than assumed, with the pre-``ixp`` name as a
        fallback only for older runs whose JSON predates this key) as
        :class:`~escape.storage.storage_timestamps.ArrayTimestamps`
        channels, loaded lazily via :meth:`DataSet.load_from_result_file`
        (cheap even for the tens-of-thousands-of-channels dumps this file
        typically is -- see that class's lazy-loading notes). Three sources
        are layered by name, each superseding the previous on a collision:
        run-start status (loaded first) < monitor channels < this run's own
        beam-synchronous :class:`~escape.Array` channels (highest priority,
        always win). A missing or unreadable monitor file is silently
        skipped, same as the pre-existing ``scan_monitor.pkl`` handling.
        Defaults to ``False`` (today's behavior: status and per-pulse Array
        data only, no monitor merge).

    Returns
    -------
    escape.storage.DataSet
        DataSet whose attributes are the parsed detector channels as escape
        :class:`~escape.Array` objects.  Every array carries a ``run_number``
        scan parameter (one value per step) that identifies the originating run,
        making steps from different runs distinguishable in
        :attr:`~escape.Scan.par_steps` even when their scan parameters are
        identical.

    Notes
    -----
    **Run-number tracking:** Each loaded array gains a ``run_number`` scan
    parameter automatically.  The value is extracted from the ``run{NNNN}``
    token found in the metadata file path (filename or parent directory).  When
    no such token is present the zero-based file index is used as a fallback.

    **Multi-run concatenation:** Arrays from successive runs are concatenated
    along the event axis using :func:`~escape.concatenate`.  All arrays across
    runs must share the same set of channel names; channels absent from one run
    but present in another will cause a concatenation error.

    Examples
    --------
    Load a single run by number (pgroup required):

    >>> ds = load_dataset_from_scan(run_number=42, pgroup="p12345")
    >>> ds.SARES11_SPEC125.scan.par_steps
    #    delay_ps  run_number  step_length
    # 0      -1.0          42          500
    # 1       0.0          42          500
    # ...

    Load multiple runs and distinguish them by ``run_number``:

    >>> ds = load_dataset_from_scan(run_numbers=[42, 43], pgroup="p12345")
    >>> ds.SARES11_SPEC125.scan.par_steps["run_number"].unique()
    array([42, 43])

    Load a specific metadata file directly:

    >>> ds = load_dataset_from_scan(
    ...     metadata_file="/sf/bernina/data/p12345/res/scan_info/run0042.json"
    ... )
    """
    # Normalise run_number / run_numbers: accept int or list, singular wins
    if run_number is not None and run_numbers is None:
        run_numbers = run_number if isinstance(run_number, (list, tuple)) else [run_number]
    elif isinstance(run_numbers, int):
        run_numbers = [run_numbers]

    metadata_files = interpret_raw_data_definition(
        metadata_file=metadata_file,
        run_numbers=run_numbers,
        pgroup=pgroup,
        exp_name=exp_name,
        instrument=instrument,
        search_path=search_path,
    )
    
    # create result gfile is desired
    if result_filename is None:
        pass
    elif result_filename == "auto":
        result_filename = Path(metadata_files[0]).stem
    else:
        result_filename = Path(result_filename).stem

    if (
        result_file is not None
        and isinstance(result_file, (str, Path))
        and result_type in _RESULT_FILE_TYPE_SUFFIXES
    ):
        # A path was given directly (as opposed to an already-open
        # h5py.File/zarr.Group, or letting results_directory/result_filename
        # build the path below) -- normalize its extension and open it
        # ourselves in a create-capable mode. Without this, the path went
        # straight through to DataSet(results_file=...) with its default
        # mode="r", which fails outright for a not-yet-existing result file
        # (e.g. zarr.open(..., mode="r") -> PathNotFoundError).
        result_filepath = _normalize_result_filepath(result_file, result_type)
        if clear_result_file and result_filepath.exists():
            if result_type == "h5":
                result_filepath.unlink()
            elif result_type == "zarr":
                shutil.rmtree(result_filepath)
        if result_type == "h5":
            result_file = h5py.File(result_filepath, "a")
        elif result_type == "zarr":
            result_file = zarr.open(result_filepath, mode="a")
        if perm_result_file:
            try:
                oschmod.set_mode_recursive(result_filepath, perm_result_file)
            except Exception:
                print(f"Warning: could not set permissions {perm_result_file}!")

    if result_filename and (not result_file):
        if result_type == "h5":
            result_filepath = Path(results_directory) / Path(
                result_filename + ".esc" + ".h5"
            )
            if clear_result_file and result_filepath.exists():
                result_filepath.unlink()
            result_file = h5py.File(result_filepath, "a")
        elif result_type == "zarr":
            print("taking zarr format")
            result_filepath = Path(results_directory) / Path(
                result_filename + ".esc" + ".zarr"
            )
            if clear_result_file and result_filepath.exists():
                shutil.rmtree(result_filepath)
            result_file = zarr.open(result_filepath)
        


        print(f"Automatic creation of result file: {result_filepath.as_posix()} .")
        if perm_result_file:
            try:
                oschmod.set_mode_recursive(result_filepath,perm_result_file)
            except:
                print(f'Warning: could not set permissions {perm_result_file:s}!')

    _parser = {1: parseScanEcoV01, 2: parseScanEcoV02, 3: parseScanEcoV03}[parse_version]
    if lazy_esc_array_parsing and parse_version != 3:
        raise ValueError("lazy_esc_array_parsing requires parse_version=3.")
    _lazy_kwargs = {"lazy_esc_array_parsing": True} if lazy_esc_array_parsing else {}

    if load_result_only:
        ds = DataSet.load_from_result_file(result_filepath)
    else:
        d = {}
        s_collection = []

        for file_idx, metadata_file in enumerate(metadata_files):
            if wait_for_data_files:
                _wait_for_data_files_on_disk(
                    metadata_file,
                    search_paths=search_paths,
                    exclude_from_files=exclude_from_files,
                    poll_interval=wait_poll_interval,
                    timeout=wait_timeout,
                    verbose=True,
                )

            effective_checknstore_parsing_result = _resolve_auto_parsing_cache(
                checknstore_parsing_result, metadata_file, parse_version
            )
            if (
                checknstore_parsing_result == "auto"
                and effective_checknstore_parsing_result == "same_directory"
                and verbose
            ):
                print(f"Using aux/ parse-result cache for {metadata_file}")

            td, s = _parser(
                metadata_file,
                search_paths=search_paths,
                memlimit_MB=memlimit_MB,
                createEscArrays=createEscArrays,
                lazyEscArrays=lazyEscArrays,
                **_lazy_kwargs,
                exclude_from_files=exclude_from_files,
                checknstore_parsing_result=effective_checknstore_parsing_result,
                clear_parsing_result=clear_parsing_result,
                return_json_info=True,
                step_selection=step_selection,
                verbose=verbose,
            )

            if raw_data_suffix:
                dum, scan_info_filepath = readScanEcoJson_v01(
                    metadata_file, exclude_from_files=exclude_from_files
                )
                if Path(metadata_file).parent.stem == "aux":
                    run_root_directory = Path(metadata_file).parent.parent
                else:
                    run_root_directory = None

                rscf = []
                for stepno, step in enumerate(s["scan_files"]):
                    rscf.append([])
                    for tf in step:
                        tp = (
                            scan_info_filepath.parent.parent
                            / Path("raw_data")
                            / Path(tf).name
                        )
                        if tp.exists():
                            rscf[stepno].append(tp.as_posix())

                sr = copy(s)

                # print(rscf)
                sr["scan_files"] = rscf
                scan_info_filepath = scan_info_filepath.parent / Path(
                    scan_info_filepath.stem + "_raw_data" + scan_info_filepath.suffix
                )
                # print(scan_info_filepath)
                trd, sr = _parser(
                    scan_info=sr,
                    scan_info_filepath=scan_info_filepath,
                    search_paths=search_paths,
                    memlimit_MB=memlimit_MB,
                    createEscArrays=createEscArrays,
                    lazyEscArrays=lazyEscArrays,
                    **_lazy_kwargs,
                    exclude_from_files=exclude_from_files,
                    checknstore_parsing_result=_resolve_auto_parsing_cache(
                        checknstore_parsing_result, scan_info_filepath, parse_version
                    ),
                    clear_parsing_result=clear_parsing_result,
                    return_json_info=True,
                    step_selection=step_selection,
                    run_root_directory=run_root_directory,
                    verbose=verbose,
                )
                for nm, ar in trd.items():
                    td[nm + raw_data_suffix] = ar

            if load_dap_data:
                print("Loading DAP data...")
                pdata, fnames_parsed = parse_dap(
                    Path(metadata_file).parent / Path("../data")
                )
                for tdet, step_data in pdata.items():  # over different detectors
                    index = []
                    ddata = {}
                    steps = []
                    step_lengths = []
                    for stepno, step_data in step_data.items():  # over steps
                        steps.append(stepno)
                        step_lengths.append(len(step_data["index"]))
                        index.append(step_data["index"])
                        for tsdno, tsd in enumerate(
                            step_data["data"]
                        ):  # over data columns
                            if tsdno not in ddata.keys():
                                ddata[tsdno] = []
                            ddata[tsdno].append(tsd)
                    for colno in range(len(ddata.keys())):
                        colname = f"{tdet}_dap_col{colno}"
                        td[colname] = Array(
                            data=np.concatenate(ddata[colno], axis=0),
                            index=np.concatenate(index, axis=0),
                            step_lengths=step_lengths,
                            parameter={"step_number": {"values": steps}},
                            name=colname,
                        )
                print("Finished loading DAP data.")

            s_collection.append(s)
            file_alias_mappings = {}
            if "namespace_aliases" in s["scan_parameters"].keys():
                talias_mappings = {
                    ta["channel"]: ta["alias"]
                    for ta in s["scan_parameters"]["namespace_aliases"]
                    if ta["channeltype"] in ["BS", "BSCAM", "JF"]
                }
                file_alias_mappings.update(talias_mappings)
                
            if "aliases" in s["scan_parameters"].keys():
                with open(
                    Path(metadata_file).parent
                    / Path("../" + s["scan_parameters"]["aliases"]),
                    "r",
                ) as fh:
                    aliases_all = json.load(fh)
                talias_mappings = {
                    ta["channel"]: ta["alias"]
                    for ta in aliases_all
                    if ta["channeltype"] in ["BS", "BSCAM", "JF"]
                }
                file_alias_mappings.update(talias_mappings)
            
            file_alias_mappings.update(alias_mappings)
            alias_mappings = file_alias_mappings

            if verbose:
                for tmpkey, tmpval in alias_mappings.items():
                    print(tmpval, "      ", tmpkey)

            run_no = _extract_run_number(metadata_file)
            if run_no is None:
                run_no = file_idx
            for nm, ar in list(td.items()):
                td[nm] = _with_scan_parameter(
                    ar,
                    lambda a, run_no=run_no: {
                        "run_number": {"values": [run_no] * len(a.scan)}
                    },
                    defer=lazy_esc_array_parsing,
                )

            for nm, ar in td.items():
                if not (nm in d.keys()):
                    d[nm] = ar
                else:
                    d[nm] = concatenate([d[nm], ar])
        if append_scan_parameter:
            for nm, ar in list(d.items()):
                try:
                    d[nm] = _with_scan_parameter(
                        ar, lambda a: append_scan_parameter, defer=lazy_esc_array_parsing
                    )
                except Exception as e:
                    print(str(e))

        ds = DataSet(
            d, name=name, alias_mappings=alias_mappings, results_file=result_file
        )

        ds._metafile_parse_results = s_collection
        ds._alias_mappings = alias_mappings

        if merge_data_sources:
            # Names already claimed by this run's own beam-synchronous Array
            # data -- captured now, before status/monitor loading adds more
            # keys, so those two can freely overwrite each other but never
            # this set (Array data has final say, see merge_data_sources'
            # docstring for the full precedence).
            bs_names = set(ds.datasets.keys())

        try:
            if type(s["scan_parameters"]["status"]) is str:
                with open(
                    Path(metadata_file).parent
                    / Path("../" + s["scan_parameters"]["status"]),
                    "r",
                ) as fh:
                    r = json.load(fh)

                if store_status:
                    for k in r.keys():
                        ds.append(r[k]["status"], name=k, as_pickle=True)
                else:
                    for k in r.keys():
                        ds.__dict__[k] = StructureGroup()
                        dict2structure(r[k]["status"], base=ds.__dict__[k])

                print("found and loaded status")
                # if result_type == "zarr":
                #     for k in r.keys():
                #         ds.results_file.require_group(k)
                #         ds.results_file[k] = pickle.dumps(r[k]["status"])
                #         ds.results_file[k].attrs["esc_type"] = "pickled"
                # elif result_type == "h5":
                #     for k in r.keys():
                #         ds.results_file.require_group(k)
                #         dictToH5Group(r[k]["status"], ds[k])
                #         ds.results_file[k].attrs["esc_type"] = "datastorage"

            else:
                pass
        except:
            traceback.print_exc()
            print("No status in dataset found.")
            pass

        if merge_data_sources:
            try:
                # Read the monitor file's path from scan_parameters, same as
                # "status" just above -- its filename has already changed
                # once (namespace_monitor.h5 -> namespace_monitor.ixp.h5,
                # observed live on run0077), and the JSON is exactly what
                # tells us the current name instead of us having to guess
                # or hardcode it. Fall back to the pre-ixp default only for
                # older runs whose JSON predates this key entirely.
                monitor_rel_path = s["scan_parameters"].get(
                    "monitors", "aux/namespace_monitor.h5"
                )
                monitor_path = Path(metadata_file).parent / Path(
                    "../" + monitor_rel_path
                )
                # Not DataSet.load_from_result_file(): it requires an
                # ".esc.<ext>" suffix, which this file (produced by the
                # namespace-monitor daemon, not by escape itself) doesn't
                # have. Opening the h5py.File ourselves and handing DataSet
                # the live handle skips that suffix check entirely (see
                # filespec_to_file's isinstance(file, h5py.File) branch).
                monitor_h5 = h5py.File(monitor_path, "r")
                monitor_ds = DataSet(results_file=monitor_h5, mode="r")
                n_added = 0
                for mname, mval in monitor_ds.datasets.items():
                    if mname in bs_names:
                        continue
                    ds.datasets[mname] = mval
                    dict2structure({mname: mval}, base=ds)
                    n_added += 1
                print(
                    f"merged {n_added} channel(s) from {monitor_rel_path} "
                    f"({len(monitor_ds.datasets) - n_added} shadowed by "
                    f"this run's own Array data)"
                )
            except Exception as exc:
                print(f"No namespace monitor file found or failed to merge it: {exc}")

        # monitor data hack
        try:
            with open(
                Path(metadata_file).parent / Path("../aux/scan_monitor.pkl"),
                "rb",
            ) as fh:
                monitored_data = pickle.load(fh)
            # ds.mon_dat = monitored_data

            ds.monitored_data = StructureGroup()
            dict2structure(
                {
                    name: MonitorData(datadict, name=name)
                    for name, datadict in monitored_data.items()
                },
                base=ds.monitored_data,
            )
            # ds.monitored_data = {
            #     name: MonitorData(datadict, name=name)
            #     for name, datadict in monitored_data.items()
            # }

        except:
            # traceback.print_exc()
            print("No monitor data in dataset found.")
            pass

            # Array()
            # ds.pdata = pdata

    return ds


class MonitorData:
    def __init__(self, datadict, name=None):
        self.name = name
        for tn, td in datadict.items():
            if hasattr(self, "channel"):
                raise Exception("Only one name value pair allowed in dictionary!")
            self.channel = tn
            for tli in td:
                for tdn, tdv in tli.items():
                    if not hasattr(self, tdn):
                        self.__dict__[tdn] = []
                    self.__dict__[tdn].append(tdv)


def parse_dap(fdir, fnames_parsed=[], N_acs_digits=4):
    p = Path(fdir)
    fnames = [tf.name for tf in p.glob("acq" + "?" * N_acs_digits + "*.dap")]
    dets = list(np.unique([tn.split(".")[1] for tn in fnames]))
    data = {}
    fnames_parsed_new = []
    for det in dets:
        data[det] = {}
        tfnames = sorted([tn for tn in fnames if det in tn])
        for tfname in tfnames:
            if tfname in fnames_parsed:
                continue
            step = int(tfname[3 : (3 + N_acs_digits)])
            tmp = np.genfromtxt(p / Path(tfname), dtype=None, unpack=True)
            if tmp:
                data[det][step] = dict(
                    index=np.atleast_1d(tmp[0]), data=np.atleast_2d(tmp)[1:]
                )
            fnames_parsed_new.append(tfname)
    return data, fnames_parsed_new
