from asyncio import run
import pickle
import re
import shutil
from unicodedata import name

import numpy as np
from escape.storage.storage import concatenate, Array, Scan
from ..parse.swissfel import readScanEcoJson_v01, parseScanEco_v01
from .cluster import parseScanEcoV01, parseScanEcoV02, parseScanEcoV03
from pathlib import Path
import json
import pathlib
import warnings
import logging
import escape
from copy import deepcopy as copy
import oschmod

try:
    from datastorage.datastorage import dictToH5Group
except:
    print("issue with datastorage import!")

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
    exclude_from_files=[],
    checknstore_parsing_result=False,
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
    exclude_from_files : list of str, optional
        Channel names or file patterns to skip during parsing.
    checknstore_parsing_result : bool, optional
        Cache the intermediate parsing result on disk and reuse it on
        subsequent calls with the same inputs.  Defaults to ``False``.
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
        Parser version to use.  ``1`` selects the original SwissFEL eco-scan
        parser; ``2`` selects the v2 rewrite (thread-based metadata scan);
        ``3`` (default) additionally prunes the per-file HDF5 discovery walk
        using dead-end knowledge shared between the scanning workers, which
        speeds up parsing scans with many files of the same instrument
        configuration.  Pass ``1`` or ``2`` to fall back to the older
        parsers if ``3`` ever misbehaves for a given beamline's file layout.

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

    if load_result_only:
        ds = DataSet.load_from_result_file(result_filepath)
    else:
        d = {}
        s_collection = []

        for file_idx, metadata_file in enumerate(metadata_files):
            td, s = _parser(
                metadata_file,
                search_paths=search_paths,
                memlimit_MB=memlimit_MB,
                createEscArrays=createEscArrays,
                lazyEscArrays=lazyEscArrays,
                exclude_from_files=exclude_from_files,
                checknstore_parsing_result=checknstore_parsing_result,
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
                    exclude_from_files=exclude_from_files,
                    checknstore_parsing_result=checknstore_parsing_result,
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
                        td[f"{tdet}_dap_col{colno}"] = Array(
                            data=np.concatenate(ddata[colno], axis=0),
                            index=np.concatenate(index, axis=0),
                            step_lengths=step_lengths,
                            parameter={"step_number": {"values": steps}},
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
            for ar in td.values():
                ar.scan.append_parameter(
                    {"run_number": {"values": [run_no] * len(ar.scan)}}
                )

            for nm, ar in td.items():
                if not (nm in d.keys()):
                    d[nm] = ar
                else:
                    d[nm] = concatenate([d[nm], ar])
        if append_scan_parameter:
            for nm, ar in d.items():
                try:
                    ar.scan.append_parameter(append_scan_parameter)
                except Exception as e:
                    print(str(e))

        ds = DataSet(
            d, name=name, alias_mappings=alias_mappings, results_file=result_file
        )

        ds._metafile_parse_results = s_collection
        ds._alias_mappings = alias_mappings

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
