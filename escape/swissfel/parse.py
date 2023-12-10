from asyncio import run
import pickle
import shutil
from unicodedata import name

import numpy as np
from escape.storage.storage import concatenate, Array, Scan
from ..parse.swissfel import readScanEcoJson_v01, parseScanEco_v01
from .cluster import parseScanEcoV01
from pathlib import Path
import json
import pathlib
import warnings
import logging
import escape
from copy import deepcopy as copy

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
        "{instrument:s}/data/{pgroup:s}/res/scan_info/run{run_number:04d}*.json",
    ],
    verbose=True,
):
    # format search paths
    if isinstance(search_path, str):
        search_path = [search_path]

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
        "{instrument:s}/data/{pgroup:s}/res/scan_info/run{run_number:04d}*.json",
    ],
    search_paths=["./", "./scan_data/", "../scan_data"],
    memlimit_MB=100,
    createEscArrays=True,
    lazyEscArrays=False,
    exclude_from_files=[],
    checknstore_parsing_result=False,
    clear_parsing_result=False,
    analyze_namespace_info=False,
    name="delme",
    alias_mappings={},
    step_selection=slice(None),
    load_dap_data=False,
    raw_data_suffix="_rawdata",
    verbose=0,
):
    metadata_files = interpret_raw_data_definition(
        metadata_file=metadata_file,
        run_numbers=run_numbers,
        pgroup=pgroup,
        exp_name=exp_name,
        instrument=instrument,
        search_path=search_path,
    )

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

    if load_result_only:
        ds = DataSet.load_from_result_file(result_filepath)
    else:
        d = {}
        s_collection = []

        for metadata_file in metadata_files:
            td, s = parse_scan(
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
                trd, sr = parse_scan(
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
                    verbose=verbose,
                )
                for nm, ar in trd.items():
                    td[nm + raw_data_suffix] = ar

            if load_dap_data:
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

            s_collection.append(s)
            if "namespace_aliases" in s["scan_parameters"].keys():
                talias_mappings = {
                    ta["channel"]: ta["alias"]
                    for ta in s["scan_parameters"]["namespace_aliases"]
                    if ta["channeltype"] in ["BS", "BSCAM", "JF"]
                }
                alias_mappings.update(talias_mappings)
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
                alias_mappings.update(talias_mappings)

            if verbose:
                for tmpkey, tmpval in alias_mappings.items():
                    print(tmpval, "      ", tmpkey)

            for nm, ar in td.items():
                if not (nm in d.keys()):
                    d[nm] = ar
                else:
                    d[nm] = concatenate([d[nm], ar])

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
                    for k in r.keys():
                        ds.__dict__[k] = StructureGroup()
                        dict2structure(r[k]["status"], base=ds.__dict__[k])
                    print("found and loaded status")

                if store_status:
                    if result_type == "zarr":
                        for k in r.keys():
                            ds.results_file.require_group(k)
                            ds.results_file[k] = pickle.dumps(r[k]["status"])
                            ds.results_file[k].attrs["esc_type"] = "pickled"
                    elif result_type == "h5":
                        for k in r.keys():
                            ds.results_file.require_group(k)
                            dictToH5Group(r[k]["status"], ds[k])
                            ds.results_file[k].attrs["esc_type"] = "datastorage"

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
            traceback.print_exc()
            print("No monitor data  in dataset found.")
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
