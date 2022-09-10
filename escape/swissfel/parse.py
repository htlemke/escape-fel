from asyncio import run

from escape.storage.storage import concatenate
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
    import bitshuffle.h5
except:
    print("Could not import bitshuffle.h5!")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm
import h5py
import zarr

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


class StructureGroup:
    pass


def dict2structure(t, base=None):
    if not base:
        base = StructureGroup()
    for tt, tv in t.items():
        p = tt.split(".")
        tbase = base
        for tp in p[:-1]:
            if tp in tbase.__dict__.keys():
                if not isinstance(tbase.__dict__[tp], StructureGroup):
                    tbase.__dict__[tp] = StructureGroup()
            else:
                tbase.__dict__[tp] = StructureGroup()

            tbase = tbase.__dict__[tp]
        # try:
        if hasattr(tbase, p[-1]):
            if not isinstance(tbase.__dict__[p[-1]], StructureGroup):
                tbase.__dict__[p[-1]] = tv
        else:
            tbase.__dict__[p[-1]] = tv
            # except:
            ...
    return base


def interpret_raw_data_definition(
    metadata_file=None,
    run_numbers=None,
    pgroup=None,
    instrument="bernina",
    search_path="{instrument:s}/data/{pgroup:s}/raw/run{run_number:04d}/aux/scan_info*.json",
):
    if metadata_file:
        return [metadata_file]
    if run_numbers and pgroup and instrument:
        metadata_files = []
        for run_number in run_numbers:
            tfiles = list(
                Path("/sf").glob(
                    search_path.format(
                        instrument=instrument, pgroup=pgroup, run_number=run_number
                    )
                )
            )
            if len(tfiles) > 1:
                print(
                    "WARNING:found more than one file matching raw data definition! Taking fist one."
                )
            tfile = tfiles[0]
            metadata_files.append(tfile.as_posix())
        return metadata_files


def load_dataset_from_scan(
    raw_data_definition,
    result_type="zarr",
    result_file=None,
    results_directory="./",
    search_paths=["./", "./scan_data/", "../scan_data"],
    memlimit_MB=100,
    createEscArrays=True,
    exclude_from_files=[],
    checknstore_parsing_result=False,
    clear_parsing_result=False,
    analyze_namespace_info=False,
    name="delme",
):
    metadata_files = interpret_raw_data_definition(**raw_data_definition)
    alias_mappings = None

    d = {}

    for metadata_file in metadata_files:
        td, s = parse_scan(
            metadata_file,
            search_paths=search_paths,
            memlimit_MB=memlimit_MB,
            createEscArrays=createEscArrays,
            exclude_from_files=exclude_from_files,
            checknstore_parsing_result=checknstore_parsing_result,
            clear_parsing_result=clear_parsing_result,
            return_json_info=True,
        )
        if (not alias_mappings) and (
            "namespace_aliases" in s["scan_parameters"].keys()
        ):
            alias_mappings = {
                ta["channel"]: ta["alias"]
                for ta in s["scan_parameters"]["namespace_aliases"]
                if ta["channeltype"] in ["BS", "BSCAM", "JF"]
            }
        if (not alias_mappings) and (
            "aliases" in s["scan_parameters"].keys()
        ):
            with open( Path(metadata_file).parent / Path('../'+s["scan_parameters"]['aliases']),'r') as fh:
                aliases_all = json.load(fh)
            alias_mappings = {
                ta["channel"]: ta["alias"]
                for ta in aliases_all
                if ta["channeltype"] in ["BS", "BSCAM", "JF"]
            }

        for nm, ar in td.items():
            if not (nm in d.keys()):
                d[nm] = ar
            else:
                d[nm] = concatenate([d[nm], ar])

        # print(alias_mappings)

    if not result_file:
        if result_type == "h5":
            result_filepath = Path(results_directory) / Path(
                Path(metadata_files[0]).stem + ".esc" + ".h5"
            )
            result_file = h5py.File(result_filepath, "a")
        elif result_type == "zarr":
            print("taking zarr format")
            result_filepath = Path(results_directory) / Path(
                Path(metadata_files[0]).stem + ".esc" + ".zarr"
            )
            result_file = zarr.open(result_filepath)

    #     namespaceobjects = {}
    #     for ch,ea in escArrays.items():
    #         if ch in channel_dict.keys():
    #             namespaceobjects[channel_dict[ch]] = ea
    # if 'namespace_status' in s['scan_parameters'].keys():
    #     namespace_status  = s['scan_parameters']['namespace_status']

    ds = DataSet(d, name=name, alias_mappings=alias_mappings, results_file=result_file)
    # try:
        # for name, data in s["scan_parameters"]["namespace_status"]["settings"].items():
        #     ds.append(data, name)
        
    if type(s["scan_parameters"]["status"]) is str:
        with open( Path(metadata_file).parent / Path('../'+s["scan_parameters"]['status']),'r') as fh:
            ds.status = json.load(fh)
            print('done')

    else:

        pass
    

    # except:
    #     print("Did not succeed to append namespace status!")
    return ds


class DataSet:
    def __init__(
        self,
        raw_datasets: dict,
        alias_mappings: dict = None,
        results_file=None,
        name=None,
    ):
        self.data_raw = raw_datasets
        self.datasets = {}

        if results_file is not None:
            self.results_file = results_file
        else:
            self.results_file = None

        if alias_mappings:
            for idname in self.data_raw.keys():
                # print(idname)
                if idname in alias_mappings.keys():
                    self.append(self.data_raw[idname], name=alias_mappings[idname])

        self.name = name

        # self.file_output = zarr.open(
        #     (self.dir_output / Path(f"run{runno:04d}.zarr")).as_posix(), "a"
        # )
        # if clear_output:
        #     self.file_output.clear()
        # self.runno = runno
        # self.data = load_runno(runno)

    def append(self, data, name=None):
        self.datasets[name] = data
        if isinstance(data, escape.Array):
            data.name = name
            self.datasets[name].set_h5_storage(self.results_file, name)

        dict2structure({name: data}, base=self)

        # try:
        #     if type(channel) is str:
        #         self.__dict__[name] = self.data[channel]
        #     else:
        #         self.__dict__[name] = channel
        #     self.__dict__[name].set_h5_storage(self.file_output, name)
        # except:
        #     print(
        #         f"Detector {name}, channel {channel} was not found in run {self.runno}."
        #     )


# class DataReduced:
#     def __init__(self, runno, dir_output=dir_output):
#         self.dir_output = dir_output
#         self.file_output = zarr.open(
#             (self.dir_output / Path(f"run{runno:04d}.zarr")).as_posix(), "r"
#         )

#         for name in self.file_output.keys():
#             try:
#                 self.__dict__[name] = escape.Array.load_from_h5(self.file_output, name)
#             except:
#                 pass


# def readScanEcoJson_v02(file_name_json, exclude_from_files=[]):
#     p = pathlib.Path(file_name_json)
#     assert p.is_file(), "Input string does not describe a valid file path."
#     with p.open(mode="r") as f:
#         s = json.load(f)
#     assert len(s["scan_files"]) == len(
#         s["scan_values"]
#     ), "number of files and scan values don't match in {}".format(file_name_json)
#     assert len(s["scan_files"]) == len(
#         s["scan_readbacks"]
#     ), "number of files and scan readbacks don't match in {}".format(file_name_json)
#     for step in s["scan_files"]:
#         for sstr in exclude_from_files:
#             kill = []
#             for i, tf in enumerate(step):
#                 if sstr in tf:
#                     kill.append(i)
#             for k in kill[-1::-1]:
#                 step.pop(k)
#     return s, p
