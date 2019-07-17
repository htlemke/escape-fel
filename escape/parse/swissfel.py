import json
import pathlib
from pathlib import Path
from . import utilities
import h5py
from dask import array as da
from .. import Array, Scan
import numpy as np
from copy import copy
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def readScanEcoJson_v01(file_name_json,exclude_from_files=[]):
    p = pathlib.Path(file_name_json)
    assert p.is_file(), "Input string does not describe a valid file path."
    with p.open(mode="r") as f:
        s = json.load(f)
    assert len(s["scan_files"]) == len(
        s["scan_values"]
    ), "number of files and scan values don't match in {}".format(file_name_json)
    assert len(s["scan_files"]) == len(
        s["scan_readbacks"]
    ), "number of files and scan readbacks don't match in {}".format(file_name_json)
    for step in s['scan_files']:
        for sstr in exclude_from_files:
            kill = []
            for i,tf in enumerate(step):
                if sstr in tf:
                    kill.append(i)
            for k in kill[-1::-1]:
                step.pop(k)

    return s, p

def parseSFh5File_v01(
    files,
    memlimit_0D_MB=5,
    memlimit_mD_MB=132,
    createEscArrays=True,
):
    """Data parser assuming the standard swissfel h5 format for raw data"""
    if (type(files) is str) or (not np.iterable(files)):
        files = [files]
    datasets_all = []
    for fina in files:
        fina = Path(fina)
        fh = h5py.File(fina.resolve(), mode="r")
        datasets = utilities.findItemnamesGroups(fh, ["data", "pulse_id"])
        logger.info("Successfully parsed file %s" % fina.resolve())
        datasets_all.append(datasets)

    names = set()
    dstores = {}

    for datasets in datasets_all:
        tnames = set(datasets.keys())
        newnames = tnames.difference(names)
        oldnames = names.intersection(tnames)
        for name in newnames:
            if datasets[name][0].size == 0:
                logger.debug("Found empty dataset in {}".format(name))
            else:
                size_data = (
                    np.dtype(datasets[name][0].dtype).itemsize
                    * datasets[name][0].size
                    / 1024 ** 2
                )
                size_element = (
                    np.dtype(datasets[name][0].dtype).itemsize
                    * np.prod(datasets[name][0].shape[1:])
                    / 1024 ** 2
                )
                if datasets[name][0].chunks:
                    chunk_size = list(datasets[name][0].chunks)
                else:
                    chunk_size = list(datasets[name][0].shape)
                if chunk_size[0] == 1:
                    chunk_size[0] = int(memlimit_mD_MB // size_element)
                dstores[name] = {}
                dstores[name]["data"] = []
                dstores[name]["data"].append(datasets[name][0])
                dstores[name]["data_chunks"] = chunk_size
                dstores[name]["eventIds"] = []
                dstores[name]["eventIds"].append(datasets[name][1])
                dstores[name]["stepLengths"] = []
                dstores[name]["stepLengths"].append(len(datasets[name][0]))
                names.add(name)
        for name in oldnames:
            if datasets[name][0].size == 0:
                logger.debug("Found empty dataset in {}".format(name))
            else:
                # dirty hack for inconsitency in writer
                dstores[name]["data"].append(datasets[name][0])
                dstores[name]["eventIds"].append(datasets[name][1])
                dstores[name]["stepLengths"].append(len(datasets[name][0]))
    if createEscArrays:
        escArrays = {}
        containers = {}
        for name, dat in dstores.items():
            containers[name] = LazyContainer(dat)
            escArrays[name] = Array(
                containers[name].get_data,
                eventIds=containers[name].get_eventIds,
                stepLengths=dat["stepLengths"],
                scan=None,
            )
        return escArrays
    else:
        return dstores

def parseScanEco_v01(
    file_name_json=None,
    search_paths=["./", "./scan_data/", "../scan_data"],
    memlimit_0D_MB=5,
    memlimit_mD_MB=10,
    createEscArrays=True,
    scan_info = None,
    scan_info_filepath=None,
    exclude_from_files = [],
):
    
    if file_name_json:
        """Data parser assuming eco-written files from pilot phase 1"""
        s, scan_info_filepath = readScanEcoJson_v01(file_name_json,exclude_from_files=exclude_from_files)
    else:
        s = scan_info
    lastpath = None
    searchpaths = None

    datasets_scan = []
    for files in tqdm(s["scan_files"], desc='Scan steps'):
        datasets = {}
        for f in tqdm(files, desc='Step files'):
            fp = pathlib.Path(f)
            fn = pathlib.Path(fp.name)
            if not searchpaths:
                searchpaths = [fp.parent] + [
                    scan_info_filepath.parent / pathlib.Path(tp.format(fp.parent.name))
                    for tp in search_paths
                ]
            for path in searchpaths:
                file_path = path / fn
                if file_path.is_file():
                    if not lastpath:
                        lastpath = path
                        searchpaths.insert(0, path)
                    break
            # assert file_path.is_file(), 'Could not find file {} '.format(fn)
            try:
                fh = h5py.File(file_path.resolve(), mode="r")
                datasets.update(utilities.findItemnamesGroups(fh, ["data", "pulse_id"]))
                logger.info("Successfully parsed file %s" % file_path.resolve())
            except:
                logger.warning(f"WARNING: could not read {file_path.absolute().as_posix()}.")
        datasets_scan.append(datasets)

    names = set()
    dstores = {}

    for stepNo, (datasets, scan_values, scan_readbacks, scan_step_info) in enumerate(
        zip(datasets_scan, s["scan_values"], s["scan_readbacks"], s["scan_step_info"])
    ):
        tnames = set(datasets.keys())
        newnames = tnames.difference(names)
        oldnames = names.intersection(tnames)
        for name in newnames:
            if datasets[name][0].size == 0:
                logger.debug("Found empty dataset in {} in cycle {}".format(name, stepNo))
            else:
                size_data = (
                    np.dtype(datasets[name][0].dtype).itemsize
                    * datasets[name][0].size
                    / 1024 ** 2
                )
                size_element = (
                    np.dtype(datasets[name][0].dtype).itemsize
                    * np.prod(datasets[name][0].shape[1:])
                    / 1024 ** 2
                )
                if datasets[name][0].chunks:
                    chunk_size = list(datasets[name][0].chunks)
                else:
                    chunk_size = list(datasets[name][0].shape)
                if chunk_size[0] == 1:
                    chunk_size[0] = int(memlimit_mD_MB // size_element)
                dstores[name] = {}
                dstores[name]["scan"] = Scan(
                    parameter_names=[str(ts) for ts in s["scan_parameters"]["name"]]
                    + [f"{tn}_readback" for tn in s["scan_parameters"]["name"]],
                    parameter_attrs={
                        tn: {"Id": ti}
                        for tn, ti in zip(
                            s["scan_parameters"]["name"], s["scan_parameters"]["Id"]
                        )
                    },
                )
                # dirty hack for inconsitency in writer
                if (
                    len(scan_readbacks)
                    > len(dstores[name]["scan"]._parameter_names) / 2
                ):
                    scan_readbacks = scan_readbacks[
                        : int(len(dstores[name]["scan"]._parameter_names) / 2)
                    ]
                dstores[name]["scan"]._append(
                    copy(scan_values) + copy(scan_readbacks),
                    scan_step_info=copy(scan_step_info),
                )
                dstores[name]["data"] = []
                dstores[name]["data"].append(datasets[name][0])
                dstores[name]["data_chunks"] = chunk_size
                dstores[name]["eventIds"] = []
                dstores[name]["eventIds"].append(datasets[name][1])
                dstores[name]["stepLengths"] = []
                dstores[name]["stepLengths"].append(len(datasets[name][0]))
                names.add(name)
        for name in oldnames:
            if datasets[name][0].size == 0:
                logger.debug("Found empty dataset in {} in cycle {}".format(name, stepNo))
            else:
                # dirty hack for inconsitency in writer
                if (
                    len(scan_readbacks)
                    > len(dstores[name]["scan"]._parameter_names) / 2
                ):
                    scan_readbacks = scan_readbacks[
                        : int(len(dstores[name]["scan"]._parameter_names) / 2)
                    ]
                dstores[name]["scan"]._append(
                    copy(scan_values) + copy(scan_readbacks),
                    scan_step_info=copy(scan_step_info),
                )
                dstores[name]["data"].append(datasets[name][0])
                dstores[name]["eventIds"].append(datasets[name][1])
                dstores[name]["stepLengths"].append(len(datasets[name][0]))
    if createEscArrays:
        escArrays = {}
        containers = {}
        for name, dat in dstores.items():
            containers[name] = LazyContainer(dat)
            escArrays[name] = Array(
                containers[name].get_data,
                eventIds=containers[name].get_eventIds,
                stepLengths=dat["stepLengths"],
                scan=dat["scan"],
            )
        return escArrays
    else:
        return dstores


class LazyContainer:
    def __init__(self, dat):
        self.dat = dat

    def get_data(self):
        return da.concatenate(
            [
                da.from_array(td, chunks=self.dat["data_chunks"])
                for td in self.dat["data"]
            ]
        )

    def get_eventIds(self):
        return np.concatenate([td[...].ravel() for td in self.dat["eventIds"]])

    
