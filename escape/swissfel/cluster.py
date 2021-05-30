import json
import pathlib
from pathlib import Path

from dask.array.routines import shape
from ..parse import utilities
import h5py
from dask import array as da
from dask import bag as db
from dask.diagnostics import ProgressBar
from dask import delayed
import dask

from .. import Array, Scan
import numpy as np
from copy import deepcopy as copy
import logging
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm
from threading import Thread
from time import sleep
import bitshuffle.h5
from dask_jobqueue import SLURMCluster
from distributed import Client

logger = logging.getLogger(__name__)

# cluter helpers


class SwissFelCluster:
    def __init__(self, cores=8, memory="24 GB", workers=5):
        self.cluster = SLURMCluster(cores=cores, memory=memory)
        self.client = Client(self.cluster)

    def _repr_html_(self):
        return self.client._repr_html_()


# parsing stuff


@delayed
def parse_bs_h5_file(fina, memlimit_MB=100):
    """Data parser assuming the standard swissfel h5 format for raw data"""
    # if (type(files) is str) or (not np.iterable(files)):
    #     files = [files]
    fina = Path(fina)
    with h5py.File(fina.resolve(), mode="r") as fh:
        datasets = utilities.findItemnamesGroups(fh, ["data", "pulse_id"])
        logger.info("Successfully parsed file %s" % fina.resolve())
        dstores = {}
        for name, (ds_data, ds_index) in datasets.items():
            print(
                f"Shapes data and index datasets found:   {ds_data.shape}, {ds_index.shape}"
            )

            if ds_data.size == 0:
                logger.debug("Found empty dataset in {}".format(name))
                continue
            # data first
            dtype = np.dtype(ds_data.dtype)
            size_element = (
                np.dtype(ds_data.dtype).itemsize
                * np.prod(ds_data.shape[1:])
                / 1024 ** 2
            )
            chunk_length = int(memlimit_MB // size_element)
            dset_size = ds_data.shape
            chunk_shapes = []
            slices = []
            for chunk_start in range(0, dset_size[0], chunk_length):
                slice_0dim = [
                    chunk_start,
                    min(chunk_start + chunk_length, dset_size[0]),
                ]
                chunk_shape = list(dset_size)
                chunk_shape[0] = slice_0dim[1] - slice_0dim[0]
                slices.append(slice_0dim)
                chunk_shapes.append(chunk_shape)

            dstores[name] = {
                "file_path": fina.resolve(),
                "data_dsp": ds_data.name,
                "data_shape": ds_data.shape,
                "data_dtype": dtype,
                "data_chunks": {"slices": slices, "shapes": chunk_shapes},
                "index_dsp": ds_index.name,
                "index_dtype": ds_index.dtype,
                "index_shape": ds_index.shape,
            }
            # dstores[name]["stepLengths"] = []
            # dstores[name]["stepLengths"].append(len(datasets[name][0]))
    return dstores


@delayed
def read_h5_chunk(fina, ds_path, slice_args):
    with h5py.File(fina, "r") as fh:
        dat = fh[ds_path][slice(*slice_args)]
    return dat


def dstore_to_darray(dstore):
    fina = pathlib.Path(dstore["file_path"])
    index = dask.array.from_delayed(
        read_h5_chunk(fina, dstore["index_dsp"], [None]),
        dstore["index_shape"],
        dtype=dstore["index_dtype"],
    )
    arrays = [
        dask.array.from_delayed(
            read_h5_chunk(fina, dstore["data_dsp"], tslice),
            tshape,
            dtype=dstore["data_dtype"],
        )
        for tslice, tshape in zip(
            dstore["data_chunks"]["slices"], dstore["data_chunks"]["shapes"]
        )
    ]
    data = dask.array.concatenate(arrays, axis=0)
    return index, data


def parse_filelist(flist):
    return dask.compute([parse_bs_h5_file(fina) for fina in flist])[0]


def readScanEcoJson_v01(file_name_json, exclude_from_files=None):
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
    for step in s["scan_files"]:
        for sstr in exclude_from_files:
            kill = []
            for i, tf in enumerate(step):
                if sstr in tf:
                    kill.append(i)
            for k in kill[-1::-1]:
                step.pop(k)

    return s, p


def parseScanEcoV01(
    file_name_json=None,
    search_paths=["./", "./scan_data/", "../scan_data"],
    memlimit_MB=100,
    createEscArrays=True,
    scan_info=None,
    scan_info_filepath=None,
    exclude_from_files=[],
):

    if file_name_json:
        """Data parser assuming eco-written files from pilot phase 1"""
        s, scan_info_filepath = readScanEcoJson_v01(
            file_name_json, exclude_from_files=exclude_from_files
        )
    else:
        s = scan_info
    # breakpoint()

    dstores = []
    for files_step in s["scan_files"]:
        dstores_step = []
        lastpath = None
        searchpaths = None
        for fina in files_step:
            fp = pathlib.Path(fina)
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
            dstores_step.append(parse_bs_h5_file(file_path))
        dstores.append(dstores_step)
    with ProgressBar():
        dstores = dask.compute(dstores)[0]

    # flatten files in step
    dstores_flat = []
    for dstore in dstores:
        tmp = {}
        for i in dstore:
            tmp.update(i)
        dstores_flat.append(tmp)

    # return dstores_flat
    chs = set()
    for dstore in dstores_flat:
        chs = chs.union(chs, set(list(dstore.keys())))

    # general scan info
    parameter = {
        parname: {"values": [], "attributes": {"Id": id_name}}
        for parname, id_name in zip(
            s["scan_parameters"]["name"], s["scan_parameters"]["Id"]
        )
    }
    parameter.update(
        {
            f"{parname}_readback": {"values": [], "attributes": {"Id": id_name}}
            for parname, id_name in zip(
                s["scan_parameters"]["name"], s["scan_parameters"]["Id"]
            )
        }
    )
    parameter.update({"scan_step_info": {"values": []}})

    escArrays = {}
    for ch in chs:
        arrays = []
        s_sl = []
        scan = []
        tparameter = copy(parameter)
        for stepNo, (scan_values, scan_readbacks, scan_step_info, dstore) in enumerate(
            zip(
                s["scan_values"], s["scan_readbacks"], s["scan_step_info"], dstores_flat
            )
        ):
            if ch not in dstore.keys():
                continue
            arrays.append(dstore_to_darray(dstore[ch]))
            s_sl.append(len(arrays[-1][0]))

            for par_name, value in zip(
                parameter.keys(),
                copy(scan_values) + copy(scan_readbacks) + [copy(scan_step_info)],
            ):
                tparameter[par_name]["values"].append(value)

        index_array = dask.array.concatenate([tr[0] for tr in arrays], axis=0).ravel()
        data_array = dask.array.concatenate([tr[1] for tr in arrays], axis=0)

        try:
            escArrays[ch] = Array(
                data=data_array,
                index=index_array,
                step_lengths=s_sl,
                parameter=tparameter,
            )
        except Exception as e:
            print(f"Could not create escape.Array for {ch};\nError: {str(e)}")

    return escArrays

    # datasets_scan = []

    # def get_datasets_from_files(n):
    #     lastpath = None
    #     searchpaths = None
    #     files = s["scan_files"][n]
    #     datasets = {}
    #     for n_file, f in enumerate(files):
    #         fp = pathlib.Path(f)
    #         fn = pathlib.Path(fp.name)
    #         if not searchpaths:
    #             searchpaths = [fp.parent] + [
    #                 scan_info_filepath.parent / pathlib.Path(tp.format(fp.parent.name))
    #                 for tp in search_paths
    #             ]
    #         for path in searchpaths:
    #             file_path = path / fn
    #             if file_path.is_file():
    #                 if not lastpath:
    #                     lastpath = path
    #                     searchpaths.insert(0, path)
    #                 break
    #         # assert file_path.is_file(), 'Could not find file {} '.format(fn)
    #         try:
    #             fh = h5py.File(file_path.resolve(), mode="r")
    #             datasets.update(utilities.findItemnamesGroups(fh, ["data", "pulse_id"]))
    #             logger.info("Successfully parsed file %s" % file_path.resolve())
    #         except:
    #             logger.warning(f"could not read {file_path.absolute().as_posix()}.")
    #     all_parses[n] = datasets

    # ts = []
    # for n in range(len(s["scan_files"])):
    #     ts.append(Thread(target=get_datasets_from_files, args=[n]))
    # for t in ts:
    #     t.start()
    # while len(all_parses) < len(s["scan_files"]):
    #     m = len(s["scan_files"])
    #     n = len(all_parses)
    #     files_bar.update(n - files_bar.n)
    #     # files_bar.update(n)
    #     sleep(0.01)
    # for t in ts:
    #     t.join()
    # # while not files_bar.n==files_bar.total:
    # # sleep(.01)
    # files_bar.update(files_bar.total - files_bar.n)

    # # datasets_scan.append(datasets)

    # names = set()
    # dstores = {}

    # # general scan info
    # parameter = {
    #     parname: {"values": [], "attributes": {"Id": id_name}}
    #     for parname, id_name in zip(
    #         s["scan_parameters"]["name"], s["scan_parameters"]["Id"]
    #     )
    # }
    # parameter.update(
    #     {
    #         f"{parname}_readback": {"values": [], "attributes": {"Id": id_name}}
    #         for parname, id_name in zip(
    #             s["scan_parameters"]["name"], s["scan_parameters"]["Id"]
    #         )
    #     }
    # )
    # parameter.update({"scan_step_info": {"values": []}})

    # for stepNo, (scan_values, scan_readbacks, scan_step_info) in enumerate(
    #     zip(s["scan_values"], s["scan_readbacks"], s["scan_step_info"])
    # ):
    #     datasets = all_parses[stepNo]
    #     tnames = set(datasets.keys())
    #     newnames = tnames.difference(names)
    #     oldnames = names.intersection(tnames)
    #     for name in newnames:
    #         if datasets[name][0].size == 0:
    #             logger.debug(
    #                 "Found empty dataset in {} in cycle {}".format(name, stepNo)
    #             )
    #         else:
    #             size_data = (
    #                 np.dtype(datasets[name][0].dtype).itemsize
    #                 * datasets[name][0].size
    #                 / 1024 ** 2
    #             )
    #             size_element = (
    #                 np.dtype(datasets[name][0].dtype).itemsize
    #                 * np.prod(datasets[name][0].shape[1:])
    #                 / 1024 ** 2
    #             )
    #             if datasets[name][0].chunks:
    #                 chunk_size = list(datasets[name][0].chunks)
    #             else:
    #                 chunk_size = list(datasets[name][0].shape)
    #             if chunk_size[0] == 1:
    #                 chunk_size[0] = int(memlimit_mD_MB // size_element)
    #             dstores[name] = {}

    #             # ToDo: get rid of bad definition in eco scan! (readbacks are just added as values but not as names).

    #             dstores[name]["parameter"] = copy(parameter)
    #             for par_name, value in zip(
    #                 parameter.keys(),
    #                 copy(scan_values) + copy(scan_readbacks) + [copy(scan_step_info)],
    #             ):
    #                 dstores[name]["parameter"][par_name]["values"].append(value)

    #             dstores[name]["data"] = []
    #             dstores[name]["data"].append(datasets[name][0])
    #             dstores[name]["data_chunks"] = chunk_size
    #             dstores[name]["eventIds"] = []
    #             dstores[name]["eventIds"].append(datasets[name][1])
    #             dstores[name]["stepLengths"] = []
    #             dstores[name]["stepLengths"].append(len(datasets[name][0]))
    #             names.add(name)
    #     for name in oldnames:
    #         if datasets[name][0].size == 0:
    #             logger.debug(
    #                 "Found empty dataset in {} in cycle {}".format(name, stepNo)
    #             )
    #         elif not len(datasets[name][0].shape) == len(
    #             dstores[name]["data"][0].shape
    #         ):
    #             logger.debug("Found inconsistent dataset in {}".format(name))
    #         elif not datasets[name][0].shape[0] == datasets[name][1].shape[0]:
    #             logger.debug("Found inconsistent dataset in {}".format(name))
    #         else:
    #             for par_name, value in zip(
    #                 parameter.keys(),
    #                 copy(scan_values) + copy(scan_readbacks) + [copy(scan_step_info)],
    #             ):
    #                 dstores[name]["parameter"][par_name]["values"].append(value)
    #             dstores[name]["data"].append(datasets[name][0])
    #             dstores[name]["eventIds"].append(datasets[name][1])
    #             dstores[name]["stepLengths"].append(len(datasets[name][0]))
    # if createEscArrays:
    #     escArrays = {}
    #     containers = {}
    #     for name, dat in dstores.items():
    #         containers[name] = LazyContainer(dat)
    #         escArrays[name] = Array(
    #             containers[name].get_data,
    #             index=containers[name].get_eventIds,
    #             step_lengths=dat["stepLengths"],
    #             parameter=dat["parameter"],
    #         )
    #     return escArrays
    # else:
    #     return dstores


class LazyContainer:
    def __init__(self, dat):
        self.dat = dat

    def get_data(self, **kwargs):
        return da.concatenate(
            [
                da.from_array(td, chunks=self.dat["data_chunks"])
                for td in self.dat["data"]
            ]
        )

    def get_eventIds(self):
        ids = {}

        def getids(n, dset):
            ids[n] = dset[...].ravel()

        ts = [
            Thread(target=getids, args=[n, td])
            for n, td in enumerate(self.dat["eventIds"])
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return np.concatenate([ids[n] for n in range(len(self.dat["eventIds"]))])
