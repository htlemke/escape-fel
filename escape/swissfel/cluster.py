from functools import partial
from glob import escape
import json
from os import stat
import os
import pathlib
from pathlib import Path

from dask.array.routines import shape
import distributed
from ..parse import utilities
import h5py
from dask import array as da
from dask import bag as db
from distributed import progress
from dask import delayed
from dask.diagnostics import ProgressBar
import dask

from .. import Array, Scan
import numpy as np
from copy import deepcopy as copy
import logging
import warnings
from lazy_object_proxy import Proxy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm
from threading import Thread
from time import sleep

try:
    import bitshuffle.h5
except:
    print("Could not import bitshuffle.h5!")
from dask_jobqueue import SLURMCluster
from distributed import Client
from dask.utils import SerializableLock
import socket
import getpass
import escape.storage
from rich.progress import track

logger = logging.getLogger(__name__)

# cluter helpers


class SwissFelCluster:
    def __init__(self, local=True, cores=8, memory="24 GB", workers=5, **kwags_cluster):
        if local:
            self.client = distributed.Client()
        else:
            self.cluster = SLURMCluster(cores=cores, memory=memory, **kwags_cluster)
            self.client = Client(self.cluster)
        self.ip = socket.gethostbyname(socket.gethostname())
        self.dashboard_port_scheduler = self.client._scheduler_identity.get("services")[
            "dashboard"
        ]
        self.username = getpass.getuser()
        self.lock = SerializableLock()
        escape.STORAGE_LOCK = self.lock

    def _repr_html_(self):
        return self.client._repr_html_()

    def scale_workers(self, N_workers):
        self.cluster.scale(N_workers)

    def create_dashboard_tunnel(self, ssh_host="ra"):
        print(
            "type following commant in a terminal, if port is taken, change first number in command."
        )
        print(
            " ".join(
                [
                    f"jupdbport={self.dashboard_port_scheduler}",
                    "&&",
                    "ssh",
                    "-f",
                    "-L",
                    f"$jupdbport:{self.ip}:{self.dashboard_port_scheduler}",
                    f"{self.username}@{ssh_host}",
                    "sleep 10",
                    "&&",
                    "firefox",
                    "http://localhost:$jupdbport",
                ]
            )
        )


# parsing stuff


@delayed
def parse_bs_h5_file(fina, memlimit_MB=100):
    """Data parser assuming the standard swissfel h5 format for raw data"""
    # if (type(files) is str) or (not np.iterable(files)):
    #     files = [files]
    fina = Path(fina)
    try:
        with h5py.File(fina.resolve(), mode="r") as fh:
            datasets = utilities.findItemnamesGroups(fh, ["data", "pulse_id"])
            logger.info("Successfully parsed file %s" % fina.resolve())
            dstores = {}
            for name, (ds_data, ds_index) in datasets.items():
                if ds_data.size == 0:
                    logger.debug("Found empty dataset in {}".format(name))
                    continue
                # data first
                dtype = np.dtype(ds_data.dtype)
                size_element = (
                    np.dtype(ds_data.dtype).itemsize
                    * np.prod(ds_data.shape[1:])
                    / 1024**2
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
                    "file_path": fina.resolve().as_posix(),
                    "data_dsp": ds_data.name,
                    "data_shape": ds_data.shape,
                    "data_dtype": dtype.str,
                    "data_chunks": {"slices": slices, "shapes": chunk_shapes},
                    "index_dsp": ds_index.name,
                    "index_dtype": ds_index.dtype.str,
                    "index_shape": ds_index.shape,
                }
                # dstores[name]["stepLengths"] = []
                # dstores[name]["stepLengths"].append(len(datasets[name][0]))
        return dstores
    except:
        return {}


@delayed
def read_h5_chunk(fina, ds_path, slice_args):
    import bitshuffle.h5

    with h5py.File(fina, "r") as fh:
        dat = fh[ds_path][slice(*slice_args)]
    return dat


def dstore_to_darray(dstore):
    fina = pathlib.Path(dstore["file_path"])
    index = dask.array.from_delayed(
        read_h5_chunk(fina, dstore["index_dsp"], [None]),
        dstore["index_shape"],
        dtype=np.dtype(dstore["index_dtype"]),
    )
    arrays = [
        dask.array.from_delayed(
            read_h5_chunk(fina, dstore["data_dsp"], tslice),
            tshape,
            dtype=np.dtype(dstore["data_dtype"]),
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
    lazyEscArrays=False,
    scan_info=None,
    scan_info_filepath=None,
    exclude_from_files=[],
    checknstore_parsing_result=False,
    clear_parsing_result=False,
    return_json_info=False,
    step_selection=slice(None),
    run_root_directory=None,
    perm="0o0665",
    verbose=0,
):
    if file_name_json:
        """Data parser assuming eco-written files from pilot phase 1"""
        s, scan_info_filepath = readScanEcoJson_v01(
            file_name_json, exclude_from_files=exclude_from_files
        )
        if Path(file_name_json).parent.stem == "aux":
            run_root_directory = Path(file_name_json).parent.parent
    else:
        s = scan_info

    if checknstore_parsing_result:
        if checknstore_parsing_result == "same_directory":
            parse_res_file = (
                scan_info_filepath.parent.resolve()
                / scan_info_filepath.with_suffix(".parse_result.json")
            )

        elif checknstore_parsing_result == "work_directory":
            tp = scan_info_filepath.parent.resolve()
            for p in tp.resolve().parents:
                if len(p.name) == 6 and p.name[0] == "p" and p.name[1:].isnumeric():
                    pgroup = p.name
                    tp = Path(f"/das/work/{pgroup[:3]}/{pgroup}")
                    break

            parse_res_file = (
                tp
                / Path(".escape_parse_result")
                / Path(*scan_info_filepath.with_suffix(".parse_result.json").parts[1:])
            )

            parse_res_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            parse_res_file = (
                Path(checknstore_parsing_result)
                / Path(".escape_parse_result")
                / Path(*scan_info_filepath.with_suffix(".parse_result.json").parts[1:])
            )
            parse_res_file.parent.mkdir(parents=True, exist_ok=True)
            if perm is not None:
                os.chmod(parse_res_file.parent, (int(perm,8) + int("0o111",8)))
        if clear_parsing_result and Path(parse_res_file).exists():
            Path(parse_res_file).unlink()

    files_parsed = set()

    if checknstore_parsing_result and Path(parse_res_file).exists():
        with open(parse_res_file, "r") as fp:
            dstores_flat = json.load(fp)
        for step in dstores_flat:
            for dsn, dss in step.items():
                files_parsed.add(Path(dss["file_path"]))
    else:
        dstores_flat = []

    dstores = []
    fls = dict(known=[], toparse=[])
    for files_step in s["scan_files"][step_selection]:
        dstores_step = []
        lastpath = None
        searchpaths = None
        for fina in files_step:
            fp = pathlib.Path(fina)
            if (not fp.is_absolute()) and run_root_directory:
                fp = run_root_directory / fp
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
            if file_path.resolve() in files_parsed:
                fls["known"].append(file_path)
                continue
            elif file_path.resolve().exists():
                fls["toparse"].append(file_path)
            else:
                continue
            dstores_step.append(parse_bs_h5_file(file_path))
        dstores.append(dstores_step)
    if verbose:
        statstr = "Data files analyzed: "
        statstr += "{} to parse of {}".format(
            len(fls["toparse"]), len(fls["toparse"]) + len(fls["known"])
        )
        if verbose > 1:
            for fl in fls["toparse"]:
                statstr += "   " + fl.as_posix() + "\n"
        print(statstr)
    if verbose:
        print("Starting to parse data files ...")
    with ProgressBar():
        dstores = dask.compute(dstores, scheduler="processes")[0]
    if verbose:
        print("... done parsing data files.")
    # flatten files in step
    if any(dstores):
        for dstore in dstores:
            tmp = {}
            for i in dstore:
                tmp.update(i)
            dstores_flat.append(tmp)

        if checknstore_parsing_result and dstores_flat:
            with open(parse_res_file, "w") as fp:
                json.dump(dstores_flat, fp)
            if perm is not None:
                os.chmod(parse_res_file, perm)

    else:
        if verbose:
            print("No new parsing results")

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

    if verbose:
        print("Starting to create escape arrays ...")

    escArrays = {}

    if lazyEscArrays:
        for ch in chs:
            escArrays[ch] = Proxy(
                partial(
                    create_arrays_from_dstores,
                    ch,
                    s,
                    dstores_flat,
                    parameter,
                    step_selection,
                )
            )

    else:
        # escArrays = []
        for ch in track(chs, description="Creating arrays ..."):
            # print(f"starting to create for {ch}")
            escArrays[ch] = create_arrays_from_dstores(
                ch, s, dstores_flat, parameter, step_selection
            )
        if verbose:
            print("really Starting to create escape arrays ...")

        with ProgressBar():
            dstores = dask.compute(escArrays, scheduler="threads")[0]

        if verbose:
            print("... done creating escape arrays.")

    if return_json_info:
        return escArrays, s
    else:
        return escArrays


def create_arrays_from_dstores(ch, s, dstores_flat, parameter, step_selection):
    arrays = []
    s_sl = []
    scan = []
    tparameter = copy(parameter)
    for iteratornumber, (
        scan_values,
        scan_readbacks,
        scan_step_info,
        dstore,
    ) in enumerate(
        zip(
            s["scan_values"][step_selection],
            s["scan_readbacks"][step_selection],
            s["scan_step_info"][step_selection],
            dstores_flat,
        )
    ):
        if ch not in dstore.keys():
            continue
        arrays.append(dstore_to_darray(dstore[ch]))
        s_sl.append(len(arrays[-1][0]))

        for par_name, value in zip(
            tparameter.keys(),
            copy(scan_values) + copy(scan_readbacks) + [copy(scan_step_info)],
        ):
            tparameter[par_name]["values"].append(value)

    index_array = dask.array.concatenate([tr[0] for tr in arrays], axis=0).ravel()
    data_array = dask.array.concatenate([tr[1] for tr in arrays], axis=0)
    try:
        tarr = Array(
            data=data_array,
            index=index_array,
            step_lengths=s_sl,
            parameter=tparameter,
        )
        return tarr

    except Exception as e:
        print(f"Could not create escape.Array for {ch};\nError: {str(e)}")


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
