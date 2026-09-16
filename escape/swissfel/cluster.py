from functools import partial
from glob import escape
import json
import re
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
from threading import Thread, Lock
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
import oschmod

logger = logging.getLogger(__name__)

# cluter helpers

import hashlib

def myhash(s,size=8):
    h = hashlib.blake2b(s.encode(), digest_size=size)
    return h.hexdigest()



class SwissFelCluster:
    def __init__(self, local=True, cores=8, memory="24 GB", workers=5, processes=8, minimum_jobs=1, maximum_jobs=10, **kwags_cluster):
        if local:
            self.client = distributed.Client()
        else:
            self.cluster = SLURMCluster(
                n_workers=workers, 
                processes=processes, 
                cores=cores, 
                memory=memory, 
                job_extra_directives=[
                    '--output=/dev/null',
                    '--error=/dev/null',
                    "--job-name=escape",
                ],
                queue='hour',
                
                **kwags_cluster
                    )
            self.client = Client(self.cluster)

            self.cluster.adapt(
                minimum_jobs=minimum_jobs, 
                maximum_jobs=maximum_jobs, 
                wait_count=5,          # Wait 5 check-cycles before scaling down
                interval="1s"          # Check every second if more resources are needed
            )
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
        self.cluster.scale(N_workers,jobs=N_workers)

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
def parse_bs_h5_file(fina, memlimit_MB=500):
    """Data parser assuming the standard swissfel h5 format for raw data"""
    # if (type(files) is str) or (not np.iterable(files)):
    #     files = [files]
    fina = Path(fina)
    try:

        
        tmtime = os.path.getmtime(fina)
        tfsize = os.stat(fina).st_size
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
                    "file_mtime" :  tmtime,
                    "file_size" : tfsize,
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
    perm=None,
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
    # >>> parse result file/directory creation clearing if needed
    parse_res_file = None
    if checknstore_parsing_result:
        try:
            if checknstore_parsing_result == "same_directory":
                parse_res_file = (
                    scan_info_filepath.parent.resolve()
                    / scan_info_filepath.with_suffix(".parse_result.json")
                )

            elif checknstore_parsing_result == "work_directory":
                tp = scan_info_filepath.parent.resolve()
                for p in tp.resolve().parents:
                    if len(p.name) == 6 and p.name[0] == "p" and p.name[1:].isnumeric():
                        tp = Path(f"/das/work/{p.name[:3]}/{p.name}")
                        break
                parse_res_file = (
                    tp
                    / Path(".escape_parse_result")
                    / Path(str(myhash(scan_info_filepath.as_posix()))).with_suffix(".parse_result.json")
                )
                parse_res_file.parent.mkdir(parents=True, exist_ok=True)
                _try_set_world_writable(parse_res_file.parent)
            else:
                parse_res_file = (
                    Path(checknstore_parsing_result)
                    / Path(".escape_parse_result")
                    / Path(str(myhash(scan_info_filepath.as_posix()))).with_suffix(".parse_result.json")
                )
                parse_res_file.parent.mkdir(parents=True, exist_ok=True)
                _try_set_world_writable(parse_res_file.parent)

            if clear_parsing_result and parse_res_file.exists():
                parse_res_file.unlink()
        except Exception as exc:
            logger.warning(
                "Cannot set up parse result path (%s) — parse result will not be stored.", exc
            )
            parse_res_file = None
    # <<< parse result file/directory creation clearing if needed

    files_parsed = set()
    dstores_flat = []

    # finding previously parsed files that didn't change in filesize.
    if parse_res_file is not None and Path(parse_res_file).exists():
        try:
            print("Parse result file found.")
            with open(parse_res_file, "r") as fp:
                dstores_flat = json.load(fp)
            for step in dstores_flat:
                for dsn, dss in step.items():
                    tpath = Path(dss["file_path"])
                    try:
                        tfsize = os.stat(tpath).st_size
                        if "file_size" in dss.keys() and dss["file_size"] == tfsize:
                            files_parsed.add(tpath.resolve())
                    except OSError:
                        pass
        except Exception as exc:
            logger.warning("Cannot read parse result cache (%s) — will re-scan all files.", exc)
            dstores_flat = []
    elif checknstore_parsing_result:
        print("No parse result file found — will scan all files.")

    dstores = []
    fls = dict(known=[], toparse=[])
    steps_complete = []
    non_existing_files = []
    for n_step,files_step in enumerate(s["scan_files"][step_selection]):
        dstores_step = []
        lastpath = None #???
        searchpaths = None #???
        complete_files_in_step = 0
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
                try:  # neded if permissions fail.
                    if file_path.is_file():
                        if not lastpath:
                            lastpath = path
                            searchpaths.insert(0, path)
                        break
                except:
                    pass
            if file_path.resolve() in files_parsed:
                fls["known"].append(file_path)
                complete_files_in_step+=1
                continue
            elif file_path.resolve().exists():
                # print(f'new_file {file_path}')
                fls["toparse"].append(file_path)
                dstores_step.append(parse_bs_h5_file(file_path, memlimit_MB=memlimit_MB))
                complete_files_in_step+=1
            else:
                non_existing_files.append(file_path)
        steps_complete.append(complete_files_in_step == len(files_step))
        # if not len(dstores_step)==0:
        dstores.append(dstores_step)
    # print(f'{sum(steps_complete)} out of selected {len(steps_complete)} steps completely parsed.')
        # print(f'dstores hash : {hash(repr(dstores))}')
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Running on a single-machine scheduler",
            category=UserWarning,
        )
        with ProgressBar():
            dstores = dask.compute(dstores, scheduler="processes")[0]
    if verbose:
        print("... done parsing data files.")
    # flatten files in step
    
    if any(dstores):
        for step_no,dstore in enumerate(dstores):
            # collect all in step as dict
            tmp = {}
            for i in dstore:
                tmp.update(i)

            if step_no < len(dstores_flat):
                dstores_flat[step_no].update(tmp)
            else:
                dstores_flat.append(tmp)

        if parse_res_file is not None and dstores_flat:
            try:
                print(f'Writing parse result ({len(fls["toparse"])} new file(s)) → {parse_res_file}')
                _atomic_write_json(parse_res_file, dstores_flat)
            except Exception as exc:
                logger.warning("Cannot write parse result cache: %s", exc)


    else:
        if verbose:
            print("No new parsing results")
     
    # finding all names 
    chs = set()
    for dstore in dstores_flat:
        chs = chs.union(set(list(dstore.keys())))
    # general scan info
    if "Id" in s["scan_parameters"]:
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
    else:
        parameter = {
            parname: {"values": []}
            for parname in s["scan_parameters"]["name"]
        }
        parameter.update(
            {
                f"{parname}_readback": {"values": []}
                for parname in s["scan_parameters"]["name"]
            }
        )

    if "grid_specs" in s["scan_parameters"]:
        grid_specs = s["scan_parameters"]["grid_specs"]
    else:
        grid_specs = None
    
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
                    grid_specs=grid_specs,
                )
            )

    else:
        # escArrays = []
        for ch in track(chs, description="Creating arrays ..."):
            # print(f"starting to create for {ch}")
            escArrays[ch] = create_arrays_from_dstores(
                ch, s, dstores_flat, parameter, step_selection, grid_specs=grid_specs
            )
        if verbose:
            print("really Starting to create escape arrays ...")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Running on a single-machine scheduler",
                category=UserWarning,
            )
            with ProgressBar():
                dstores = dask.compute(escArrays, scheduler="threads")[0]

        if verbose:
            print("... done creating escape arrays.")

    if return_json_info:
        return escArrays, s
    else:
        return escArrays


def create_arrays_from_dstores(ch, s, dstores_flat, parameter, step_selection, grid_specs=None):
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
            grid_specs=grid_specs,
            name=ch,
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


def _try_set_world_writable(path):
    """Set world-readable/writable permissions; silently ignore failures.

    Directories get 0o777 plus setgid (0o2000, "g+s") — instead of clobbering
    the mode outright — so files later created inside inherit the directory's
    (project) group rather than falling back to a caller's own default group,
    which on shared GPFS filesets can be charged against an unrelated group
    quota. Files get 0o666. Errors from network filesystems, quota limits, or
    missing ownership are swallowed so callers never crash over a permission
    issue.
    """
    try:
        if Path(path).is_dir():
            os.chmod(path, 0o777 | 0o2000)
        else:
            os.chmod(path, 0o666)
    except (PermissionError, OSError):
        pass


def _atomic_write_json(path, data):
    """Write JSON to `path` atomically via a temp file + os.replace().

    A plain open("w")+json.dump() truncates `path` immediately, so a crash or
    quota error mid-write leaves a permanently corrupt, permission-locked
    0-byte file that later readers/writers (possibly different users on a
    shared cache) can never recover from. Writing to a per-process temp file
    first and swapping it in with os.replace() (atomic on POSIX) means
    `path` is either left fully intact or fully replaced.
    """
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        with tmp_path.open("w") as fh:
            json.dump(data, fh)
        _try_set_world_writable(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


# =============================================================================
# v02 parser — clean rewrite of parseScanEcoV01
# =============================================================================
#
# Design changes vs v01
# ---------------------
# * HDF5 datasets are wrapped in _H5ProxyV02 and passed directly to
#   da.from_array().  The file is opened and closed on every chunk read, so
#   no file handle is held open and the proxy is fully picklable.
# * The metadata scan (previously @delayed + processes scheduler) runs in a
#   ThreadPoolExecutor.  h5py releases the GIL during I/O, making threads
#   as fast as processes with none of the pickling overhead.
# * The JSON cache stores only (path, dataset-path, shape, dtype, chunk0) per
#   channel — no pre-computed slice lists.  The default suffix is
#   ".parse_result_v02.json" so v01 and v02 caches never collide.
# * Array construction is a plain Python loop (no additional dask.compute
#   call — the v01 compute on the Array dict was a no-op).
# =============================================================================


class _H5ProxyV02:
    """Serialisable, file-closing proxy to an HDF5 dataset.

    Implements the array-like interface expected by ``dask.array.from_array``.
    Opens the HDF5 file on every ``__getitem__`` call and closes it
    immediately, so no file handle is kept alive between chunk reads.

    Because it contains only plain Python types it is picklable and therefore
    compatible with every dask scheduler (sync, threads, processes,
    distributed).
    """

    __slots__ = ("file_path", "dset_path", "shape", "dtype", "ndim")

    def __init__(
        self,
        file_path: str,
        dset_path: str,
        dset_shape: tuple,
        dtype: str,
    ):
        self.file_path = str(file_path)
        self.dset_path = dset_path
        self.shape = tuple(dset_shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        try:
            import bitshuffle.h5  # noqa: F401 — side-effect only: registers HDF5 filter
        except ImportError:
            pass
        with h5py.File(self.file_path, "r") as fh:
            return fh[self.dset_path][key]

    def __dask_tokenize__(self):
        # Stable token so dask can deduplicate identical datasets in one graph.
        return (self.file_path, self.dset_path, self.shape)


def readScanEcoJson_v02(file_name_json, exclude_from_files=()):
    """Load an eco scan JSON file and apply file-name exclusion filters.

    Equivalent to ``readScanEcoJson_v01`` but raises ``FileNotFoundError``
    instead of an assertion error when the path does not exist.

    Returns
    -------
    tuple[dict, pathlib.Path]
        ``(scan_info_dict, resolved_json_path)``
    """
    p = Path(file_name_json).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Scan JSON not found: {p}")
    with p.open() as fh:
        s = json.load(fh)
    assert len(s["scan_files"]) == len(s["scan_values"]), (
        f"scan_files / scan_values length mismatch in {p}"
    )
    assert len(s["scan_files"]) == len(s["scan_readbacks"]), (
        f"scan_files / scan_readbacks length mismatch in {p}"
    )
    for step in s["scan_files"]:
        to_drop = [
            i for i, f in enumerate(step)
            if any(ex in f for ex in exclude_from_files)
        ]
        for i in reversed(to_drop):
            step.pop(i)
    return s, p


def scan_h5_file_v02(fina, memlimit_MB: float = 500) -> dict:
    """Scan one SwissFEL HDF5 file and return JSON-serialisable channel metadata.

    Unlike ``parse_bs_h5_file`` (v01) this function reads **only** dataset
    paths, shapes and dtypes — no data and no chunk-slice lists.  The first-
    dimension chunk hint (``data_chunk0``) is a single integer derived from
    *memlimit_MB*.

    Designed to run in a thread: h5py releases the GIL during file I/O so
    many files can be scanned concurrently without a process pool.

    Returns an empty dict if the file cannot be opened.
    """
    fina = Path(fina).resolve()
    try:
        tmtime = os.path.getmtime(fina)
        tfsize = os.stat(fina).st_size
        with h5py.File(fina, "r") as fh:
            datasets = utilities.findItemnamesGroups(fh, ["data", "pulse_id"])
            result = {}
            for name, (ds_data, ds_index) in datasets.items():
                if ds_data.size == 0:
                    logger.debug("Empty dataset %s in %s", name, fina)
                    continue
                elem_bytes = ds_data.dtype.itemsize * int(
                    np.prod(ds_data.shape[1:]) if ds_data.ndim > 1 else 1
                )
                chunk0 = max(1, int(memlimit_MB * 1024 ** 2 / elem_bytes))
                result[name] = {
                    "file_path": fina.as_posix(),
                    "data_dsp": ds_data.name,
                    "data_shape": list(ds_data.shape),
                    "data_dtype": ds_data.dtype.str,
                    "data_chunk0": chunk0,
                    "index_dsp": ds_index.name,
                    "index_shape": list(ds_index.shape),
                    "index_dtype": ds_index.dtype.str,
                    "file_mtime": tmtime,
                    "file_size": tfsize,
                }
        return result
    except Exception as exc:
        logger.warning("Could not scan %s: %s", fina, exc)
        return {}


def _resolve_cache_path_v02(mode, scan_info_filepath: Path, perm=None) -> Path:
    """Return the Path where the v02 parse-result JSON cache should be stored.

    *mode* follows the same convention as ``checknstore_parsing_result`` in
    ``parseScanEcoV01``:

    * ``"same_directory"`` — next to the scan JSON file.
    * ``"work_directory"`` — under the pgroup's ``/das/work`` directory.
    * Any other truthy string — treated as the cache parent directory.

    The default cache filename suffix is ``.parse_result_v02.json`` to avoid
    colliding with v01 caches that use ``.parse_result.json``.
    """
    hashed_name = Path(myhash(scan_info_filepath.as_posix())).with_suffix(
        ".parse_result_v02.json"
    )

    if mode == "same_directory":
        return scan_info_filepath.with_suffix(".parse_result_v02.json")

    if mode == "work_directory":
        tp = scan_info_filepath.parent.resolve()
        for p in tp.parents:
            if len(p.name) == 6 and p.name[0] == "p" and p.name[1:].isnumeric():
                tp = Path(f"/das/work/{p.name[:3]}/{p.name}")
                break
        parent = tp / ".escape_parse_result"
    else:
        parent = Path(mode) / ".escape_parse_result"

    parent.mkdir(parents=True, exist_ok=True)
    _try_set_world_writable(parent)

    return parent / hashed_name


def _build_step_darray_v02(ch_meta: dict):
    """Build ``(index_darray, data_darray)`` from one channel's file metadata.

    Uses ``da.from_array`` with an ``_H5ProxyV02`` so the HDF5 file is opened
    and closed per chunk read without any pre-loading.
    """
    data_shape = tuple(ch_meta["data_shape"])
    chunk0 = min(ch_meta["data_chunk0"], data_shape[0])
    # chunk only the event (first) axis; trailing dimensions are kept whole
    data_chunks = (chunk0,) + data_shape[1:]

    index_shape = tuple(ch_meta["index_shape"])

    data_proxy = _H5ProxyV02(
        ch_meta["file_path"], ch_meta["data_dsp"], data_shape, ch_meta["data_dtype"]
    )
    index_proxy = _H5ProxyV02(
        ch_meta["file_path"], ch_meta["index_dsp"], index_shape, ch_meta["index_dtype"]
    )

    data_arr  = da.from_array(data_proxy,  chunks=data_chunks)
    index_arr = da.from_array(index_proxy, chunks=index_shape)  # read index in one shot
    return index_arr, data_arr


def _build_escape_array_v02(
    ch: str,
    scan_info: dict,
    dstores_flat: list,
    base_parameter: dict,
    step_selection,
    grid_specs,
):
    """Build an ``escape.Array`` for channel *ch* from pre-scanned HDF5 metadata.

    Iterates the selected scan steps, concatenates per-step dask arrays, and
    assembles the scan parameter dict.  Returns ``None`` if the channel is
    absent from all steps.
    """
    tparameter = copy(base_parameter)
    index_parts = []
    data_parts  = []
    step_lengths = []

    for sv, srb, ssi, step_dstore in zip(
        scan_info["scan_values"][step_selection],
        scan_info["scan_readbacks"][step_selection],
        scan_info["scan_step_info"][step_selection],
        dstores_flat,
    ):
        if ch not in step_dstore:
            continue

        idx_arr, dat_arr = _build_step_darray_v02(step_dstore[ch])
        index_parts.append(idx_arr)
        data_parts.append(dat_arr)
        step_lengths.append(dat_arr.shape[0])

        for par_name, val in zip(tparameter, copy(sv) + copy(srb) + [copy(ssi)]):
            tparameter[par_name]["values"].append(val)

    if not data_parts:
        return None

    index = da.concatenate(index_parts).ravel()
    data  = da.concatenate(data_parts, axis=0)

    try:
        return Array(
            data=data,
            index=index,
            step_lengths=step_lengths,
            parameter=tparameter,
            grid_specs=grid_specs,
            name=ch,
        )
    except Exception as exc:
        logger.error("Could not create Array for %s: %s", ch, exc)
        return None


def parseScanEcoV02(
    file_name_json=None,
    search_paths=("./", "./scan_data/", "../scan_data"),
    memlimit_MB=100,
    scan_info=None,
    scan_info_filepath=None,
    exclude_from_files=(),
    checknstore_parsing_result=False,
    clear_parsing_result=False,
    return_json_info=False,
    step_selection=slice(None),
    run_root_directory=None,
    perm=None,
    n_scan_workers=None,
    createEscArrays=True,
    lazyEscArrays=False,
    verbose=0,
):
    """Parse a SwissFEL eco scan and return a dict of escape.Array objects.

    Drop-in replacement for ``parseScanEcoV01`` with the following changes:

    **HDF5 → dask via da.from_array**
        Each dataset is wrapped in a serialisable :class:`_H5ProxyV02` and
        passed directly to ``da.from_array``.  The HDF5 file is opened and
        closed on every chunk read; no file handle is kept alive.

    **Thread-based file scanning**
        Metadata scanning (path, shape, dtype) is I/O-bound and h5py releases
        the GIL, so a ``ThreadPoolExecutor`` replaces the v01
        ``dask.compute(..., scheduler="processes")`` approach.  This avoids
        pickling overhead and is typically faster for metadata-only reads.

    **Simpler JSON cache**
        The cache stores ``(file_path, dset_path, shape, dtype, chunk0)`` per
        channel — no pre-computed slice lists.  The default suffix is
        ``.parse_result_v02.json`` so v01 and v02 caches never collide.

    Parameters
    ----------
    file_name_json : str or Path, optional
        Path to the eco scan JSON file.
    search_paths : sequence of str
        Relative path patterns tried when a data file is not at its recorded
        absolute path.
    memlimit_MB : float
        Target maximum chunk size in MB for the first array dimension.
    scan_info : dict, optional
        Pre-loaded scan JSON dict (alternative to *file_name_json*).
    scan_info_filepath : Path, optional
        Path used for cache-key hashing when *scan_info* is given directly.
    exclude_from_files : sequence of str
        Files whose path contains any of these substrings are skipped.
    checknstore_parsing_result : bool or str
        Cache mode: ``False`` — no cache; ``"same_directory"`` — next to the
        JSON; ``"work_directory"`` — under the pgroup work tree; any other
        string — that path is used as the cache parent directory.
    clear_parsing_result : bool
        Delete any existing cache before running.
    return_json_info : bool
        If ``True``, return ``(arrays_dict, scan_info_dict)`` instead of just
        the arrays dict.
    step_selection : slice or array-like
        Subset of scan steps to load.  Defaults to all steps.
    run_root_directory : str or Path, optional
        Prepended to relative file paths recorded in the JSON.
    perm : int, optional
        Unix permission bits applied to newly created cache directories/files.
    n_scan_workers : int, optional
        Thread-pool size for parallel file scanning.  Defaults to
        ``min(32, n_new_files)``.
    verbose : int
        0 = silent, 1 = progress messages, 2 = per-file detail.

    Returns
    -------
    dict[str, escape.Array]
        One Array per detector / channel name present in the scan data.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── 1. Load scan JSON ────────────────────────────────────────────────────
    if file_name_json is not None:
        s, scan_info_filepath = readScanEcoJson_v02(
            file_name_json, exclude_from_files=exclude_from_files
        )
        if Path(file_name_json).parent.stem == "aux":
            run_root_directory = Path(file_name_json).parent.parent
    elif scan_info is not None:
        s = scan_info
    else:
        raise ValueError("Provide either file_name_json or scan_info.")

    # ── 2. Resolve (and optionally clear) the cache file path ────────────────
    cache_path = None
    if checknstore_parsing_result:
        try:
            cache_path = _resolve_cache_path_v02(
                checknstore_parsing_result, scan_info_filepath
            )
            if clear_parsing_result and cache_path.exists():
                cache_path.unlink()
                logger.info("Cleared parse cache: %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Cannot set up parse result path (%s) — parse result will not be stored.", exc
            )
            cache_path = None

    # ── 3. Load existing cache ───────────────────────────────────────────────
    # dstores_flat: list[dict], one dict per selected step, mapping
    #               channel_name -> metadata_dict
    dstores_flat: list = []
    cached_files: set = set()

    if cache_path is not None and cache_path.exists():
        try:
            print(f"Parse result file found: {cache_path}")
            with cache_path.open() as fh:
                dstores_flat = json.load(fh)
            for step in dstores_flat:
                for ch_meta in step.values():
                    fp = Path(ch_meta["file_path"])
                    try:
                        if "file_size" in ch_meta and os.stat(fp).st_size == ch_meta["file_size"]:
                            cached_files.add(fp)
                    except OSError:
                        pass
        except Exception as exc:
            logger.warning("Cannot read parse result cache (%s) — will re-scan all files.", exc)
            dstores_flat = []
    elif checknstore_parsing_result:
        print("No parse result file found — will scan all files.")

    # ── 4. Discover files; separate cached from new ──────────────────────────
    # step_file_map[i] = {resolved_path: "cached"|"new"|None}
    step_file_map: list = []
    files_to_scan: list = []

    for files_step in s["scan_files"][step_selection]:
        step_files = {}
        searchpaths = None

        for fina_str in files_step:
            fp = Path(fina_str)
            if not fp.is_absolute() and run_root_directory:
                fp = Path(run_root_directory) / fp

            if searchpaths is None:
                searchpaths = [fp.parent] + [
                    scan_info_filepath.parent
                    / Path(pat.format(fp.parent.name))
                    for pat in search_paths
                ]

            found = None
            for sp in searchpaths:
                candidate = (sp / fp.name).resolve()
                try:
                    if candidate.is_file():
                        found = candidate
                        if found.parent not in searchpaths:
                            searchpaths.insert(0, found.parent)
                        break
                except OSError:
                    pass

            if found is None:
                logger.warning("File not found: %s", fp.name)
                step_files[None] = None
                continue

            if found in cached_files:
                step_files[found] = "cached"
            else:
                step_files[found] = "new"
                files_to_scan.append(found)

        step_file_map.append(step_files)

    if verbose:
        n_new = sum(1 for v in (v for s in step_file_map for v in s.values()) if v == "new")
        n_known = sum(1 for v in (v for s in step_file_map for v in s.values()) if v == "cached")
        print(f"Files: {n_known} cached, {n_new} to scan.")
        if verbose > 1:
            for fp in files_to_scan:
                print(f"  scan: {fp}")

    # ── 5. Parallel metadata scan of new files ────────────────────────────────
    scan_results: dict = {}
    if files_to_scan:
        if verbose:
            print(f"Scanning {len(files_to_scan)} file(s) …")
        n_workers = n_scan_workers or min(32, len(files_to_scan))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_map = {
                ex.submit(scan_h5_file_v02, fp, memlimit_MB): fp
                for fp in files_to_scan
            }
            for fut in as_completed(future_map):
                scan_results[future_map[fut]] = fut.result()
        if verbose:
            print("… scan complete.")

    # ── 6. Merge new scan results into dstores_flat ───────────────────────────
    for step_idx, step_files in enumerate(step_file_map):
        new_channels = {}
        for fp, status in step_files.items():
            if fp is not None and status == "new" and fp in scan_results:
                new_channels.update(scan_results[fp])

        if step_idx < len(dstores_flat):
            dstores_flat[step_idx].update(new_channels)
        else:
            dstores_flat.append(new_channels)

    # ── 7. Persist updated cache if anything is new ───────────────────────────
    if cache_path is not None and scan_results:
        try:
            print(f"Writing parse result ({len(files_to_scan)} new file(s)) → {cache_path}")
            _atomic_write_json(cache_path, dstores_flat)
        except Exception as exc:
            logger.warning("Cannot write parse result cache: %s", exc)

    # ── 8. Build scan parameter template ─────────────────────────────────────
    par_names = s["scan_parameters"]["name"]
    has_ids   = "Id" in s["scan_parameters"]

    if has_ids:
        par_ids = s["scan_parameters"]["Id"]
        base_parameter = {
            name: {"values": [], "attributes": {"Id": pid}}
            for name, pid in zip(par_names, par_ids)
        }
        base_parameter.update({
            f"{name}_readback": {"values": [], "attributes": {"Id": pid}}
            for name, pid in zip(par_names, par_ids)
        })
    else:
        base_parameter = {name: {"values": []} for name in par_names}
        base_parameter.update(
            {f"{name}_readback": {"values": []} for name in par_names}
        )

    base_parameter["scan_step_info"] = {"values": []}
    grid_specs = s["scan_parameters"].get("grid_specs")

    # ── 9. Collect all channel names across steps ─────────────────────────────
    all_channels: set = set()
    for step in dstores_flat:
        all_channels.update(step.keys())

    if verbose:
        print(f"Building escape.Arrays for {len(all_channels)} channel(s) …")

    # ── 10. Assemble escape.Array per channel ─────────────────────────────────
    esc_arrays = {}
    if lazyEscArrays:
        for ch in all_channels:
            esc_arrays[ch] = Proxy(
                partial(
                    _build_escape_array_v02,
                    ch, s, dstores_flat, base_parameter, step_selection, grid_specs,
                )
            )
    else:
        for ch in track(all_channels, description="Building arrays …"):
            arr = _build_escape_array_v02(
                ch, s, dstores_flat, base_parameter, step_selection, grid_specs
            )
            if arr is not None:
                esc_arrays[ch] = arr

    if verbose:
        print(f"Done. {len(esc_arrays)} array(s) returned.")

    return (esc_arrays, s) if return_json_info else esc_arrays


# =============================================================================
# v03 parser — dead-end poaching on top of v02
# =============================================================================
#
# Design changes vs v02
# ---------------------
# * v02 still pays for a full HDF5 tree walk (findItemnamesGroups) on every
#   file: h5py's visititems commits to visiting every descendant of every
#   group before the callback runs, so groups that never contain a channel
#   cost just as much as ones that do.
# * v03 replaces that with findItemnamesGroups_v03, a manually-recursed walk
#   that can be *pruned*: once a group's subtree has been found (on an
#   earlier file of the same *kind*, see _file_kind below) to contain no
#   channel anywhere below it, later files of that kind skip descending
#   into it entirely — no re-verification, by design, since re-verifying
#   would cost exactly what pruning is meant to avoid. Groups that *do*
#   match a channel are still re-checked every time (one cheap `.keys()`
#   call) so a channel dropping mid-run is still caught.
#
# A real bug found while testing against p23415/run0076 (exp "26h"), fixed
# in this version
# --------------------------------------------------------------------------
# A single scan step's file list mixes genuinely different file *kinds* —
# e.g. ``acqNNNN.BSDATA.h5`` (flat, ~176 channel groups), ``acqNNNN.PVDATA.h5``
# (metadata only, *zero* channel groups anywhere), and
# ``acqNNNN.JF01T03V01.h5`` (one channel nested under /data/JF01T03V01).
# The first version of this module shared *one* dead-end registry across
# every file in the scan regardless of kind. Since an h5py.File's root
# group is always named "/" no matter which physical file it belongs to,
# scanning a PVDATA file (root "/" legitimately has no matches) marked "/"
# itself as a dead end — and because PVDATA files are trivial to scan they
# tend to finish first, poisoning "/" for every BSDATA/JF file scanned
# afterward in the same call, which then returned nothing at all without
# recursing into anything. Dead-end knowledge is only valid for files that
# actually share a layout, so it is now scoped per file kind
# (_file_kind, one _PathRegistry per kind) and never shared across kinds.
#
# * This "dead end" knowledge is kept in one _PathRegistry per file kind,
#   shared between the worker pool's threads (plain dict + Lock) or, if
#   use_processes=True, across a process pool (multiprocessing.Manager
#   dict + Lock) — workers populate it as they scan, and later-scheduled
#   files of the same kind benefit as soon as any earlier one of that kind
#   has established a subtree is empty.
# * The registries' dead-end sets are persisted per kind in the JSON cache,
#   so a second run against the same instrument configuration starts
#   pruned from file one instead of only after its first scan.
# * Residual assumption: within one file kind, the set of channel groups is
#   stable across the run (verified for BSDATA on run0076: identical 176
#   channels on file 1 and file 31). Pass poach_dead_ends=False to disable
#   pruning entirely and fall back to a full walk per file, if a given
#   beamline's file layout ever turns out not to hold that invariant.
# * Everything else (array construction, HDF5 proxy, caching of shape/dtype
#   per channel) is unchanged from v02 and is reused directly.
# =============================================================================


def _file_kind(path) -> str:
    """Group key for dead-end registry scoping: files sharing this key are
    assumed to share an HDF5 layout and may safely share dead-end knowledge.

    SwissFEL raw files follow the convention ``acqNNNN.KIND.h5`` (e.g.
    ``acq0001.BSDATA.h5``, ``acq0001.PVDATA.h5``, ``acq0001.JF01T03V01.h5``)
    where KIND identifies the DAQ stream/detector and determines the file's
    internal layout — different kinds can have completely different
    structure (see the module comment above: PVDATA files have no channel
    groups at all, unlike BSDATA/JF). The leading ``acqNNNN`` token is
    stripped so files differing only by acquisition number share a key.

    Falls back to the full file name (i.e. no sharing with any other file)
    when the name doesn't match that convention, which is always safe —
    it just disables pruning for that file rather than risking an unsafe
    assumption.
    """
    name = Path(path).name
    parts = name.split(".", 1)
    if len(parts) == 2 and re.fullmatch(r"acq\d+", parts[0]):
        return parts[1]
    return name


class _PathRegistry:
    """Shared record of HDF5 group subtrees known to contain no channel data.

    Backed by a plain dict + threading.Lock by default, which is safe and
    fast when workers are threads (the default in parseScanEcoV03, matching
    v02's rationale: h5py releases the GIL during I/O). Pass a
    multiprocessing.Manager to make the same interface safe to hand to a
    ProcessPoolExecutor instead — the manager-backed dict/lock are plain
    picklable proxies, so no other change is needed to run across processes.
    """

    def __init__(self, manager=None, initial_dead_ends=None):
        if manager is not None:
            self._dead_ends = manager.dict({p: True for p in (initial_dead_ends or ())})
            self._lock = manager.Lock()
        else:
            self._dead_ends = dict.fromkeys(initial_dead_ends or ())
            self._lock = Lock()

    def dead_ends_snapshot(self) -> set:
        with self._lock:
            return set(self._dead_ends.keys())

    def update_dead_ends(self, paths) -> None:
        if not paths:
            return
        with self._lock:
            for p in paths:
                self._dead_ends[p] = True


readScanEcoJson_v03 = readScanEcoJson_v02


def scan_h5_file_v03(
    fina,
    registry: "_PathRegistry",
    item_names=("data", "pulse_id"),
    memlimit_MB: float = 500,
    poach: bool = True,
) -> dict:
    """v03 of scan_h5_file_v02: prunes the HDF5 walk using *registry*.

    Identical return shape to scan_h5_file_v02 (one dict per channel with
    file_path/data_dsp/data_shape/data_dtype/data_chunk0/index_* /
    file_mtime/file_size) so it is a drop-in for _build_step_darray_v02 and
    _build_escape_array_v02. The only difference is that discovery goes
    through findItemnamesGroups_v03 with the registry's current dead-end
    snapshot, and any newly-established dead ends are fed back into the
    registry so concurrently- or later-running scans of other files of the
    *same kind* (see _file_kind — callers must never share one registry
    across files of different kinds) benefit immediately. Pass poach=False
    to skip consulting/updating the registry and always do a full,
    unpruned walk (identical to scan_h5_file_v02's discovery step).
    """
    fina = Path(fina).resolve()
    try:
        tmtime = os.path.getmtime(fina)
        tfsize = os.stat(fina).st_size
        with h5py.File(fina, "r") as fh:
            known_dead_ends = registry.dead_ends_snapshot() if poach else set()
            datasets, new_dead_ends, _ = utilities.findItemnamesGroups_v03(
                fh, list(item_names), known_dead_ends=known_dead_ends
            )
            result = {}
            for name, (ds_data, ds_index) in datasets.items():
                if ds_data.size == 0:
                    logger.debug("Empty dataset %s in %s", name, fina)
                    continue
                elem_bytes = ds_data.dtype.itemsize * int(
                    np.prod(ds_data.shape[1:]) if ds_data.ndim > 1 else 1
                )
                chunk0 = max(1, int(memlimit_MB * 1024 ** 2 / elem_bytes))
                result[name] = {
                    "file_path": fina.as_posix(),
                    "data_dsp": ds_data.name,
                    "data_shape": list(ds_data.shape),
                    "data_dtype": ds_data.dtype.str,
                    "data_chunk0": chunk0,
                    "index_dsp": ds_index.name,
                    "index_shape": list(ds_index.shape),
                    "index_dtype": ds_index.dtype.str,
                    "file_mtime": tmtime,
                    "file_size": tfsize,
                }
        if poach:
            registry.update_dead_ends(new_dead_ends)
        return result
    except Exception as exc:
        logger.warning("Could not scan %s: %s", fina, exc)
        return {}


def _resolve_cache_path_v03(mode, scan_info_filepath: Path, perm=None) -> Path:
    """Same convention as _resolve_cache_path_v02, with its own suffix
    (.parse_result_v03.json) so v01/v02/v03 caches never collide and each
    version can be compared or reverted to independently."""
    hashed_name = Path(myhash(scan_info_filepath.as_posix())).with_suffix(
        ".parse_result_v03.json"
    )

    if mode == "same_directory":
        return scan_info_filepath.with_suffix(".parse_result_v03.json")

    if mode == "work_directory":
        tp = scan_info_filepath.parent.resolve()
        for p in tp.parents:
            if len(p.name) == 6 and p.name[0] == "p" and p.name[1:].isnumeric():
                tp = Path(f"/das/work/{p.name[:3]}/{p.name}")
                break
        parent = tp / ".escape_parse_result"
    else:
        parent = Path(mode) / ".escape_parse_result"

    parent.mkdir(parents=True, exist_ok=True)
    _try_set_world_writable(parent)

    return parent / hashed_name


def parseScanEcoV03(
    file_name_json=None,
    search_paths=("./", "./scan_data/", "../scan_data"),
    memlimit_MB=100,
    scan_info=None,
    scan_info_filepath=None,
    exclude_from_files=(),
    checknstore_parsing_result=False,
    clear_parsing_result=False,
    return_json_info=False,
    step_selection=slice(None),
    run_root_directory=None,
    perm=None,
    n_scan_workers=None,
    use_processes=False,
    poach_dead_ends=True,
    createEscArrays=True,
    lazyEscArrays=True,
    verbose=0,
):
    """Parse a SwissFEL eco scan and return a dict of escape.Array objects.

    Drop-in replacement for parseScanEcoV02 that additionally prunes the
    per-file HDF5 discovery walk using dead-end knowledge shared between
    the scanning workers — see the module-level "v03 parser" comment block
    above for the rationale, including a real bug found and fixed while
    testing against p23415/run0076 ("26h"): dead-end knowledge must be
    scoped per file *kind* (_file_kind) — a single scan step can mix file
    kinds with completely different HDF5 layouts (e.g. SwissFEL raw scans
    combine BSDATA, PVDATA, and per-detector JF files in the same step),
    and PVDATA files legitimately have no channel groups at all, so sharing
    one registry across kinds could prune real data in other files. v01 and
    v02 are left untouched; switch back by calling parseScanEcoV01/V02
    instead if v03 ever misbehaves for a given beamline's file layout.

    New/changed parameters vs parseScanEcoV02
    ------------------------------------------
    use_processes : bool
        False (default): scan files with a ThreadPoolExecutor, same as v02
        — h5py releases the GIL for I/O, so threads are as fast as
        processes here without pickling overhead, and let the dead-end
        registries be plain dict+Lock. True: scan with a
        ProcessPoolExecutor instead, backing the registries with a
        multiprocessing.Manager so dead-end knowledge is still shared
        across the worker processes.
    poach_dead_ends : bool
        True (default): prune the HDF5 walk using dead-end knowledge shared
        between files of the same kind (see _file_kind). False: fall back
        to a full, unpruned walk per file (identical behaviour to v02's
        discovery step) while keeping v03's other bookkeeping — use this if
        a beamline's files ever violate the assumption that a file kind's
        channel set is stable across the run.
    lazyEscArrays : bool
        True (default here, unlike v01/v02 where it defaults to False):
        build dask-backed lazy arrays instead of eagerly computing them.
        Array construction (dask.compute(escArrays, scheduler="threads"))
        is unchanged from v02 and shares none of v03's scanning speedup, so
        defaulting to lazy avoids paying that eager cost up front regardless
        of parser version.

    All other parameters match parseScanEcoV02.
    """
    from concurrent.futures import (
        ThreadPoolExecutor,
        ProcessPoolExecutor,
        as_completed,
    )

    # ── 1. Load scan JSON ────────────────────────────────────────────────────
    if file_name_json is not None:
        s, scan_info_filepath = readScanEcoJson_v03(
            file_name_json, exclude_from_files=exclude_from_files
        )
        if Path(file_name_json).parent.stem == "aux":
            run_root_directory = Path(file_name_json).parent.parent
    elif scan_info is not None:
        s = scan_info
    else:
        raise ValueError("Provide either file_name_json or scan_info.")

    # ── 2. Resolve (and optionally clear) the cache file path ────────────────
    cache_path = None
    if checknstore_parsing_result:
        try:
            cache_path = _resolve_cache_path_v03(
                checknstore_parsing_result, scan_info_filepath
            )
            if clear_parsing_result and cache_path.exists():
                cache_path.unlink()
                logger.info("Cleared parse cache: %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Cannot set up parse result path (%s) — parse result will not be stored.", exc
            )
            cache_path = None

    # ── 3. Load existing cache (dstores + previously-known dead ends) ────────
    # cached_dead_ends_by_kind maps _file_kind(path) -> set of dead-end group
    # paths; kinds are never mixed (see module comment: PVDATA/BSDATA/JF
    # files in the same run have unrelated HDF5 layouts).
    dstores_flat: list = []
    cached_dead_ends_by_kind: dict = {}
    cached_files: set = set()

    if cache_path is not None and cache_path.exists():
        try:
            print(f"Parse result file found: {cache_path}")
            with cache_path.open() as fh:
                cached = json.load(fh)
            dstores_flat = cached.get("dstores_flat", [])
            cached_dead_ends_by_kind = {
                kind: set(paths)
                for kind, paths in cached.get("dead_ends_by_kind", {}).items()
            }
            for step in dstores_flat:
                for ch_meta in step.values():
                    fp = Path(ch_meta["file_path"])
                    try:
                        if "file_size" in ch_meta and os.stat(fp).st_size == ch_meta["file_size"]:
                            cached_files.add(fp)
                    except OSError:
                        pass
        except Exception as exc:
            logger.warning("Cannot read parse result cache (%s) — will re-scan all files.", exc)
            dstores_flat = []
            cached_dead_ends_by_kind = {}
    elif checknstore_parsing_result:
        print("No parse result file found — will scan all files.")

    # ── 4. Discover files; separate cached from new ──────────────────────────
    step_file_map: list = []
    files_to_scan: list = []

    for files_step in s["scan_files"][step_selection]:
        step_files = {}
        searchpaths = None

        for fina_str in files_step:
            fp = Path(fina_str)
            if not fp.is_absolute() and run_root_directory:
                fp = Path(run_root_directory) / fp

            if searchpaths is None:
                searchpaths = [fp.parent] + [
                    scan_info_filepath.parent
                    / Path(pat.format(fp.parent.name))
                    for pat in search_paths
                ]

            found = None
            for sp in searchpaths:
                candidate = (sp / fp.name).resolve()
                try:
                    if candidate.is_file():
                        found = candidate
                        if found.parent not in searchpaths:
                            searchpaths.insert(0, found.parent)
                        break
                except OSError:
                    pass

            if found is None:
                logger.warning("File not found: %s", fp.name)
                step_files[None] = None
                continue

            if found in cached_files:
                step_files[found] = "cached"
            else:
                step_files[found] = "new"
                files_to_scan.append(found)

        step_file_map.append(step_files)

    if verbose:
        n_new = sum(1 for v in (v for s in step_file_map for v in s.values()) if v == "new")
        n_known = sum(1 for v in (v for s in step_file_map for v in s.values()) if v == "cached")
        print(f"Files: {n_known} cached, {n_new} to scan.")
        if verbose > 1:
            for fp in files_to_scan:
                print(f"  scan: {fp}")

    # ── 5. Parallel metadata scan of new files, pruned via per-kind registries
    # Registries are keyed by _file_kind and never shared across kinds — see
    # the module comment above for why that scoping is load-bearing (a
    # single shared registry let an empty PVDATA file poison "/" as a dead
    # end for every BSDATA/JF file scanned afterward in the same run).
    scan_results: dict = {}
    manager = None
    if files_to_scan:
        if verbose:
            print(f"Scanning {len(files_to_scan)} file(s) …")
        n_workers = n_scan_workers or min(32, len(files_to_scan))

        kinds_present = {_file_kind(fp) for fp in files_to_scan}
        if use_processes:
            import multiprocessing

            manager = multiprocessing.Manager()
            registries = {
                kind: _PathRegistry(
                    manager=manager,
                    initial_dead_ends=cached_dead_ends_by_kind.get(kind, set()),
                )
                for kind in kinds_present
            }
            executor_cls = ProcessPoolExecutor
        else:
            registries = {
                kind: _PathRegistry(
                    initial_dead_ends=cached_dead_ends_by_kind.get(kind, set())
                )
                for kind in kinds_present
            }
            executor_cls = ThreadPoolExecutor

        with executor_cls(max_workers=n_workers) as ex:
            future_map = {
                ex.submit(
                    scan_h5_file_v03,
                    fp,
                    registries[_file_kind(fp)],
                    ("data", "pulse_id"),
                    memlimit_MB,
                    poach_dead_ends,
                ): fp
                for fp in files_to_scan
            }
            for fut in track(
                as_completed(future_map),
                total=len(files_to_scan),
                description="Scanning files …",
            ):
                scan_results[future_map[fut]] = fut.result()

        for kind, reg in registries.items():
            cached_dead_ends_by_kind[kind] = reg.dead_ends_snapshot()
        if manager is not None:
            manager.shutdown()
        if verbose:
            n_dead = sum(len(v) for v in cached_dead_ends_by_kind.values())
            print(f"… scan complete ({n_dead} dead-end subtree(s) known across {len(cached_dead_ends_by_kind)} file kind(s)).")

    # ── 6. Merge new scan results into dstores_flat ───────────────────────────
    for step_idx, step_files in enumerate(step_file_map):
        new_channels = {}
        for fp, status in step_files.items():
            if fp is not None and status == "new" and fp in scan_results:
                new_channels.update(scan_results[fp])

        if step_idx < len(dstores_flat):
            dstores_flat[step_idx].update(new_channels)
        else:
            dstores_flat.append(new_channels)

    # ── 7. Persist updated cache (dstores + per-kind dead ends) if new ───────
    if cache_path is not None and (scan_results or cached_dead_ends_by_kind):
        try:
            print(f"Writing parse result ({len(files_to_scan)} new file(s)) → {cache_path}")
            _atomic_write_json(
                cache_path,
                {
                    "dstores_flat": dstores_flat,
                    "dead_ends_by_kind": {
                        kind: sorted(paths)
                        for kind, paths in cached_dead_ends_by_kind.items()
                    },
                },
            )
        except Exception as exc:
            logger.warning("Cannot write parse result cache: %s", exc)

    # ── 8. Build scan parameter template ─────────────────────────────────────
    par_names = s["scan_parameters"]["name"]
    has_ids   = "Id" in s["scan_parameters"]

    if has_ids:
        par_ids = s["scan_parameters"]["Id"]
        base_parameter = {
            name: {"values": [], "attributes": {"Id": pid}}
            for name, pid in zip(par_names, par_ids)
        }
        base_parameter.update({
            f"{name}_readback": {"values": [], "attributes": {"Id": pid}}
            for name, pid in zip(par_names, par_ids)
        })
    else:
        base_parameter = {name: {"values": []} for name in par_names}
        base_parameter.update(
            {f"{name}_readback": {"values": []} for name in par_names}
        )

    base_parameter["scan_step_info"] = {"values": []}
    grid_specs = s["scan_parameters"].get("grid_specs")

    # ── 9. Collect all channel names across steps ─────────────────────────────
    all_channels: set = set()
    for step in dstores_flat:
        all_channels.update(step.keys())

    if verbose:
        print(f"Building escape.Arrays for {len(all_channels)} channel(s) …")

    # ── 10. Assemble escape.Array per channel (unchanged from v02) ───────────
    esc_arrays = {}
    if lazyEscArrays:
        for ch in all_channels:
            esc_arrays[ch] = Proxy(
                partial(
                    _build_escape_array_v02,
                    ch, s, dstores_flat, base_parameter, step_selection, grid_specs,
                )
            )
    else:
        for ch in track(all_channels, description="Building arrays …"):
            arr = _build_escape_array_v02(
                ch, s, dstores_flat, base_parameter, step_selection, grid_specs
            )
            if arr is not None:
                esc_arrays[ch] = arr

    if verbose:
        print(f"Done. {len(esc_arrays)} array(s) returned.")

    return (esc_arrays, s) if return_json_info else esc_arrays
