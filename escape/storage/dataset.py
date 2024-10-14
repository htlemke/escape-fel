import base64
import os
import pickle
import hickle
from hickle.fileio import file_opener
import escape
from pathlib import Path
from escape.utilities import StructureGroup, dict2structure
from rich.tree import Tree
import logging
import h5py
import zarr
import numpy as np
import oschmod

try:
    from datastorage.datastorage import dictToH5Group, unwrapArray
except:
    print("issue with datastorage import!")

logger = logging.getLogger(__name__)


class DataSet:
    def __init__(
        self,
        raw_datasets: dict = None,
        alias_mappings: dict = None,
        results_file=None,
        mode="r",
        perm=None,
        name=None,
    ):
        self.data_raw = raw_datasets
        self.datasets = {}
        self._esc_types = {}

        if results_file is not None:
            # self.results_file = results_file
            self.results_file = filespec_to_file(results_file, mode=mode, perm=perm)
            self._init_datasets()
        else:
            self.results_file = None

        if alias_mappings:
            # print(alias_mappings)
            for idname in self.data_raw.keys():
                # print(idname)
                if idname in alias_mappings.keys():
                    # print(idname)
                    self.append(self.data_raw[idname], name=alias_mappings[idname])

        self.name = name

    def append(
        self,
        data,
        auto_format=True,
        as_hickle=False,
        as_pickle=False,
        as_datastorage=False,
        esc_type=None,
        name=None,
    ):
        self.datasets[name] = data

        if esc_type:
            auto_format = False
            as_hickle = False
            as_pickle = False
            as_datastorage = False
            if esc_type == "pickled":
                as_pickle = True
            elif esc_type == "hickled":
                as_hickle = True
            elif esc_type == "datastorage":
                as_datastorage = True

        if self.results_file is not None:
            if (
                auto_format
                and (not isinstance(data, escape.Array))
                and (not as_hickle)
                and (not as_pickle)
                and (not as_datastorage)
            ):
                if isinstance(self.results_file, h5py.File):
                    as_hickle = True
                elif isinstance(self.results_file, zarr.Group):
                    as_pickle = True

            if isinstance(data, escape.Array):
                data.name = name
                if self.results_file is not None:
                    self.datasets[name].set_h5_storage(self.results_file, name)
            else:
                if as_pickle:
                    # self.results_file.require_dataset(name)
                    self.results_file[name] = np.string_(pickle.dumps(data))
                    self.results_file[name].attrs["esc_type"] = "pickled"
                    self._esc_types[name] = "pickled"
                if as_hickle:
                    # self.results_file.require_dataset(name)
                    hickle.dump(data, self.results_file, path=f"/{name}")
                    self.results_file[name].attrs["esc_type"] = "hickled"
                    self._esc_types[name] = "hickled"
                elif as_datastorage:
                    self.results_file.require_group(name)
                    dictToH5Group(data, self.results_file[name])
                    self.results_file[name].attrs["esc_type"] = "datastorage"
                    self._esc_types[name] = "datastorage"
        else:
            pass
            # print(
            #     f"No dataset results_file defined, data {name} will be attached in memory only."
            # )

        if isinstance(data, dict):
            self.__dict__[name] = StructureGroup()
            dict2structure(data, base=self.__dict__[name])
        else:
            dict2structure({name: data}, base=self)

        return data

    def get_datasets_max_element_size(self, max_element_size=5000, verbose=0):
        ks = []
        for k, v in self.datasets.items():
            if not isinstance(v, escape.Array):
                continue
            try:
                if np.prod(v.shape[1:]) <= max_element_size:
                    if verbose:
                        print(k)
                    ks.append(k)
            except:
                pass
        return ks

    def store_datasets_max_element_size(
        self, max_element_size=5000, lock="auto", verbose=0, **kwargs
    ):
        ks = []
        for k, v in self.datasets.items():
            if not isinstance(v, escape.Array):
                continue
            try:
                if np.prod(v.shape[1:]) < max_element_size:
                    if verbose:
                        print(k)
                    ks.append(k)
            except:
                pass
        return escape.store([self.datasets[k] for k in ks], lock=lock, **kwargs)

    def compute_datasets_max_element_size(
        self, max_element_size=5000, verbose=0, **kwargs
    ):
        ds = {}
        for k, v in self.datasets.items():
            if not isinstance(v, escape.Array):
                continue
            try:
                if np.prod(v.shape[1:]) < max_element_size:
                    if verbose:
                        print(k)
                    ds[k] = v
            except:
                pass
        lo = escape.compute(*[v for k, v in ds.items()], **kwargs)
        for n, (k, v) in enumerate(ds.items()):
            self.append(lo[n], name=k)

    def __repr__(self):
        s = object.__repr__(self)
        s += "\n"
        s += "items\n"
        for k in self.__dict__.keys():
            if not k[0] == "_":
                s += "    " + k + "\n"
        return s

    def get_structure_tree(self, base=None):
        if not base:
            base = Tree("")
        for key, item in self.__dict__.items():
            if hasattr(item, "get_structure_tree"):
                item.get_structure_tree(base=base.add(key))
            else:
                base.add(key).add(str(item))
        return base

    def _init_datasets(self):
        for tname in self.results_file.keys():
            if "esc_type" in self.results_file[tname].attrs.keys():
                if self.results_file[tname].attrs["esc_type"] == "array_dataset":
                    larray = escape.Array.load_from_h5(self.results_file, tname)
                    if larray:
                        self.append(larray, name=tname)
                    self._esc_types[tname] = "array_dataset"
                else:
                    if self.results_file[tname].attrs["esc_type"] == "pickled":
                        self.datasets[tname] = pickle.loads(
                            self.results_file[tname][()]
                        )
                        dict2structure({tname: self.datasets[tname]}, base=self)
                        self._esc_types[tname] = "pickled"
                    elif self.results_file[tname].attrs["esc_type"] == "hickled":
                        self.datasets[tname] = hickle.load(
                            self.results_file, path=f"/{tname}"
                        )
                        dict2structure({tname: self.datasets[tname]}, base=self)
                        self._esc_types[tname] = "hickled"
                    elif self.results_file[tname].attrs["esc_type"] == "datastorage":
                        self.datasets[tname] = unwrapArray(self.results_file[tname])
                        dict2structure({tname: self.datasets[tname]}, base=self)
                        self._esc_types[tname] = "datastorage"
                    if isinstance(self.datasets[tname], dict):
                        self.__dict__[tname] = StructureGroup()
                        dict2structure(self.datasets[tname], base=self.__dict__[tname])
                    else:
                        dict2structure({tname: self.datasets[tname]}, base=self)

            else:
                try:
                    self.append(
                        escape.Array.load_from_h5(result_file, tname), name=tname
                    )
                except:
                    pass

    @classmethod
    def load_from_result_file(cls, results_filepath, mode="r", name=None, perm=None):
        ds = cls(results_file=results_filepath, name=name, perm=perm)
        return ds

    @classmethod
    def create_with_new_result_file(
        cls, results_filepath, mode="w", force_overwrite=False, name=None
    ):
        if Path(results_filepath).exists() and not force_overwrite:
            if (
                input(
                    f"Filename {results_filepath} exists, would you like to overwrite its contents? (y/n)"
                )
                == "y"
            ):
                pass
            else:
                return

        ds = cls(results_file=results_filepath, mode=mode, name=name)
        return ds


def filespec_to_file(
    file,
    mode="r",
    perm="g+rw",
    default_dataset_compression="lzf",
    default_dataset_compression_opts=None,
):
    if isinstance(file, Path) or isinstance(file, str):
        results_filepath = Path(file)
        if not ".esc" in results_filepath.suffixes:
            raise Exception("Expecting esc suffix in filename")
        if ".h5" in results_filepath.suffixes:
            result_file = h5py.File(results_filepath, mode)

        elif ".zarr" in results_filepath.suffixes:
            result_file = zarr.open(results_filepath, mode=mode)
        if perm is not None:
            print("changing perms")
            try:
                oschmod.set_mode_recursive(results_filepath, perm)
            except:
                print(f"Warning:failed setting permissions {perm:s}")
    elif isinstance(file, h5py.File):
        result_file = file
    elif isinstance(file, zarr.Group):
        result_file = file
    if default_dataset_compression:
        result_file.attrs["default_dataset_compression"] = default_dataset_compression
        result_file.attrs["default_dataset_compression_opts"] = (
            default_dataset_compression_opts
        )
    return result_file


def merge_datasets(datasets, only_escape_arrays=False, **kwargs_dataset):
    """Merges datasets of multiple dataset containers into one dataset container. escape.arrays are here concatenated
    to a merged escape_array, other datatypes are only merges as python lists.

    Args:
        datasets list or iterable of the dataset containers to be merged (type DataSet).
        only_escape_arrays (bool, optional): optionally only merging of escape.arrays. Defaults to False.

    Returns:
        dataset_merged: new DataSet instance where the data have been merged into.
    """

    dsets_common = list(set.intersection(*[set(td.datasets.keys()) for td in datasets]))
    dsets_all = list(set.union(*[set(td.datasets.keys()) for td in datasets]))
    dsets_stranded = set(dsets_all) - set(dsets_common)
    # print(dsets_stranded)

    d_merged = escape.DataSet(**kwargs_dataset)
    for dset_name in dsets_common:
        dsets = [td.datasets[dset_name] for td in datasets]
        if all([isinstance(tdset, escape.Array) for tdset in dsets]):

            try:
                ta = escape.concatenate(dsets)
            except:
                dsets_simple = [
                    escape.Array(data=td.data, index=td.index) for td in dsets
                ]
                ta = escape.concatenate(dsets_simple)
            d_merged.append(ta, name=dset_name)
        elif not only_escape_arrays:
            try:
                d_merged.append(dsets, name=dset_name)
            except:
                print(f"NB: Could not merge and append common dataset {dset_name}")
    return d_merged


def convert_resultsfile(
    filename,
    out_filename=None,
    out_directory=None,
    out_type="h5",
    force_overwrite=False,
    close_if_feasible=True,
):
    """Convert resultsfile, typically from zarr to h5 for easier handling.

    Args:
        filename (string): Input file name
        out_filename (string, optional): output filename, if None is deduced from input name. Defaults to None.
        out_directory (string or Path instance, optional): output directory. Defaults to None.
        out_type (str, optional): output file type. Defaults to 'h5'.
        force_overwrite (bool, optional): Fore overwriting if output file exists. Defaults to False.
        close_if_feasible (bool, optional): close the covertet file (for h5). Defaults to True.

    Returns:
        str or dataset: returns output filename in case file is closed otherwise the output dataset.
    """

    filename = Path(filename)
    filename = filename.resolve()
    if not filename.exists():
        raise (Exception(f"File {filename} is not existing!"))

    if not out_filename:
        out_filename = ".".join([filename.stem, out_type])

    if out_directory:
        out_filename = Path(out_directory) / out_filename
    out_filename

    ds_in = DataSet.load_from_result_file(filename)
    ds_out = DataSet.create_with_new_result_file(
        results_filepath=out_filename, force_overwrite=force_overwrite
    )

    escapearrays = {}
    for tdsname, tdsdat in ds_in.datasets.items():
        esc_type = ds_in._esc_types.get(tdsname, None)
        if isinstance(tdsdat, escape.Array):
            escapearrays[tdsname] = escape.Array(
                data=tdsdat.data,
                index=tdsdat.index,
                step_lengths=tdsdat.scan.step_lengths,
                parameter=tdsdat.scan.parameter,
                name=tdsdat.name,
            )
            ds_out.append(escapearrays[tdsname], esc_type=esc_type, name=tdsname)
        else:
            ds_out.append(tdsdat, esc_type=esc_type, name=tdsname)

    escape.store([ds_out.datasets[tname] for tname in escapearrays.keys()])

    if close_if_feasible:
        if hasattr(ds_out.results_file, "close"):
            ds_out.results_file.close()
            print("closed")

        return out_filename
    else:
        return ds_out
