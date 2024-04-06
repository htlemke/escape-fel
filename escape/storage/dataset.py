import base64
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
        name=None,
    ):
        self.data_raw = raw_datasets
        self.datasets = {}
        self._esc_types = {}

        if results_file is not None:
            # self.results_file = results_file
            self.results_file = filespec_to_file(results_file, mode=mode)
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
    def load_from_result_file(cls, results_filepath, mode="r", name=None):
        ds = cls(results_file=results_filepath, name=name)
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


def filespec_to_file(file, mode="r"):
    if isinstance(file, Path) or isinstance(file, str):
        results_filepath = Path(file)
        if not ".esc" in results_filepath.suffixes:
            raise Exception("Expecting esc suffix in filename")
        if ".h5" in results_filepath.suffixes:
            result_file = h5py.File(results_filepath, mode)
        elif ".zarr" in results_filepath.suffixes:
            result_file = zarr.open(results_filepath, mode=mode)
    elif isinstance(file, h5py.File):
        result_file = file
    elif isinstance(file, zarr.Group):
        result_file = file
    return result_file
