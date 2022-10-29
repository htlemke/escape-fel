import escape
from pathlib import Path
from escape.utilities import dict2structure
from rich.tree import Tree
import logging
import h5py
import zarr

logger = logging.getLogger(__name__)


class DataSet:
    def __init__(
        self,
        raw_datasets: dict = None,
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
            # print(alias_mappings)
            for idname in self.data_raw.keys():
                # print(idname)
                if idname in alias_mappings.keys():
                    # print(idname)
                    self.append(self.data_raw[idname], name=alias_mappings[idname])

        self.name = name

    def append(self, data, name=None):
        self.datasets[name] = data
        if isinstance(data, escape.Array):
            data.name = name
            self.datasets[name].set_h5_storage(self.results_file, name)

        dict2structure({name: data}, base=self)

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

    @classmethod
    def load_from_result_file(cls, results_filepath, mode="r", name=None):
        results_filepath = Path(results_filepath)
        if not ".esc" in results_filepath.suffixes:
            raise Exception("Expecting esc suffix in filename")
        if ".h5" in results_filepath.suffixes:
            result_file = h5py.File(results_filepath, mode)
        elif ".zarr" in results_filepath.suffixes:
            result_file = zarr.open(results_filepath, mode=mode)

        ds = cls(results_file=result_file, name=name)

        for tname in result_file.keys():
            try:
                ds.append(escape.Array.load_from_h5(result_file, tname), name=tname)
            except:
                pass

        return ds
