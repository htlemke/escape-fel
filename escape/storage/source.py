import pickle
from distributed.protocol import serialize, deserialize
from datastorage.datastorage import dictToH5, dictToH5Group
import inspect

SOURCETYPES = ["factory", "dataset", "status", "array_map_index_blocks"]


class Source:
    def __init__(
        self,
        type,
        factory=None,
        args=[],
        kwargs={},
        base_dataset=None,
        iargout=None,
        name_dataset=None,
    ):
        if type not in SOURCETYPES:
            raise ValueError(f'Type "{type}" not in {SOURCETYPES}!')
        self.type = type
        if type == "factory":
            # sig = inspect.signature(factory)
            self.factory = factory
            self.args = args
            self.kwargs = kwargs
            self.iargout = iargout
        elif type == "array_map_index_blocks":
            self.factory = factory
            self.args = args
            self.kwargs = kwargs
            self.iargout = 0
            self.base_dataset = Source("dataset", name_dataset=base_dataset.name)
        elif type == "dataset":
            self.name_dataset = name_dataset

    @classmethod
    def from_group(cls, group):
        pass

    def as_dict(self):
        d = {"type": self.type}
        if d["type"] == "factory":
            d["factory"] = self.factory
            d["args"] = self.args
            d["kwargs"] = self.kwargs
            d["iargaut"] = self.iargout
        if d["type"] == "dataset":
            d["name_dataset"] = self.name_dataset

        return d

    def get_factory_cfg(self):
        cfg = {}
        cfg["module"] = self.factory.__module__
        cfg["name"] = self.factory.__name__
        cfg["pickle"] = pickle.dumps(self.factory)
        # cfg['dask'] = serialize

    def get_result(self, datasets=None):
        """ "Create the output from a source. This might crequire a collection of datasets
        (this is a dictionary where keys are the dataset name"""
        if self.type == "dataset":
            return datasets[self.name_dataset]
        if self.type == "factory":
            res = self.factory(*self.args, **self.kwargs)
            if not self.iargout == None:
                return res[self.iargout]
            else:
                return res

    def read_from_h5py_group(self):
        ...

    def write_to_h5py_group(self, group):
        """

        Args:
            group (h5py or zarr group): h5py __parent__ group object into which the source group will be written.
        """
        dictToH5Group(self.as_dict(), group=group)
