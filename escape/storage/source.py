import pickle
from distributed.protocol import serialize, deserialize
import inspect

SOURCETYPES = [
    "factory",
    "dataset",
    "status",
    "array_map_index_blocks",
    # A leaf Array with no real data, standing in for data to be supplied
    # later (see escape.storage.graph.placeholder) -- and the recorded
    # result of an operation applied to one (see .../graph.py's record_op) --
    # both unbound until escape.storage.graph.bind() supplies real data.
    "placeholder",
    "symbolic_op",
]


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
        role=None,
        shape=None,
        dtype=None,
        func_name=None,
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
        elif type == "placeholder":
            self.role = role
            self.shape = shape
            self.dtype = dtype
        elif type == "symbolic_op":
            # func_name resolves via escape.storage.graph's registry, not a
            # live function reference -- see that module's docstring for why
            # (this needs to survive being reloaded in a different process).
            self.func_name = func_name
            self.args = args
            self.kwargs = kwargs

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
        from datastorage.datastorage import dictToH5Group

        dictToH5Group(self.as_dict(), group=group)
