import pickle
from distributed.protocol import serialize, deserialize


SOURCETYPES = ["factory", "dataset", "status"]


class Source:
    def __init__(
        self, type, factory=None, args=[], kwargs={}, iargout=None, name_dataset=None
    ):
        if type not in SOURCETYPES:
            raise ValueError(f'Type "{type}" not in {SOURCETYPES}!')
        self.type = type
        if type == "factory":
            self.factory = factory
            self.args = args
            self.kwargs = kwargs
            self.iargout = iargout
        if type == "dataset":
            self.name_dataset = name_dataset

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

    def write_to_h5py_group(self):
        ...