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

    def read_from_h5py_group(self):
        ...

    def write_to_h5py_group(self):
        ...
