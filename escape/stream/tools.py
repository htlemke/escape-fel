from functools import partial as _partial


class Dict2obj:
    def __init__(self, dictionary):
        self.__dict__ = dictionary
