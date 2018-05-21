import numpy as np
from dask import array as da

class Array:
    def __init__(self, data=None, eventIds=None,
                 stepLengths=None, scan=None):
        assert len(data) == len(eventIds), \
            "lengths of data and event IDs must mutch!"
        if not stepLengths is None:
            assert sum(stepLengths)==len(eventIds), \
                "StepsLength need to add up to dataset length!"
            if scan is None:
                print("No information about event groups (steps) \
                    available!")
        self._eventIds = eventIds
        self.data = data
        self.stepLengths = stepLengths
        self.scan = scan
    
    @property
    def eventIds(self):
        if isinstance(self._eventIds,da.Array):
            self._eventIds = self._eventIds.compute()
        elif callable(self._eventIds):
            self._eventIds = self._eventIds()
        return self._eventIds

    def __len__(self):
        return len(self.data)

    def __getitem__(self,*args,**kwargs):
        return self.data.__getitem__(*args,**kwargs)


def wrap4escData(func,convertOutput2EscData='auto'):
    def wrapped(*args,escSorter='first',**kwargs):
        argsIsEsc = [(n,arg) for n,arg in enumerate(args)\
                     if isinstance(arg,Array)]
        kwargsIsEsc = {key:kwarg
                       for key,kwarg in kwargs.items()\
                       if isinstance(kwarg,Array)}
        allEscs = [a for n,a in argsIsEsc]\
            .extend(kwargsIsEsc.values())
        if escSorter is 'first':
            if len(allEscs)>0:
                sorter = allEscs[0]
            else:
                sorter = None
                print("Did not find any Array instance \
                      in input parameters!")
        else:
            sorter = escSorter
        if not sorter is None:
            ixsorter = allEscs.index(sorter)
            allEscs.pop(ixsorter)
            ixmaster,ixslaves,stepLengthsNew = matchIDs(sorter.eventIds,[t.eventIds
                                              for t in allEscs])
            ixslaves.insert(ixsorter,ixmaster)
            ids_res = sorter.eventIds[ixmaster]
            for n,arg in argsIsEsc:
                args.pop(n)
                args.insert(n,arg.data[ixslaves.pop[0]])
            for key,kwarg in kwargsIsEsc.items():
                kwargs.pop(key)
                kwargs[key] = kwarg.data[ixslaves.pop[0]]
        output = func(*args,**kwargs)
        if not convertOutput2EscData is False:
            if convertOutput2EscData is 'auto':
                lip = delme
                for toutput in output:
                    fff



class Scan:
    def __init__(self, parameter_names=None, values=[], readbacks=[], parameter_Ids=None, scan_step_info=[]):
        """
        """
        self._parameter_names = parameter_names
        self._parameter_Ids = parameter_Ids
        self._values = values
        self._readbacks = readbacks
        self._scan_step_info = scan_step_info

    def _append(self,values,readbacks,scan_step_info=None):
        assert len(values) == len(self._parameterNames), 'Not enough values supplied'
        assert len(readbacks) == len(self._parameterNames), 'Not enough values supplied'
        self._values.append(values)
        self._readbacks.append(readbacks)
        self._scan_step_info.append(scan_step_info)

    def keys(self):
        return self._parameter_names

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        if type(item) is slice or type(item) is int:
            return np.asarray(self._values).T[item]
        elif type(item) is str:
            return np.asarray(self._values).T[self._parameter_names.index(item)]








#def getMatchSortedIndices(ids_master,ids_slave):
    #"""Get indices that bring id integer numbers from a
    #slave dataset in order of those of a matching master id
    #number set"""
    #com = ids_master[np.in1d(ids_master,ids_slave,assume_unique=True)]
    #srt = ids_slave.argsort(axis=0)
    #return srt[np.searchsorted(ids_slave,com,sorter=srt)]



def matchIDs(ids_master,ids_slaves,stepLengths_master=None):
    ids_res = ids_master
    for tid in ids_slaves:
        ids_res = ids_res[np.in1d(ids_res,tid,assume_unique=True)]
    inds_slaves = []
    for tid in ids_slaves:
        srt = tid.argsort(axis=0)
        inds_slaves.append(srt[np.searchsorted(tid,ids_res,sorter=srt)])
    srt = ids_res.argsort(axis=0)
    inds_master = srt[np.searchsorted(ids_master,ids_res,sorter=srt)]

    if not stepLengths_master is None:
        stepLensNew = \
            np.bincount(
                np.digitize(inds_master,bins=np.cumsum(stepLengths_master)))
    else:
        stepLensNew = None
    return inds_master,inds_slaves,stepLensNew

