import numpy as np
from dask import array as da

class DataStore:
    def __init__(self, data=None, eventIds=None,
                 stepLengths=None, scan=None):
        assert len(data) == len(eventIds), \
            "lengths of data and event IDs must mutch!"
        if not stepLengths is None:
            assert sum(stepLengths)==len(eventIds), \
                "StepsLength need to add up to dataset length!"
            if scan is None:
                print"No information about event groups (steps) \
                    available!"
        self._eventIds = eventIds
        self.data = data
        self.stepLengths = stepLengths
        self.scan = scan
    
    @propery
    def eventIds(self):
        if isinstance(self._eventIds,da.Array):
            self._eventIds = self._eventIds.compute()
        elif callable(self._eventIds):
            self._eventIds = self._eventIds()
        return self._eventIds

    def __len__(self):
        return len(self.data)

def wrap4escData(func,convertOutput2EscData='auto'):
    def wrapped(*args,escSorter='first',**kwargs):
        argsIsEsc = [(n,arg) for n,arg in enumerate(args)\
                     if isinstance(arg,DataStore)]
        kwargsIsEsc = {key:kwarg
                       for key,kwarg in kwargs.items()\
                       if isinstance(kwarg,DataStore)}
        allEscs = [a for n,a in argsIsEsc]\
            .extend(kwargsIsEsc.values())
        if escSorter is 'first':
            if len(allEscs)>0:
                sorter = allEscs[0]
            else:
                sorter = None
                print("Did not find any DataStore instance \
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
                lip =
                for toutput in output:












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

