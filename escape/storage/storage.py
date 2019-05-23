import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
import operator
from ..utilities import hist_asciicontrast, Hist_ascii



class Array:
    def __init__(
        self,
        data=None,
        eventIds=None,
        stepLengths=None,
        scan=None,
        eventDim=None,
        name=None,
    ):
        if eventDim is None:
            print(
                "No event dimension eventDim defined,\
                    assuming 0th Dimension."
            )
            self.eventDim = 0
        else:
            self.eventDim = eventDim
        if not (callable(data) or callable(eventIds)):
            assert data.shape[self.eventDim] == len(
                eventIds
            ), "lengths of data and event IDs must mutch!"
        if not stepLengths is None:
            if not callable(eventIds):
                assert sum(stepLengths) == len(
                    eventIds
                ), "StepsLength need to add up to dataset length!"
            if scan is None:
                print(
                    "No information about event groups (steps) \
                    available!"
                )
        self._eventIds = eventIds
        self._data = data
        self.stepLengths = stepLengths
        self.scan = scan
        self.name = name
        self._add_methods()

    @property
    def eventIds(self):
        if isinstance(self._eventIds, da.Array):
            self._eventIds = self._eventIds.compute()
            self._eventIds, self._data, self.stepLengths = get_unique_Ids(
                self._eventIds, self.data, self.stepLengths
            )
        elif callable(self._eventIds):
            self._eventIds = self._eventIds()
            self._eventIds, self._data, self.stepLengths = get_unique_Ids(
                self._eventIds, self.data, self.stepLengths
            )
        return self._eventIds

    @property
    def data(self):
        if callable(self._data):
            self._data = self._data()
            self._add_methods()
        return self._data
    
    def _add_methods(self):
        if isinstance(self.data,np.ndarray):
            for m in ['mean','std','median','percentile','max','min']:
                self.__dict__[m] = escaped(np.__dict__[m],convertOutput2EscData=[0])
        elif isinstance(self.data,da.Array):
            for m in ['mean','std','max','min']:
                self.__dict__[m] = escaped(da.__dict__[m],convertOutput2EscData=[0])


    def get_step_data(self, n):
        assert n >= 0, "Step index needs to be positive"
        if n == 0 and self.stepLengths is None:
            return self.data[:]
        assert not self.stepLengths is None, "No step sizes defined."
        assert n < len(self.stepLengths), f"Only {len(self.stepLengths)} steps"
        return self.data[sum(self.stepLengths[:n]) : sum(self.stepLengths[: (n + 1)])]

    def step_data(self):
        """returns iterator over all steps"""
        n = 0
        while n < len(self.stepLengths):
            yield self.get_step_data(n)
            n += 1

    def __len__(self):
        return len(self.eventIds)

    def __getitem__(self, *args, **kwargs):
        if type(args[0]) is tuple:
            # this is multi dimensional itemgetting
            
            #expand ellipses
            if Ellipsis in args[0]:
                rargs = list(args[0])
                elind = rargs.index(Ellipsis)
                rargs.pop(elind)
                eventsel = [ta for ta in rargs if ta]
                missing_dims = self.ndim-len(eventsel)
                for n in range(missing_dims):
                    rargs.insert(elind,slice(None,None,None))
                args = (tuple(rargs),)
            #get event selector
            eventIx = -1
            for n,targ in enumerate(args[0]):
                if targ:
                    eventIx += 1
                if eventIx == self.eventDim:
                    break
            if type(args[0][eventIx]) is int:
                rargs = list(args[0])
                if rargs[eventIx] == -1:
                    rargs[eventIx] = slice(rargs[eventIx], None)
                else:
                    rargs[eventIx] = slice(rargs[eventIx], rargs[eventIx] + 1)
                args = (tuple(rargs),)
            events = args[0][eventIx]

        else:
            # this is single dimension itemgetting, which is by default along
            # event dimension, raise error if inconsistent with data shape
            assert self.eventDim == 0, "requesting slice not along event dimension!"
            if type(args[0]) is int:
                rargs = list(args)
                if rargs[0] == -1:
                    rargs[0] = slice(rargs[0], None)
                else:
                    rargs[0] = slice(rargs[0], rargs[0] + 1)
                args = tuple(rargs)
            events = args[0]
        if isinstance(events, slice):
            events = list(range(*events.indices(len(self))))
        elif isinstance(events, np.ndarray) and events.dtype==bool:
            events = events.nonzero()[0]
        stepLengths, scan = get_scan_step_selections(
            events, self.stepLengths, scan=self.scan
        )
        return Array(
            data=self.data.__getitem__(*args),
            eventIds=self.eventIds.__getitem__(events),
            stepLengths=stepLengths,
            scan=scan,
            eventDim=self.eventDim,
        )

    @property
    def shape(self, *args, **kwargs):
        return self.data.shape

    @property
    def ndim(self, *args, **kwargs):
        return self.data.ndim

    def transpose(self, *args):
        if not args:
            axes = tuple(range(self.ndim - 1, -1, -1))
        elif len(args) == 1:
            axes = args[0]
        else:
            axes = args
        neweventDim = axes.index(self.eventDim)
        return Array(
            data=self.data.transpose(*args),
            eventIds=self.eventIds,
            stepLengths=self.stepLengths,
            scan=self.scan,
            eventDim=neweventDim,
        )

    @property
    def T(self):
        return self.transpose()

    def compute(self):
        with ProgressBar():
            return Array(
                data=self.data.compute(),
                eventIds=self.eventIds,
                stepLengths=self.stepLengths,
                scan=self.scan,
                eventDim=self.eventDim,
            )

    def map_blocks(self,*args,event_dim='same',**kwargs):
        if event_dim=='same':
            event_dim = self.eventDim
        return Array(
            data=self.data.map_blocks(*args,**kwargs),
            eventIds=self.eventIds,
            stepLengths=self.stepLengths,
            scan=self.scan,
            eventDim=event_dim,
        )

    def _get_ana_str(self,perc_limits=[5,95]):
        sqaxes = list(range(self.data.ndim))
        sqaxes.pop(self.eventDim)
        try:
            d = self.data.squeeze(axis=tuple(sqaxes))
        except:
            return ''
        if d.ndim==1:
            ostr = ''
            hrange = np.percentile(d[~np.isnan(d)],perc_limits)
            formnum = lambda num: '{:<9}'.format('%0.4g' %(num))
            for n,td in enumerate(self.step_data()):
                ostr+='Step %04d:'%n + hist_asciicontrast(td.squeeze(),bins=40,range=hrange,disprange=False) +'\n'
            ho = Hist_ascii(d,range=hrange,bins=40)
            ostr+=ho.horizontal()
            return ostr
        else:
            return ''

    def __repr__(self):
        s = "<%s.%s object at %s>" % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self)),
        )
        s += " {}; shape {}".format(self.name, self.shape)
        s += "\n"
        if isinstance(self.data,np.ndarray):
            s += self._get_ana_str()
        if self.scan:
            s += self.scan.__repr__()
        return s


def escaped(func, convertOutput2EscData="auto"):
    def wrapped(
        *args, escSorter="first", convertOutput2EscData=convertOutput2EscData, **kwargs
    ):
        args = [ta for ta in args]
        kwargs = {tk: tv for tk, tv in kwargs.items()}
        argsIsEsc = [(n, arg) for n, arg in enumerate(args) if isinstance(arg, Array)]
        kwargsIsEsc = {
            key: kwarg for key, kwarg in kwargs.items() if isinstance(kwarg, Array)
        }
        allEscs = [a for n, a in argsIsEsc]
        allEscs.extend(kwargsIsEsc.values())
        if escSorter is "first":
            if len(allEscs) > 0:
                sorter = allEscs[0]
            else:
                sorter = None
                print(
                    "Did not find any Array instance \
                      in input parameters!"
                )
        else:
            sorter = escSorter
        if not sorter is None:
            ixsorter = allEscs.index(sorter)
            allEscs.pop(ixsorter)
            ixmaster, ixslaves, stepLengthsNew = matchIDs(
                sorter.eventIds, [t.eventIds for t in allEscs]
            )
            ixslaves.insert(ixsorter, ixmaster)
            ids_res = sorter.eventIds[ixmaster]
            for n, arg in argsIsEsc:
                args.pop(n)
                args.insert(n, arg.data[ixslaves.pop(0)])
            for key, kwarg in kwargsIsEsc.items():
                kwargs.pop(key)
                kwargs[key] = kwarg.data[ixslaves.pop(0)]
        output = func(*args, **kwargs)
        if not type(output) is tuple:
            output = (output,)
        output = list(output)
        if convertOutput2EscData:
            stepLengths, scan = get_scan_step_selections(
                ixmaster, sorter.stepLengths, scan=sorter.scan
            )
            if convertOutput2EscData is "auto":
                convertOutput2EscData = []
                for i, toutput in enumerate(output):
                    if hasattr(toutput, "__len__") and len(ids_res) == len(toutput):
                        convertOutput2EscData.append(i)

            for n in convertOutput2EscData:
                toutput = output.pop(n)
                output.insert(
                    n,
                    Array(
                        data=toutput,
                        eventIds=ids_res,
                        stepLengths=stepLengths,
                        scan=scan,
                        eventDim=0,
                    ),
                )

            if len(output) == 1:
                output = output[0]
            elif len(output) == 0:
                output = None
        return output

    return wrapped


_operatorsJoin = [
    (operator.add, "+"),
    (operator.contains, "in"),
    (operator.truediv, "/"),
    (operator.floordiv, "//"),
    (operator.and_, "&"),
    (operator.xor, "^"),
    (operator.or_, "|"),
    (operator.pow, "**"),
    (operator.is_, "is"),
    (operator.is_not, "is not"),
    (operator.lshift, "<<"),
    (operator.mod, "%"),
    (operator.mul, "*"),
    (operator.rshift, ">>"),
    (operator.sub, "-"),
    (operator.lt, "<"),
    (operator.le, "<="),
    (operator.eq, "=="),
    (operator.ne, "!="),
    (operator.ge, ">="),
    (operator.gt, ">"),
]


_operatorsSingle = [
    (operator.invert, "~"),
    (operator.neg, "-"),
    (operator.not_, "not"),
    (operator.pos, "pos"),
]

for opJoin, symbol in _operatorsJoin:
    setattr(
        Array, "__%s__" % opJoin.__name__, escaped(opJoin, convertOutput2EscData=[0])
    )

for opSing, symbol in _operatorsSingle:
    setattr(
        Array, "__%s__" % opSing.__name__, escaped(opSing, convertOutput2EscData=[0])
    )


class Scan:
    def __init__(
        self,
        parameter_names=None,
        values=None,
        parameter_attrs=None,
        scan_step_info=None,
    ):
        """
        """
        self._parameter_names = parameter_names
        self._parameter_attrs = {}
        for name, attr in parameter_attrs.items():
            assert (
                str(name) in self._parameter_names
            ), f"Attribute {name} not in scan parameter names."
            self._parameter_attrs[str(name)] = attr
        if values is None:
            values = []
        self._values = values
        if scan_step_info is None:
            scan_step_info = []
        self._step_info = scan_step_info

    def _append(self, values, scan_step_info=None):
        assert len(values) == len(
            self._parameter_names
        ), "number of values doesn't fit no of parameter names"
        self._values.append(values)
        self._step_info.append(scan_step_info)

    def keys(self):
        return self._parameter_names

    def get_steps(self, selection):
        selection = np.atleast_1d(selection)
        if selection.dtype == bool:
            selection = selection.nonzero()[0]
        return {
            "parameter_names": self._parameter_names,
            "parameter_attrs": self._parameter_attrs,
            "values": [self._values[i] for i in selection],
            "scan_step_info": [self._step_info[i] for i in selection],
        }

    def __len(self):
        return len(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        if type(item) is slice or type(item) is int:
            return np.asarray(self._values).T[item]
        elif type(item) is str:
            return np.asarray(self._values).T[self._parameter_names.index(item)]

    def __repr__(self):
        s = "Scan over {} steps".format(len(self))
        s += "\n"
        s += "Parameters {}".format(", ".join(self._parameter_names))
        return s

#<<<<<<< HEAD
# def to_dataframe(*args):
#     for arg in args:
#         if arg.data.shape>1:
#             raise(NotImplementedError('Only 1D Arrays can be converted to dataframes.'))
    
#     from dask import dataframe as dd
#     # dd.concat()

@escaped
def matchArrays(*args):
    return args


def matchIDs(ids_master, ids_slaves, stepLengths_master=None):
    ids_res = ids_master
    for tid in ids_slaves:
        ids_res = ids_res[np.in1d(ids_res, tid, assume_unique=True)]
    inds_slaves = []
    for tid in ids_slaves:
        srt = tid.argsort(axis=0)
        inds_slaves.append(srt[np.searchsorted(tid, ids_res, sorter=srt)])
    srt = ids_master.argsort(axis=0)
    inds_master = srt[np.searchsorted(ids_master, ids_res, sorter=srt)]

    if not stepLengths_master is None:
        stepLensNew = np.bincount(
            np.digitize(inds_master, bins=np.cumsum(stepLengths_master))
        )
    else:
        stepLensNew = None
    return inds_master, inds_slaves, stepLensNew


def get_unique_Ids(eventIds, array_data, stepLengths=None, delete_Ids=[0]):
    eventIds, idxs = np.unique(eventIds, return_index=True)
    good_Ids = np.ones_like(idxs, dtype=bool)
    for bad_Id in delete_Ids:
        good_Ids[eventIds == bad_Id] = False
    eventIds = eventIds[good_Ids]
    idxs = idxs[good_Ids]
    if stepLengths:
        stepLengths = np.bincount(np.digitize(idxs, bins=np.cumsum(stepLengths)))
    return eventIds, array_data[idxs], stepLengths


def get_scan_step_selections(ix, stepLengths, scan=None):
    ix = np.atleast_1d(ix)
    stepLengths = np.bincount(
        np.digitize(ix, bins=np.cumsum(stepLengths)), minlength=len(stepLengths)
    )
    if scan:
        validsteps = ~(stepLengths == 0)
        scan = Scan(**scan.get_steps(validsteps))
    stepLengths = stepLengths[~(stepLengths == 0)]
    return stepLengths, scan

def escaped_FuncsOnEscArray(array,inst_funcs,*args,**kwargs):
    for inst,func in inst_funcs:
        if isinstance(array.data, inst):
            return escaped(func,*args,**kwargs)

def digitize(array,*args,foo=da.digitize,**kwargs):
    """ digitize array data"""
    bins = foo(array.data,*args,**kwargs)



