"""
Welcome to the

Event Synchronous Categorisation And Processing Environment,

a high level, object oriented module which abstracts event processing to
high level objects that can get updated from live data.
"""
from threading import Thread
import time
from .es_wrappers import EventHandler_SFEL
import numpy as np
from . import tools
import operator
from . import plots
import copy
import matplotlib.pyplot as plt
from collections import deque


from multiprocessing import Process
from .testStream import createStream


class TestStream:
    def __init__(self):
        self.test_stream_running = None

    def start(self):
        if self.test_stream_running:
            print("Stream already running")
            return

        self.test_stream_running = Process(target=createStream)
        self.test_stream_running.start()


test_stream = TestStream()


def initEscDataInstances():
    eventWorker = EventWorker()
    sources = eventWorker._eventHandler.getSourceIDs()
    out = dict()
    for sourceId in sources:
        out[sourceId] = EscData(source=EventSource(sourceId, eventWorker))
    eventWorker.startEventLoop()
    return tools.Dict2obj(out)


class Scan:
    def __init__(self, parameters=None, values=[], precision=dict(), sortValues=True):
        """
        parameters are EscDatainstances, that all should have a name.
        """
        self._parameters = parameters
        if self._parameters is None:
            self._parameterNames = None
            self._precision = None
            self._values = [None]
        else:
            self._parameterNames = [tp.name for tp in self._parameters]
            self._applyPrecision(precision)
            self._values = values
        self._sortValues = sortValues

    def _applyPrecision(self, precision=dict()):
        if isinstance(precision, np.ndarray):
            self._precision = precision
        else:
            self._precision = np.zeros(len(self._parameters))
            for key in precision.keys():
                if key in self._parameterNames:
                    self._precision[self._parameterNames.index(key)] = precision[key]

    def _roundParameters(self, parValues):
        pv = np.asarray(parValues)
        ind = self._precision.nonzero()[0]
        if len(ind) > 0:
            pv[ind] = np.round(pv[ind] / self._precision[ind]) * self._precision[ind]
        return tuple(pv)

    def _isValid(self):
        if self._parameters is None:
            return True
        parValues = [tp._getEventData() for tp in self._parameters]
        return not np.isnan(parValues).any()

    def _append(self):
        """Appends scan parameter values in case they don't exist already.
        returns a boolean "is to append" and the index of the data"""
        if self._parameters is None:
            return False, 0
        parValues = [tp._getEventData() for tp in self._parameters]
        if np.isnan(parValues).any():
            return None, None
        parValues = self._roundParameters(parValues)
        if parValues in self._values:
            return False, self._values.index(parValues)
        else:
            self._values.append(parValues)
            return True, len(self._values) - 1

    def keys(self):
        return self._parameterNames

    def copy(self):
        return Scan(
            parameters=self._parameters, values=self._values, precision=self._precision
        )

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        try:
            if type(item) is slice or type(item) is int:
                return np.asarray(self._values).T[item]
            elif type(item) is str:
                return np.asarray(self._values).T[self._parameterNames.index(item)]
        except:
            return []


class EscData:
    def __init__(self, source=None, dataManager=None, scan=Scan()):
        # source is an object of different types,
        # e.g. event Collector,
        # processing object,
        # indexed file source
        self._source = source
        self.unit = self._source.unit
        self.name = self._source.name
        self.scan = scan

        if dataManager is None:
            dataManager = DataManager(scan=scan)
        self._dataManager = dataManager
        self.data = self._dataManager.data
        self.eventIds = self._dataManager.eventIds
        self._accPassively = False
        self._isesc = True
        self._lastEventId = None

    def shape(self):
        return self._dataManager._get_shape()

    def lens(self):
        return self._dataManager.lens()

    def _getEventDataRaw(self):
        # eventId = self._source.eventWorker.event.getEventId()
        # if eventId == self._lastEventId:
        # return
        return self._source.getEventData()

    def __len__(self):
        return len(self._dataManager)

    def _getEventData(self):
        if self.scan._isValid():
            return self._getEventDataRaw()
        else:
            return None

    def _appendEventData(self):
        eventId = self._source.eventWorker.event.getEventId()
        if eventId == self._lastEventId:
            return
        data = self._getEventData()
        if not data is None:
            self._dataManager.append(data, eventId)
        self._lastEventId = eventId

    def _get_shape(self):
        pass

    def _update(self):
        self._appendEventData()

    def accumulate(self, do_accumulate=None):

        if do_accumulate is None:
            print("Toggling accumulation to", ["on", "off"][self._is_accumulating()])
            do_accumulate = not self._is_accumulating()

        if do_accumulate:
            self._source.eventWorker.eventCallbacks.append(self._appendEventData)
        else:
            try:
                i = self._source.eventWorker.eventCallbacks.index(self._appendEventData)
                self._source.eventWorker.eventCallbacks.pop(i)
            except:
                pass

    def _is_accumulating(self):
        return self._appendEventData in self._source.eventWorker.eventCallbacks

    def digitize(self, target, edges, side="left"):
        pass

    def categorizeBy(self, escdata_category, binning_def, side="left"):
        """
        Create new EscData instance of same data source, but catagorized
        according to another EscData instance (escdata_category),
        binned according to binning_def, which can be
        a)  a float providing a bin precision (the minimum binsize for continuous
            data). Will not predefine bins but create them according to
            incoming data
        b)  a list of binning edges used e.g. by np.digitize.
        """
        if type(binning_def) is float:
            s = Scan(
                parameters=[escdata_category],
                precision={escdata_category.name: binning_def},
            )
        elif np.iterable(binning_def):
            s = digitizeScan(escdata_category, binning_def)
        return EscData(source=self._source, scan=s)

    def digitize(self, binning_def, side="left"):
        """
        Create new EscData instance of same data source, but digitized
        according to bins of the own data, could be
        a)  a float providing a bin precision (the minimum binsize for continuous
            data). Will not predefine bins but create them according to
            incoming data
        b)  a list of binning edges used e.g. by np.digitize.
        """
        if type(binning_def) is float:
            s = Scan(parameters=[self], precision={self.name: binning_def})
        elif np.iterable(binning_def):
            s = digitizeScan(self, binning_def)
        return EscData(source=self._source, scan=s)

    def mean(self):
        return [np.mean(td, axis=0) for td in self.data]

    def std(self):
        return [np.std(td, axis=0) for td in self.data]

    def median(self):
        return [np.median(td, axis=0) for td in self.data]

    def centerPerc(self, perc=68.3):
        pervals = [50 - perc / 2.0, 50 + perc / 2.0]
        return [np.percentile(td, pervals, axis=0) for td in self.data]

    def plotHist(self, update=0.5, axes=None):
        self.accumulate(1)
        if axes is None:
            fig = plt.figure("%s histogram" % self.name)
            axes = fig.gca()
        self._histPlot = plots.HistPlot(self)
        self._histPlot.plot()
        if update:
            self._histPlot.updateContinuously(interval=update)

    def plotMed(self, update=0.5, axes=None):
        self.accumulate(1)
        if axes is None:
            fig = plt.figure("%s median" % self.name)
            axes = fig.gca()
        self._medPlot = plots.Plot(self)
        self._medPlot.plot()
        if update:
            self._medPlot.updateContinuously(interval=update)

    def plotCorr(self, xVar, Npoints=300, update=0.5, axes=None):
        self.accumulate(1)
        xVar.accumulate(1)
        if axes is None:
            fig = plt.figure("%s %s Correlation" % (self.name, xVar.name))
            axes = fig.gca()
        self._corrPlot = plots.PlotCorrelation(xVar, self, Nlast=Npoints)
        self._corrPlot.plot()
        if update:
            self._corrPlot.updateContinuously(interval=update)


# class SortedData:
# def __init__(self,data,sorter,issorted):
# self.data = data
# self.sorter = sorter
# self.issorted = issorted
# def __getitem__(self,item):
# self.data.__getitem(self.sorter().)


class DataManager:
    def __init__(self, data=None, eventIds=None, maxlen=1000, scan=Scan()):
        self.scan = scan
        if data is None and eventIds is None:
            self._data = [deque(maxlen=maxlen) for n in range(len(scan._values))]
            self._eventIds = [deque(maxlen=maxlen) for n in range(len(scan._values))]
        else:
            self._data = data
            self._eventIds = eventIds
        self._lastEventId = None

    def append(self, data, eventId, index=None):
        if eventId == self._lastEventId:
            return
        self._lastEventId = eventId
        if index is None:
            doappend, index = self.scan._append()
        if not doappend is None:
            if doappend:
                self._data.append([])
                self._eventIds.append([])
            self._data[index].append(data)
            self._eventIds[index].append(eventId)

    def _getDataShape(self):
        lens = self.lens()
        if max(lens) > 0:
            return np.shape(self.data[lens.index(max(lens))][0])
        else:
            return None

    def __len__(self):
        return len(self.data)

    def lens(self):
        le = [
            len(te) if len(te) == len(td) else print("Trouble in step %d" % n)
            for n, (te, td) in enumerate(zip(self._eventIds, self._data))
        ]
        return le

    def _get_data(self):
        return self._data

    def _get_eventIds(self):
        return self._eventIds

    data = property(_get_data)
    eventIds = property(_get_eventIds)


class EventSource:
    def __init__(self, sourceId, eventWorker, unit="a.u."):
        self.name = sourceId
        self.unit = unit
        self.eventWorker = eventWorker

    def getEventData(self):
        return self.eventWorker.event.getFromSource(self.name)


class ProcSource:
    def __init__(self, procObj, eventWorker, returnIndex=0, name=None, unit="a.u."):
        self.name = name
        self.unit = unit
        self.eventWorker = eventWorker
        self.procObj = procObj
        self.returnIndex = returnIndex

    def getEventData(self):
        if self.procObj.getEventData():
            self.procObj.updateChildren(self)
        return self.procObj.ret_values[self.returnIndex]


class FileSource:
    """Place holder for a file source with an indexed file,
    i.e. the functionality of ixppy"""

    pass


class EventWorker:
    def __init__(self, eventHandler=EventHandler_SFEL()):
        self._eventHandler = eventHandler
        self.eventCallbacks = []
        self.sources = []
        self.event = None
        self.initSourceFunc = eventHandler.registerSource
        self.eventGenerator = eventHandler.eventGenerator
        self._lastTime = time.time()
        self.runningFrequency = 0.0

    def registerSource(self, sourceID):
        self.initSourceFunc(sourceID)

    def eventLoop(self):
        for event in self.eventGenerator():
            self.event = event
            ttime = time.time()
            self.runningFrequency = 1 / (ttime - self._lastTime)
            self._lastTime = ttime

            for ecb in self.eventCallbacks:
                ecb()
                # event has getFromSource(sourcename) method
            if not self.running:
                break

    def startEventLoop(self):
        self.loopThread = Thread(target=self.eventLoop)
        self.loopThread.setDaemon(True)
        self.running = True
        self.loopThread.start()

    def stopEventLoop(self):
        self.running = False


class ProcObj:
    def __init__(
        self,
        func,
        args=[],
        kwargs=dict(),
        returns_is_esc=[True],
        returns_names=None,
        returns_units=None,
        objects=[],
        scan=None,
        scanIndex=None,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.objects = objects
        self.returns_is_esc = returns_is_esc
        self.returns_names = returns_names
        self.returns_units = returns_units
        self.args_is_esc = [isesc(targ) for targ in args]
        self.nonesc_returns = []
        self.kwargs_is_esc = {name: isesc(value) for name, value in kwargs.items()}

        self.scan = scan
        if scan is None:
            if not scanIndex is None:
                self.scan = self.getEscArgs()[scanIndex].scan.copy()
            else:
                scan = Scan()
        self.children = None
        self.eventWorker = None
        self._last_processed_eventId = None
        self.getEventWorker()

    def getEscArgs(self):
        escArgs = []
        for n in np.nonzero(self.args_is_esc)[0]:
            escArgs.append(self.args[n])

        return escArgs

    def getEscKwargs(self):
        escKwargs = dict()
        for key, isEsc in self.kwargs_is_esc.items():
            if isEsc:
                escKwargs[key] = self.kwargs[key]
        return escKwargs

    def getEventWorker(self):
        allEsc = self.getEscArgs() + list(self.getEscKwargs().values())
        allEventWorkers = [te._source.eventWorker for te in allEsc]
        assert all(
            x == allEventWorkers[0] for x in allEventWorkers
        ), "Problem with eventWorkers! seems there is more than one! "
        self.eventWorker = allEventWorkers[0]

    def createChildren(self):
        self.children = []
        for returnIndex, isesc in enumerate(self.returns_is_esc):
            if isesc:
                if self.returns_names is None:
                    name = "none"
                else:
                    name = self.returns_names[returnIndex]
                if self.returns_units is None:
                    unit = "none"
                else:
                    unit = self.returns_units[returnIndex]
                newscan = self.scan.copy()
                self.children.append(
                    EscData(
                        source=ProcSource(
                            self,
                            name=name,
                            unit=unit,
                            eventWorker=self.eventWorker,
                            returnIndex=returnIndex,
                        ),
                        scan=newscan,
                    )
                )
        return self.children

    def getEventData(self):
        tid = self.eventWorker.event.getEventId()
        if not self._last_processed_eventId == tid:
            args = [
                targ._getEventData() if aie else targ
                for targ, aie in zip(self.args, self.args_is_esc)
            ]
            kwargs = {
                key: tkwarg._getEventData() if self.kwargs_is_esc[key] else tkwarg
                for key, tkwarg in self.kwargs.items()
            }
            self._last_processed_eventId = tid
            if not any(
                [arg is None for arg in args] + [arg is None for arg in kwargs.values()]
            ):
                ret_values = self.func(*args, **kwargs)
                if not type(ret_values) is tuple:
                    ret_values = (ret_values,)
                self.ret_values = ret_values
            return True
        else:
            return False

    def updateChildren(self, caller):
        for child in self.children:
            if not child._source is caller:
                child._update()


def isesc(obj):
    try:
        return obj._isesc
    except:
        return False


def digitize(data, edges, side="left"):
    data = np.atleast_1d(data)
    edges = np.asarray(edges)
    assert (np.diff(edges) >= 0).all(), "edges must be monotonic, increasing"
    # edges = np.hstack([np.nan,np.asarray(edges),np.nan])
    indices = edges.searchsorted(data, side=side)
    indout = np.logical_or(indices == 0, indices == len(edges))
    edgelower = np.nan * np.ones_like(data)
    edgeupper = np.nan * np.ones_like(data)
    edgelower[~indout] = edges[indices[~indout] - 1]
    edgeupper[~indout] = edges[indices[~indout]]
    bincenter = (edgeupper + edgelower) / 2.0

    return np.squeeze(bincenter), np.squeeze(edgelower), np.squeeze(edgeupper)


def digitizeEsc(escdata, edges, side="left"):
    po = ProcObj(
        digitize,
        args=[escdata, edges],
        returns_is_esc=[True, True, True],
        returns_names=[
            "%s_bincenter" % escdata.name,
            "%s_edgelower" % escdata.name,
            "%s_edgeupper" % escdata.name,
        ],
        returns_units=[escdata.unit] * 3,
        scan=escdata.scan,
    )
    return po.createChildren()


def digitizeScan(escdata, edges, side="left"):
    escdats = digitizeEsc(escdata, edges, side=side)
    values = [
        (sum(edges[n : n + 2]) / 2.0, edges[n], edges[n + 1])
        for n in range(len(edges) - 1)
    ]
    return Scan(escdats, values=values)


def wrapFunc_singleOutput(func, name=None, unit=None, scan=None):
    if name is None:
        name = "none"
    if unit is None:
        unit = "none"

    def newFunc(*args, **kwargs):
        p = ProcObj(
            func,
            args=args,
            returns_is_esc=[True],
            returns_names=[name],
            returns_units=[unit],
            scan=Scan(None),
        )
        return p.createChildren()[0]

    return newFunc


def _wrapOperatorJoin(func, symbol):
    def newFunc(*args):
        names = [
            arg.name if hasattr(arg, "name") else type(arg).__name__ for arg in args
        ]
        return_name = (" %s " % symbol).join(names).join(["(", ")"])
        units = [arg.unit if hasattr(arg, "unit") else "no unit" for arg in args]
        return_unit = (" %s " % symbol).join(units).join(["(", ")"])

        p = ProcObj(
            func,
            args=args,
            returns_is_esc=[True],
            returns_names=[return_name],
            returns_units=[return_unit],
            scan=args[0].scan,
        )
        return p.createChildren()[0]

    return newFunc


def _wrapOperatorSingle(func, symbol):
    def newFunc(*args):
        name = args[0].name
        return_name = ("%s %s" % (symbol, name)).join(["(", ")"])
        units = [arg.unit for arg in args]
        return_unit = ("%s %s" % (symbol, name)).join(units).join(["(", ")"])

        p = ProcObj(
            func,
            args=args,
            returns_is_esc=[True],
            returns_names=[return_name],
            returns_units=[return_unit],
            scan=args[0].scan,
        )
        return p.createChildren()[0]

    return newFunc


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
    setattr(EscData, "__%s__" % opJoin.__name__, _wrapOperatorJoin(opJoin, symbol))
for opSing, symbol in _operatorsSingle:
    setattr(EscData, "__%s__" % opSing.__name__, _wrapOperatorSingle(opSing, symbol))
