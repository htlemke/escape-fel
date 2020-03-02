import numpy as np
from dask import array as da
from dask.diagnostics import ProgressBar
import operator
from ..utilities import hist_asciicontrast, Hist_ascii
import logging
from itertools import chain
from numbers import Number
from functools import partial
import re

logger = logging.getLogger(__name__)


class ArraySelector:
    def __init__(self, arrayitem, dims=None):
        """ Container object for selecting array subsets in functions mapped on escape Arrays."""
        self.arrayitem = arrayitem
        self.dims = dims

    def __call__(self, sel):
        if max(self.dims) <= (len(sel) - 1):
            return self.arrayitem.__getitem__(tuple(sel[n] for n in self.dims))
        else:
            return self.arrayitem


class Array:
    def __init__(
        self,
        data=None,
        index=None,
        step_lengths=None,
        parameter=None,
        index_dim=None,
        name=None,
    ):
        if index_dim is None:
            logger.debug(
                "No event dimension event_dim defined,\
                    assuming 0th Dimension."
            )
            self.index_dim = 0
        else:
            self.index_dim = index_dim
        if not (callable(data) or callable(index)):
            assert data.shape[self.index_dim] == len(
                index
            ), "lengths of data and event IDs must match!"
        if not step_lengths is None:
            if not callable(index):
                assert sum(step_lengths) == len(
                    index
                ), "StepsLength need to add up to dataset length!"
            if parameter is None:
                logger.debug(
                    "No information about event groups (steps) \
                    available!"
                )
        self._index = index
        self._data = data
        self._data_selector = None
        self.scan = Scan(parameter, step_lengths, self)
        self.name = name
        self._add_methods()

    @property
    def index(self):
        if isinstance(self._index, da.Array):
            self._index = self._index.compute()
            self._index, self._data, self.step_lengths = get_unique_indexes(
                self._index, self.data, self.scan.step_lengths
            )
        elif callable(self._index):
            self._index = self._index()
            self._index, self._data, self.scan.step_lengths = get_unique_indexes(
                self._index, self.data, self.scan.step_lengths
            )
        return self._index

    @property
    def data(self):
        if callable(self._data):
            op = self._data(data_selector=self._data_selector)
            if len(op) == 2 and op[1] == "nopersist":
                self._add_methods(op[0])
                return op[0]
            else:
                self._data = op
                self._add_methods()
                return self._data
        else:
            return self._data

    def _add_methods(self, data=None):
        if data is None:
            data = self._data
        if isinstance(data, np.ndarray):
            for m in [
                "nansum",
                "nanmean",
                "nanstd",
                "sum",
                "mean",
                "std",
                "median",
                "percentile",
                "max",
                "min",
            ]:
                self.__dict__[m] = partial(
                    escaped(np.__dict__[m], convertOutput2EscData=[0]), self
                )
        elif isinstance(data, da.Array):
            for m in [
                "nanmean",
                "nansum",
                "nanstd",
                "sum",
                "mean",
                "std",
                "max",
                "min",
            ]:
                self.__dict__[m] = partial(
                    escaped(da.__dict__[m], convertOutput2EscData=[0]), self
                )

    def get_step_data(self, n):
        assert n >= 0, "Step index needs to be positive"
        if n == 0 and self.step_lengths is None:
            return self.data[:]
        assert not self.step_lengths is None, "No step sizes defined."
        assert n < len(self.step_lengths), f"Only {len(self.step_lengths)} steps"
        return self.data[sum(self.step_lengths[:n]) : sum(self.step_lengths[: (n + 1)])]

    # TODO: get_steps() method that returns arrays for step selection or an iterator over all.

    def step_data(self):
        """returns iterator over all steps"""
        n = 0
        while n < len(self.step_lengths):
            yield self.get_step_data(n)
            n += 1

    def update(self, array):
        """Update one escape array from another. Only array elements not existing  in the present array will be added to it."""
        pass

    def __len__(self):
        return len(self.index)

    def __getitem__(self, *args, **kwargs):

        # this is multi dimensional itemgetting
        if type(args[0]) is tuple:
            # expanding ellipses --> TODO: multiple ellipses possible?
            if Ellipsis in args[0]:
                rargs = list(args[0])
                elind = rargs.index(Ellipsis)
                rargs.pop(elind)
                eventsel = [ta for ta in rargs if ta]
                missing_dims = self.ndim - len(eventsel)
                for n in range(missing_dims):
                    rargs.insert(elind, slice(None, None, None))
                args = (tuple(rargs),)
            # get event selector for the event ID selection
            eventIx = -1
            for n, targ in enumerate(args[0]):
                if targ:
                    eventIx += 1
                if eventIx == self.index_dim:
                    break

            if type(args[0][eventIx]) is int:
                rargs = list(args[0])
                if rargs[eventIx] == -1:
                    rargs[eventIx] = slice(rargs[eventIx], None)
                else:
                    rargs[eventIx] = slice(rargs[eventIx], rargs[eventIx] + 1)
                args = (tuple(rargs),)
            events = args[0][eventIx]
            events_type = "eventIx"

        # Single dimension itemgetting, which is by default along
        # event dimension, raise error if inconsistent with data shape
        else:
            assert self.index_dim == 0, "requesting slice not along event dimension!"
            # making sure slices are taken in a way the event dimention is not squeezed away.
            if type(args[0]) is int:
                rargs = list(args)
                if rargs[0] == -1:
                    rargs[0] = slice(rargs[0], None)
                else:
                    rargs[0] = slice(rargs[0], rargs[0] + 1)
                args = tuple(rargs)
            events = args[0]
            events_type = "none"
            # expand all dimensions for potential use in derived functions

        if isinstance(events, slice):
            events = list(range(*events.indices(len(self))))
        elif isinstance(events, np.ndarray) and events.dtype == bool:
            events = events.nonzero()[0]
        elif isinstance(events, Array):
            inds_self, [inds_selector], dum = match_indexes(self.index, [events.index])
            events = inds_self[events.data[inds_selector].nonzero()[0]]
            if events_type == "eventIx":
                args[0][eventIx] = events
            elif events_type == "none":
                args = (events,)
            else:
                raise Exception(
                    "Issue in escape array getitem using another escape array!"
                )

        stepLengths, scan = get_scan_step_selections(
            events, self.scan.step_lengths, scan=self.scan
        )
        # Save indices for potential use in derived functions
        self._data_selector = args
        return Array(
            data=self.data.__getitem__(*args),
            index=self.index.__getitem__(events),
            step_lengths=stepLengths,
            parameter=scan.parameter,
            index_dim=self.index_dim,
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
        new_index_dim = axes.index(self.index_dim)
        return Array(
            data=self.data.transpose(*args),
            index=self.index,
            step_lengths=self.scan.step_lengths,
            parameter=self.scan.parameter,
            index_dim=new_index_dim,
        )

    @property
    def T(self):
        return self.transpose()

    def compute(self):
        with ProgressBar():
            return Array(
                data=self.data.compute(),
                index=self.index,
                step_lengths=self.scan.step_lengths,
                parameter=self.scan.parameter,
                index_dim=self.index_dim,
            )

    def map_index_blocks(
        self,
        foo,
        *args,
        chunks=None,
        drop_axis=None,
        new_axis=None,
        event_dim="same",
        **kwargs,
    ):
        """map a function which works for a chunk of the Array (events along index_dim). This is only really relevant for dask array array data."""

        # Getting chunks in the event dimension
        if event_dim == "same":
            event_dim = self.index_dim
        chunks_edim = self.data.chunks[self.index_dim]

        # making sure that chunks in other dimensions are "flat"
        shp = self.data.shape
        newchunks = []
        rechunk = False
        for dim, dimchunks in enumerate(self.data.chunks):
            if dim == self.index_dim:
                newchunks.append(dimchunks)
            else:
                rechunk = len(dimchunks) > 1 or rechunk
                newchunks.append((sum(dimchunks),))
        if rechunk:
            data = self.data.rechunk(tuple(newchunks))
        else:
            data = self.data
            newchunks = self.data.chunks

        # checking if any inputs are to be selected
        if any([isinstance(x, ArraySelector) for x in chain(args, kwargs.values())]):

            def get_data(data_selector=None):
                args_sel = [
                    ta if not isinstance(ta, ArraySelector) else ta(data_selector)
                    for ta in args
                ]
                kwargs_sel = {}
                for tk, tv in kwargs.items():
                    if isinstance(tv, ArraySelector):
                        kwargs_sel[tk] = tv(data_selector)
                    else:
                        kwargs_sel[tk] = tv
                return (
                    data.map_blocks(
                        foo,
                        *args_sel,
                        chunks=chunks,
                        drop_axis=drop_axis,
                        new_axis=new_axis,
                        **kwargs_sel,
                    ),
                    "nopersist",
                )

            return Array(
                data=get_data,
                index=self.index,
                step_lengths=self.step_lengths,
                parameter=self.scan.parameter,
                index_dim=event_dim,
            )
        else:
            return Array(
                data=data.map_blocks(
                    foo,
                    *args,
                    chunks=chunks,
                    drop_axis=drop_axis,
                    new_axis=new_axis,
                    **kwargs,
                ),
                index=self.index,
                step_lengths=self.scan.step_lengths,
                parameter=self.scan.parameter,
                index_dim=event_dim,
            )

    def store(self, parent_h5py=None, name=None, unit=None, **kwargs):
        """ a way to store data, especially expensively computed data, into a new file. 
        """
        if not hasattr(self, "h5"):
            self.h5 = ArrayH5Dataset(parent_h5py, name)
        with ProgressBar():
            self.h5.append(self.data, self.index, self.scan)
        self._data = self.h5.get_data_da()
        self._index = self.h5.index
        self.scan._save_to_h5(parent_h5py[name])
    
    def set_h5_storage(self,parent_h5py,name=None):
        if not hasattr(self, "h5"):
            if not name:
                name = self.name
            self.h5 = ArrayH5Dataset(parent_h5py, name)
        else:
            logger.info(f"h5 storage already set at {self.h5.name} in {self.h5.file.filename}")

        

    @classmethod
    def load_from_h5(cls, parent_h5py, name):
        h5 = ArrayH5Dataset(parent_h5py, name)
        parameter, step_lengths = Scan._load_from_h5(parent_h5py[name])
        return cls(
            index=h5.index,
            data=h5.get_data_da(),
            parameter=parameter,
            step_lengths=step_lengths,
            name=name,
        )

    def ones(self):
        return Array(
            data=np.ones(len(self)),
            index=self.index,
            step_lengths=self.scan.step_lengths,
            parameter=self.scan.parameter,
        )

    def _get_ana_str(self, perc_limits=[5, 95]):
        sqaxes = list(range(self.data.ndim))
        sqaxes.pop(self.index_dim)
        try:
            d = self.data.squeeze(axis=tuple(sqaxes))
        except:
            return ""
        if d.ndim == 1:
            ostr = ""
            hrange = np.percentile(d[~np.isnan(d)], perc_limits)
            formnum = lambda num: "{:<9}".format("%0.4g" % (num))
            for n, td in enumerate(self.scan):
                ostr += (
                    "Step %04d:" % n
                    + hist_asciicontrast(
                        td.data.squeeze(), bins=40, range=hrange, disprange=False
                    )
                    + "\n"
                )
            ho = Hist_ascii(d, range=hrange, bins=40)
            ostr += ho.horizontal()
            return ostr
        else:
            return ""

    def __repr__(self):
        s = "<%s.%s object at %s>" % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self)),
        )
        s += " {}; shape {}".format(self.name, self.shape)
        s += "\n"
        if isinstance(self.data, np.ndarray):
            s += self._get_ana_str()
        if self.scan:
            s += self.scan.__repr__()
        return s


def load_from_h5(parent_h5py, name):
    h5 = ArrayH5Dataset(parent_h5py, name)
    parameter, step_lengths = Scan._load_from_h5(parent_h5py)
    return Array(
        index=h5.index,
        data=h5.get_data_da(),
        parameter=parameter,
        step_lengths=step_lengths,
        name=name,
    )


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
            ixmaster, ixslaves, stepLengthsNew = match_indexes(
                sorter.index, [t.index for t in allEscs]
            )
            ixslaves.insert(ixsorter, ixmaster)
            ids_res = sorter.index[ixmaster]
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
                ixmaster, sorter.scan.step_lengths, scan=sorter.scan
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
                        index=ids_res,
                        step_lengths=stepLengths,
                        parameter=scan.parameter,
                        index_dim=0,
                    ),
                )

            if len(output) == 1:
                output = output[0]
            elif len(output) == 0:
                output = None
        return output

    return wrapped


def _scan_wrap(func, **default_kws):
    def wrapped(scan, **kwargs):
        default_kws.update(kwargs)
        return [func(step.data, **default_kws) for step in scan]

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
    def __init__(self, parameter={}, step_lengths=None, array=None):

        self.step_lengths = list(step_lengths)
        for par, pardict in parameter.items():
            if not len(pardict["values"]) == len(self):
                raise Exception(
                    f"Parameter array length of {par} does not fit the defined steps."
                )
        self.parameter = parameter
        self._array = array
        self._add_methods()

    def _add_methods(self):
        # if isinstance(data, np.ndarray):
        #     for m in [
        #         "nansum",
        #         "nanmean",
        #         "nanstd",
        #         "sum",
        #         "mean",
        #         "std",
        #         "median",
        #         "percentile",
        #         "max",
        #         "min",
        #     ]:
        #         self.__dict__[m] = partial(
        #             _scan_wrap(np.__dict__[m],axis=0), self
        #         )
        # elif isinstance(data, da.Array):
        for m in [
            "nanmean",
            "nansum",
            "nanstd",
            "sum",
            "mean",
            "std",
            "max",
            "min",
            # "count",
        ]:
            self.__dict__[m] = partial(_scan_wrap(da.__dict__[m], axis=0), self)

    def append_step(self, parameter, step_length):
        self.step_lengths.append(step_length)
        for par, pardict in parameter:
            self.parameter[par]["values"].append(pardict["values"])

    def __len__(self):
        return len(self.step_lengths)

    def __getitem__(self, sel):
        """array getter for scan"""
        if isinstance(sel, slice):
            sel = range(*sel.indices(len(self)))
        if isinstance(sel, Number):
            if sel < 0:
                sel = len(self) + sel
            return self.get_step_array(sel)
        else:
            return concatenate([self.get_step_array(n) for n in sel])

    def get_step_array(self, n):
        """array getter for scan"""
        assert n >= 0, "Step index needs to be positive"
        if n == 0 and self.step_lengths is None:
            data = self._array.data[:]
            index = self._array.index[:]
            step_lengths = self._array.step_lengths
            parameter = self._array.parameter

        # assert not self.step_lengths is None, "No step sizes defined."
        elif not n < len(self.step_lengths):
            raise IndexError(f"Only {len(self.step_lengths)} steps")
        else:
            data = self._array.data[
                sum(self.step_lengths[:n]) : sum(self.step_lengths[: (n + 1)])
            ]
            index = self._array.index[
                sum(self.step_lengths[:n]) : sum(self.step_lengths[: (n + 1)])
            ]
            step_lengths = [self.step_lengths[n]]
            parameter = {}
            for par_name, par in self.parameter.items():
                parameter[par_name] = {}
                parameter[par_name]["values"] = [par["values"][n]]
                if "attributes" in par.keys():
                    parameter[par_name]["attributes"] = par["attributes"]
        return Array(
            data=data, index=index, parameter=parameter, step_lengths=step_lengths
        )

    def _check_consistency(self):
        for par, pardict in self.parameter.items():
            if not len(self) == len(pardict["values"]):
                raise Exception(f"Scan length does not fit parameter {par}")

    def get_parameter_selection(self, selection):
        selection = np.atleast_1d(selection)
        if selection.dtype == bool:
            selection = selection.nonzero()[0]
        par_out = {}
        for par, pardict in self.parameter.items():
            par_out[par] = {}
            par_out[par]["values"] = [pardict["values"][i] for i in selection]
            if "attributes" in pardict.keys():
                par_out[par]["attributes"] = pardict["attributes"]
        return par_out

    def _save_to_h5(self, group):
        self._check_consistency()
        if "scan" in group.keys():
            del group["scan"]
        scan_group = group.require_group("scan")
        scan_group["step_lengths"] = self.step_lengths
        par_group = scan_group.require_group("parameter")
        for parname, pardict in self.parameter.items():
            tpg = par_group.require_group(parname)
            try:
                tpg["values"] = pardict["values"]
            except:
                tpg["values"] = [np.nan] * len(self)

            if "attributes" in pardict.keys():
                tpg.require_group("attributes")
                for attname, attvalue in pardict["attributes"].items():
                    tpg["attributes"][attname] = attvalue

    @staticmethod
    def _load_from_h5(group):
        if not "scan" in group.keys():
            raise Exception("Did not find group scan!")
        step_lengths = group["scan"]["step_lengths"][()]
        parameter = {}
        for parname, pargroup in group["scan"]["parameter"].items():
            values = pargroup["values"]
            if not len(values) == len(step_lengths):
                raise Exception(
                    f"The length of data array in {parname} parameter in scan does not fit!"
                )
            parameter[parname] = {}
            parameter[parname]["values"] = list(values[()])
            if "attributes" in pargroup.keys():
                parameter[parname]["attributes"] = {}
                for att_name, att_data in pargroup["attributes"].items():
                    parameter[parname]["attributes"][att_name] = att_data[()]
        return parameter, step_lengths


class Scan_old:
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


def to_dataframe(*args):
    for arg in args:
        if arg.data.shape > 1:
            raise (
                NotImplementedError("Only 1D Arrays can be converted to dataframes.")
            )


@escaped
def match_arrays(*args):
    return args


def concatenate(arraylist):
    data = da.concatenate([array.data for array in arraylist], axis=0)
    index = da.concatenate([array.index for array in arraylist])
    parameter = {}
    step_lengths = []

    for array in arraylist:
        if not parameter:
            parameter.update(array.scan.parameter)
        else:
            if not all(tk in parameter.keys() for tk in array.scan.parameter.keys()):
                raise Exception(
                    "Scans can not be concatenated due to mismatch in parameters!"
                )
            for par_name, par_dict in array.scan.parameter.items():
                parameter[par_name]["values"].extend(list(par_dict["values"]))
                if hasattr(par_dict, "attributes") and (
                    not parameter[par_name]["attributes"] == par_dict["attributes"]
                ):
                    raise Exception(
                        f"parameter attributes of {par_name} don't fit toghether in concatenated arrays."
                    )
        step_lengths.extend(list(array.scan.step_lengths))

    return Array(data=data, index=index, parameter=parameter, step_lengths=step_lengths)


def match_indexes(ids_master, ids_slaves, stepLengths_master=None):
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


def get_unique_indexes(index, array_data, stepLengths=None, delete_Ids=[0]):
    index, idxs = np.unique(index, return_index=True)
    good_Ids = np.ones_like(idxs, dtype=bool)
    for bad_Id in delete_Ids:
        good_Ids[index == bad_Id] = False
    index = index[good_Ids]
    idxs = idxs[good_Ids]
    if stepLengths:
        stepLengths = np.bincount(np.digitize(idxs, bins=np.cumsum(stepLengths)))
    return index, array_data[idxs], stepLengths


def get_scan_step_selections(ix, stepLengths, scan=None):
    ix = np.atleast_1d(ix)
    stepLengths = np.bincount(
        np.digitize(ix, bins=np.cumsum(stepLengths)), minlength=len(stepLengths)
    )
    stepLengths = stepLengths[~(stepLengths == 0)]
    if scan:
        validsteps = ~(stepLengths == 0)
        scan = Scan(
            parameter=scan.get_parameter_selection(validsteps), step_lengths=stepLengths
        )
    return stepLengths, scan


def escaped_FuncsOnEscArray(array, inst_funcs, *args, **kwargs):
    # TODO
    for inst, func in inst_funcs:
        if isinstance(array.data, inst):
            return escaped(func, *args, **kwargs)


def digitize(array, bins, foo=np.digitize, include_outlier_bins=False, sort_groups_by_index=True,**kwargs):
    """digitization function for escape arrays. checking for 1D arrays"""
    if not np.prod(np.asarray(array.shape)) == array.shape[array.index_dim]:
        raise NotImplementedError(
            "Only 1d escape arrays can be digitized in a sensible way."
        )
    darray = array.data.ravel()
    if include_outlier_bins:
        direction = np.sign(bins[-1]-bins[0])
        bins = np.concatenate([np.atleast_1d(direction*np.inf),bins,np.atleast_1d(direction*-np.inf)])
    inds = foo(darray, bins, **kwargs)
    ix = inds.argsort()[(0<inds)&(inds<len(bins))]
    bin_nos, counts = np.unique(inds[(0<inds)&(inds<len(bins))]-1, return_counts=True)
    if sort_groups_by_index:
        for n,bin_no in enumerate(bin_nos):
            tmn = sum(counts[:n])
            tmx = sum(counts[:n+1])
            tix = array.index[ix[tmn:tmx]].argsort()
            ix[tmn:tmx] = ix[tmn:tmx][tix]
    bin_left = bins[1:]
    bin_right = bins[:-1]
    bin_center = (bin_left+bin_right)/2
        
    parameter = {
        f"bin_center_{array.name}": {"values": bin_center[bin_nos], "attributes": kwargs},
        f"bin_left_{array.name}": {"values": bin_left[bin_nos], "attributes": kwargs},
        f"bin_right_{array.name}": {"values": bin_right[bin_nos], "attributes": kwargs},
    }

    return Array(
        data=array.data[ix],
        index=array.index[ix],
        parameter=parameter,
        step_lengths=counts,
    )


def filter(array, *args, foos_filtering=[operator.gt, operator.lt], **kwargs):
    """general filter function for escape arrays. checking for 1D arrays, applies arbitrary number of 
    filter functions that take one argument as input and """
    if not np.prod(np.asarray(array.shape)) == array.shape[array.index_dim]:
        raise NotImplementedError(
            "Only 1d escape arrays can be filtered in a sensible way."
        )
    darray = array.data
    if isinstance(darray, da.Array):
        print("filtering, i.e. downsizing of arrays requires to convert to numpy.")
        darray = darray.compute()
    # darray = array.data.ravel()
    ix = da.logical_and(
        *[tfoo(darray, targ) for tfoo, targ in zip(foos_filtering, args)]
    ).nonzero()[0]
    stepLengths, scan = get_scan_step_selections(ix, array.stepLengths, scan=array.scan)
    return Array(
        data=array.data[ix],
        index=array.index[ix],
        step_lengths=stepLengths,
        index_dim=array.index_dim,
        parameter=scan.parameter,
    )


def broadcast_to(ndarray_list, arraydef):
    if isinstance(arraydef, Array):
        index = arraydef.index
        step_lengths = arraydef.scan.step_lengths
        parameter = arraydef.scan.parameter
    if not len(ndarray_list) == len(step_lengths):
        raise Exception(
            "Cannot broadcast list of arrays that does not fit length of array scan"
        )
    data = []
    for ndarray, step_length in zip(ndarray_list, step_lengths):
        tbc = da.atleast_1d(ndarray)
        data.append(da.broadcast_to(tbc, [step_length] + list(tbc.shape)))
    data = da.concatenate(data, axis=0)
    return Array(data=data, index=index, parameter=parameter, step_lengths=step_lengths)


class ArrayH5Dataset:
    def __init__(self, parent, name):
        self.parent = parent
        self.grp = parent.require_group(name)
        self._data_finder = re.compile("^data_[0-9]{4}$")
        self._index_finder = re.compile("^index_[0-9]{4}$")
        self._n_d = []
        self._n_i = []
        for key in self.grp.keys():
            if self._data_finder.match(key):
                self._n_d.append(int(key[-4:]))
            if self._index_finder.match(key):
                self._n_i.append(int(key[-4:]))
        self._n_d.sort()
        self._n_i.sort()
        if not self._n_d == self._n_i:
            raise Exception(
                "Corrupt escape ArrayH5Dataset, not equal numbered data and id sub-datasets!"
            )

    @property
    def index(self):
        if self._n_i:
            return np.concatenate(
                [np.asarray(self.grp[f"index_{n:04d}"][:]) for n in self._n_i], axis=0
            )
        else:
            return np.asarray([], dtype=int)

    def append(self, data, event_ids, scan=None):
        ids_stored = self.index
        if len(event_ids) < len(ids_stored):
            raise Exception("fewer event_ids to append than already stored!")
        if not (event_ids[: len(ids_stored)] == ids_stored).all():
            raise Exception("new event_ids don't extend existing ones!")
        if len(event_ids) == len(ids_stored):
            print("Nothing new to append.")
            return
        n_new = len(self._n_i)
        self.grp[f"index_{n_new:04d}"] = event_ids[len(ids_stored) :]
        if isinstance(data, np.ndarray):
            self.grp[f"data_{n_new:04d}"] = data[len(ids_stored) :, ...]
        elif isinstance(data, da.Array):
            new_data = data[len(ids_stored) :, ...]
            # ToDo, smarter chunking when writing small data
            new_chunks = tuple(c[0] for c in new_data.chunks)
            dset = self.grp.create_dataset(
                f"data_{n_new:04d}",
                shape=new_data.shape,
                chunks=new_chunks,
                dtype=new_data.dtype,
            )
            da.store(new_data, dset)
        if scan:
            scan._save_to_h5(self.grp)
        self._n_i.append(n_new)
        self._n_d.append(n_new)

    def get_data_da(self, memlimit_MB=50):
        allarrays = []
        for n in self._n_i:
            ds = self.grp[f"data_{n:04d}"]
            if ds.chunks:
                chunk_size = list(ds.chunks)
            else:
                chunk_size = list(ds.shape)
            if chunk_size[0] == 1:
                size_element = (
                    np.dtype(ds.dtype).itemsize * np.prod(ds.shape[1:]) / 1024 ** 2
                )

                chunk_size[0] = int(memlimit_MB // size_element)

            allarrays.append(da.from_array(ds, chunks=chunk_size))

        return da.concatenate(allarrays)

    def create_array(self):
        return Array(data=self.get_data_da(), index=self.index)


# if 'data' in grp.keys():
# print(f'Dataset {name} already exists, data:')
# print(str(grp['data']))
# if input('Would you like to delete and overwrite the data ? (y/n)')=='y':
# del grp['data']
# del grp['event_ids']
# else:
# return
# grp['event_ids'] = self.index
# if isinstance(self.data,np.array):
# grp['data'] = self.data
# elif isinstance(self.data,da.array):
# dset = grp.create_dataset('data',shape=self.data.shape,chunks=self.data.chunks,dtype=self.data.dtype)
# self.data.store(dset)

