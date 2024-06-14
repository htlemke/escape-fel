import io
import numpy as np
import dask
from dask import array as da
from dask import dataframe as ddf
from dask.diagnostics import ProgressBar
import operator

from escape.storage.source import Source
from ..utilities import get_corr, hist_asciicontrast, Hist_ascii, plot2D
import logging
from itertools import chain
from numbers import Number
import re
from .. import utilities
import h5py
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import html
import base64
from io import BytesIO
from .storage_tools import ArrayTools, ScanTools


logger = logging.getLogger(__name__)

import escape


class ArraySelector:
    def __init__(self, arrayitem, dims=None):
        """Container object for selecting array subsets in functions mapped on escape Arrays."""
        self.arrayitem = arrayitem
        self.dims = dims

    def __call__(self, sel):
        if max(self.dims) <= (len(sel) - 1):
            return self.arrayitem.__getitem__(tuple(sel[n] for n in self.dims))
        else:
            return self.arrayitem


def _apply_method(
    foo_np,
    foo_da,
    data,
    is_dask_array,
    *args,
    convertesc_axis_kw=False,
    convertOutput2EscData="auto",
    **kwargs,
):
    if convertesc_axis_kw:
        axis = kwargs.get("axis", "noaxis")
        if isinstance(axis, Number):
            axis = [axis]
        if (axis == "noaxis") or (0 in axis):
            convertOutput2EscData = []

    if is_dask_array:
        if not foo_da:
            raise NotImplementedError(
                f"Function {foo_np.__name__} is not defined for dask based arrays!"
            )
        return escaped(foo_da, convertOutput2EscData=convertOutput2EscData)(
            data, *args, **kwargs
        )
    else:
        if not foo_np:
            raise NotImplementedError(
                f"Function {foo_da.__name__} is not defined for numpy based arrays!"
            )
        return escaped(foo_np, convertOutput2EscData=convertOutput2EscData)(
            data, *args, **kwargs
        )


class Array:
    def __init__(
        self,
        data=None,
        index=None,
        step_lengths=None,
        parameter=None,
        index_dim=None,
        name=None,
        source=None,
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
        else:
            step_lengths = [len(index)]
        self._index = index
        self._data = data
        self._data_selector = None

        self.scan = Scan(parameter, step_lengths, self)
        self.name = name
        self.source = source
        self._touched = False
        self.tools = ArrayTools(self)

    def _touch(self):
        if not self._touched:
            dum = self.index

    @property
    def index(self):
        if isinstance(self._index, da.Array):
            self._index = self._index.compute()
            self._index, self._data, self.scan.step_lengths = get_unique_indexes(
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
        self._touch()
        # TODO: try getting the properties outside of storage in the
        # specific parser section
        if callable(self._data):
            # TODO: cludgy solution need fix at some point.
            op = self._data(data_selector=self._data_selector)
            if len(op) == 2 and (type(op[1]) is str) and op[1] == "nopersist":
                return op[0]
            else:
                self._data = op
                return self._data
        else:
            return self._data

    def is_dask_array(self):
        return isinstance(self.data, da.Array)

    def nansum(self, *args, **kwargs):
        return _apply_method(
            np.nansum,
            da.nansum,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def nanmean(self, *args, **kwargs):
        return _apply_method(
            np.nanmean,
            da.nanmean,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def nanstd(self, *args, **kwargs):
        return _apply_method(
            np.nanstd,
            da.nanstd,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def nanmedian(self, *args, **kwargs):
        return _apply_method(
            np.nanmedian,
            None,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def nanmin(self, *args, **kwargs):
        return _apply_method(
            np.nanmin,
            da.nanmin,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def nanmax(self, *args, **kwargs):
        return _apply_method(
            np.nanmax,
            da.nanmax,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def sum(self, *args, **kwargs):
        return _apply_method(
            np.sum,
            da.sum,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def isnan(self, *args, **kwargs):
        return _apply_method(
            np.isnan,
            da.isnan,
            self,
            self.is_dask_array(),
            *args,
            convertOutput2EscData=[0],
            **kwargs,
        )

    def mean(self, *args, **kwargs):
        return _apply_method(
            np.mean,
            da.mean,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def average(self, *args, **kwargs):
        return _apply_method(
            np.average,
            da.average,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def std(self, *args, **kwargs):
        return _apply_method(
            np.std,
            da.std,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def median(self, *args, **kwargs):
        return _apply_method(
            np.median,
            None,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def min(self, *args, **kwargs):
        return _apply_method(
            np.min,
            da.min,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def max(self, *args, **kwargs):
        return _apply_method(
            np.max,
            da.max,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def all(self, *args, **kwargs):
        return _apply_method(
            np.all,
            da.all,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def any(self, *args, **kwargs):
        return _apply_method(
            np.any,
            da.any,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def abs(self, *args, **kwargs):
        return _apply_method(
            np.abs,
            da.abs,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def percentile(self, *args, **kwargs):
        return _apply_method(
            np.percentile,
            None,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def quantile(self, *args, **kwargs):
        return _apply_method(
            np.quantile,
            None,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=False,
            **kwargs,
        )

    def nanpercentile(self, *args, **kwargs):
        return _apply_method(
            np.nanpercentile,
            None,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=True,
            **kwargs,
        )

    def nanquantile(self, *args, **kwargs):
        return _apply_method(
            np.nanquantile,
            None,
            self,
            self.is_dask_array(),
            *args,
            convertesc_axis_kw=True,
            **kwargs,
        )

    def filter(self, *args, **kwargs):
        return filter(self, *args, **kwargs)

    def digitize(self, bins, **kwargs):
        return digitize(self, bins, **kwargs)

    def get_modulo_array(self, mod, offset=0):
        index = self.index
        out_bool = np.mod(index, mod) == offset
        return self[out_bool]

    def update(self, array):
        """Update one escape array from another. Only array elements not
        existing  in the present array will be added to it."""
        pass

    def correlation_analysis_to(self, ref, order=2):
        td, tr = match_arrays(self, ref)
        std_rel, std_fx_rel = get_corr(td.data, tr.data, order=order)
        return std_rel, std_fx_rel

    def __len__(self):
        self._touch()
        return len(self.index)

    def categorize(self, other_array):
        return match_arrays(self, other_array)[1]

    def __getitem__(self, *args, **kwargs):
        # this is multi dimensional itemgetting
        if type(args[0]) is tuple:
            # expanding ellipses --> TODO: multiple ellipses possible?
            if Ellipsis in [type(ta) for ta in args[0]]:
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

    def get_random_events(self, n, seed=None):
        np.random.seed(seed)
        inds = np.random.randint(0, len(self), size=(n,))
        return self[list(inds)]

    @property
    def shape(self, *args, **kwargs):
        self._touch()
        return self.data.shape

    @property
    def ndim(self, *args, **kwargs):
        return self.data.ndim

    @property
    def ndim_nonzero(self, *args, **kwargs):
        return len(np.asarray(self.shape)[np.nonzero(self.shape)[0]])

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

    def ravel_event_data(self, *args):
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

    def compute(self, **kwargs):
        if self.is_dask_array():
            with ProgressBar():
                return Array(
                    data=self.data.compute(**kwargs),
                    index=self.index,
                    step_lengths=self.scan.step_lengths,
                    parameter=self.scan.parameter,
                    index_dim=self.index_dim,
                )
        else:
            if self.name:
                print(f"No `compute` necessary for {self.name}")
            else:
                print(f"No `compute` necessary")
            return self

    def persist(self):
        self.data.persist()

    # def get_progress()

    def map_index_blocks(
        self,
        foo,
        *args,
        # chunks=None,
        drop_axis=None,
        new_axis=None,
        new_element_size=None,
        event_dim="same",
        **kwargs,
    ):
        """map a function which works for a chunk of the Array
        (events along index_dim). This is only really relevant for dask array array data.
        """

        # Test: creating Source instance for origin tracking
        src = Source(
            "factory",
            factory=foo,
            args=args,
            kwargs=kwargs,
            iargout=0,
            name_dataset=self.name,
        )

        # Getting chunks in the event dimension
        if event_dim == "same":
            event_dim = self.index_dim
        chunks_edim = self.data.chunks[self.index_dim]

        # making sure that chunks in other dimensions are "flat"
        shp = self.data.shape
        if new_element_size:
            new_size = list(new_element_size)
            new_size.insert(self.index_dim, None)
        newchunks = []
        rechunk = False
        for dim, dimchunks in enumerate(self.data.chunks):
            if dim == self.index_dim:
                newchunks.append(dimchunks)
            elif drop_axis and dim in drop_axis:
                continue
            else:
                rechunk = len(dimchunks) > 1 or rechunk
                if new_element_size:
                    newchunks.append((new_size[dim],))
                else:
                    newchunks.append((sum(dimchunks),))
        if rechunk:
            data = self.data.rechunk(tuple(newchunks))
        else:
            data = self.data

        if new_element_size:
            chunks = newchunks
        else:
            chunks = None

        # checking if any inputs are to be selected
        if any([isinstance(x, ArraySelector) for x in chain(args, kwargs.values())]):
            print("Is arg selector")

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

    def store(self, parent_h5py=None, name=None, unit=None, lock="auto", **kwargs):
        """a way to store data, especially expensively computed data, into a new file."""
        if lock == "auto":
            lock = get_lock()
        if not hasattr(self, "h5"):
            self.h5 = ArrayH5Dataset(parent_h5py, name)

        with ProgressBar():
            self.h5.append(self.data, self.index, self.scan, lock=lock, **kwargs)
        self._data = self.h5.get_data_da()
        self._index = self.h5.index
        self.scan._save_to_h5(self.h5.grp)

    def set_h5_storage(self, parent_h5py, name=None):
        if not hasattr(self, "h5"):
            if not name:
                name = self.name
            self.h5 = ArrayH5Dataset(parent_h5py, name)
        else:
            try:
                logger.info(
                    f"h5 storage already set at {name} in {self.h5.file.filename}"
                )
            except:
                logger.info(f"h5 storage already set for {name}")

    def store_file(self, parent_h5py=None, name=None, unit=None, **kwargs):
        """a way to store data, especially expensively computed data, into a new file."""
        if not hasattr(self, "h5"):
            self.h5 = ArrayH5Dataset(parent_h5py, name)

        with ProgressBar():
            self.h5.append(self.data, self.index, self.scan)
        self._data = self.h5.get_data_da()
        self._index = self.h5.index

    def set_h5_storage_file(self, file_name, parent_group_name, name=None):
        if not hasattr(self, "h5"):
            if not name:
                name = self.name
            self.h5 = ArrayH5File(file_name, parent_group_name, name)
        else:
            logger.info(f"h5 storage already set at {name} in {self.h5.file_name}")

    @classmethod
    def load_from_h5(cls, parent_h5py, name):
        h5 = ArrayH5Dataset(parent_h5py, name)
        try:
            parameter, step_lengths = Scan._load_from_h5(parent_h5py[name])
        except:
            print(f"could not read scan metadata of {name}")
            parameter = None
            step_lengths = None

        data = h5.get_data_da()
        if data is None:
            return None
        else:
            return cls(
                index=h5.index,
                data=data,
                parameter=parameter,
                step_lengths=step_lengths,
                name=name,
            )

    def ones(self, **kwargs):
        return Array(
            data=np.ones(len(self), **kwargs),
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
            if d.dtype == bool:
                hrange = [0, 1]
            else:
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

    def get_index_array(self, N_index_aggregation=None):
        if N_index_aggregation:
            tmp = Array(data=self.index, index=self.index)
            return tmp.digitize(
                np.arange(min(tmp.data), max(tmp.data), N_index_aggregation)
            )
        else:
            return Array(data=self.index, index=self.index)

    def correct_for_references(
        self, isref_bool, N_index_aggregation=None, operation=operator.truediv
    ):
        refs = self[isref_bool]
        noref = self[~isref_bool]
        indxs = self.get_index_array(N_index_aggregation=N_index_aggregation)
        indxs = indxs[[slice(None), *([None] * (self.ndim - 1))]]

        return concatenate(
            operation(tanr, tar)
            for tanr, tar in zip(indsrt * noref, (indxs * ref).scan.mean(axis=0))
        )

    def plot(
        self,
        axis=None,
        linespec=".",
        *args,
        **kwargs,
    ):
        y = self.data
        x = self.index
        if not axis:
            axis = plt.gca()
        axis.plot(x, y, linespec, *args, **kwargs)

        if self.name:
            axis.set_ylabel(self.name)
        axis.set_xlabel("index")

    def plot_corr(
        self,
        arr,
        ratio=False,
        axis=None,
        linespec=".",
        polyfit_order=None,
        *args,
        **kwargs,
    ):
        yarr, xarr = match_arrays(self, arr)
        y = yarr.data
        x = xarr.data
        if not axis:
            axis = plt.gca()
        if ratio:
            axis.plot(x, y / x, linespec, *args, **kwargs)
        else:
            axis.plot(x, y, linespec, *args, **kwargs)

        if arr.name:
            axis.set_xlabel(arr.name)
        if self.name and arr.name:
            axis.set_ylabel(f"{self.name} / {arr.name}")
        if not polyfit_order is None:
            pres = np.polyfit(x, y, polyfit_order)
            xp = np.linspace(np.min(x), np.max(x), 1000)
            yp = np.polyval(pres, xp)
            if ratio:
                plt.plot(xp, yp / xp, "r")
            else:
                plt.plot(xp, yp, "r")
            return pres

    def hist(
        self,
        cut_percentage=0,
        bins="auto",
        normalize_to=None,
        scanpar_name=None,
        plot_results=True,
        plot_axis=None,
    ):
        if self.is_dask_array():
            raise Exception(
                "escape array needs to be numpy type for histogramming, compute first."
            )
        [hmin, hmax] = np.nanpercentile(
            self.data.ravel(), [cut_percentage, 100 - cut_percentage]
        )
        hbins = np.histogram_bin_edges(self.data.ravel(), bins, range=[hmin, hmax])
        # hbins = np.linspace(hmin, hmax, N_intervals + 1)
        hdat, bin_edges = np.histogram(self.data.ravel(), bins=hbins)
        if normalize_to == "max":
            hdat = hdat / hdat.max()
        elif normalize_to == "sum":
            hdat = hdat / hdat.sum()

        if plot_results:
            if not plot_axis:
                plot_axis = plt.gca()
            plt.step(hbins[:-1], hdat, where="post")
            plt.xlabel(self.name)
        return hdat, hbins

    def __repr__(self, bare=False):
        s = "<%s.%s object at %s>" % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self)),
        )
        s += " {}; shape {}".format(self.name, self.shape)
        s += "\n"
        if not bare:
            if isinstance(self.data, np.ndarray):
                s += self._get_ana_str()
        if self.scan:
            s += self.scan.__repr__()
        return s

    @property
    def dtype(self):
        return self.data.dtype

    def astype(self, newtype):
        return Array(
            data=self.data.astype(newtype),
            index=self.index,
            step_lengths=self.scan.step_lengths,
            parameter=self.scan.parameter,
        )

    def _get_repr_hist_plot(self, fmt="png", figsize=[5, 3]):
        plt.ioff()
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(111)
        # ax_steps = ax.twiny()

        if self.dtype == bool:
            flims = [0, 1]
        else:
            stepmns, stepmxs = zip(
                *[
                    [np.min(tmp), np.max(tmp)]
                    for tmp in self.scan.nanpercentile([5, 95])
                ]
            )
            flims = [min(stepmns), max(stepmxs)]

        if len(self.scan) > 1:
            self.filter(*flims).scan.hist(
                plot_axis=ax,
                cmap=plt.cm.Reds,
                cut_percentage=0,
            )
            if self.dtype == bool:
                self.scan.plot(axis=ax, fmt="k.-", use_quantiles=False, label="mean")
            else:
                self.scan.plot(axis=ax, fmt="k.-", use_quantiles=False, label="median")

            plt.ylabel(self.name)
            ax.legend(fancybox=True, framealpha=0.3, loc="best")
        else:
            to_hist = self.data
            if self.dtype == bool:
                to_hist = to_hist.astype(int)
            plt.hist(to_hist, "auto")
            plt.xlabel(self.name)

        ax.grid("on")
        # ax_steps.set_xlim(0, len(self.scan) - 1)
        # ax_steps.set_xlabel("Step number")
        f.tight_layout()
        if fmt == "svg":
            s = io.StringIO()
            f.savefig(s, format="svg", bbox_inches="tight")
            imgobj = s.getvalue()
        elif fmt == "png":
            tmpfile = BytesIO()
            f.savefig(tmpfile, format="png", bbox_inches="tight")
            imgobj = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        plt.ion()
        return imgobj

    def _get_repr_map_plot(self, fmt="png", figsize=[5, 3]):
        plt.ioff()
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(111)
        # ax_steps = ax.twiny()

        if len(self.scan) > 1:
            self.scan.plot(axis=ax, cmap=plt.cm.Greens)
        else:
            print("non scan repr not implemented yet for maps/waveforms or images!")

        ax.grid("on")
        # ax_steps.set_xlim(0, len(self.scan) - 1)
        # ax_steps.set_xlabel("Step number")
        f.tight_layout()
        if fmt == "svg":
            s = io.StringIO()
            f.savefig(s, format="svg", bbox_inches="tight")
            imgobj = s.getvalue()
        elif fmt == "png":
            tmpfile = BytesIO()
            f.savefig(tmpfile, format="png", bbox_inches="tight")
            imgobj = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        plt.ion()
        return imgobj

    def _repr_html_(self):
        if self.is_dask_array():
            return (
                html.escape(self.__repr__(bare=True)).replace("\n", "<br />\n")
                + self.data._repr_html_()
            )
        else:
            if (self.ndim == 1) or all([ts <= 1 for ts in self.shape[1:]]):
                return (
                    html.escape(self.__repr__(bare=True)).replace("\n", "<br />\n")
                    + "<br />\n"
                    + "<img src='data:image/png;base64,{}'>".format(
                        self._get_repr_hist_plot(fmt="png")
                    )
                )

            elif self.ndim_nonzero == 2:
                return (
                    html.escape(self.__repr__(bare=True)).replace("\n", "<br />\n")
                    + "<br />\n"
                    + "<img src='data:image/png;base64,{}'>".format(
                        self._get_repr_map_plot(fmt="png")
                    )
                )
            else:
                return None

    #     s = "<%s.%s object at %s>" % (
    #         self.__class__.__module__,
    #         self.__class__.__name__,
    #         hex(id(self)),
    #     )
    #     s += " {}; shape {}".format(self.name, self.shape)
    #     s += "\n"
    #     if isinstance(self.data, np.ndarray):
    #         s += self._get_ana_str()
    #     if self.scan:
    #         s += self.scan.__repr__()
    #     return s


def load_from_h5_file(file_name, name, parent_group_name=""):
    h5 = ArrayH5File(
        file_name=file_name, parent_group_name=parent_group_name, name=name
    )
    parameter, step_lengths = Scan._load_from_h5(parent_h5py)
    return Array(
        index=h5.index,
        data=h5.get_data_da(),
        parameter=parameter,
        step_lengths=step_lengths,
        name=name,
    )


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
        if escSorter == "first":
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
            single_output = True
            output = (output,)
        else:
            single_output = False
        output = list(output)
        if convertOutput2EscData:
            stepLengths, scan = get_scan_step_selections(
                ixmaster, sorter.scan.step_lengths, scan=sorter.scan
            )
            if convertOutput2EscData == "auto":
                convertOutput2EscData = []
                for i, toutput in enumerate(output):
                    try:
                        lentoutput = len(toutput)
                        if len(ids_res) == len(toutput):
                            convertOutput2EscData.append(i)
                    except TypeError:
                        pass

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


def scan_escaped(func):
    def wrapped(*args, **kwargs):
        args = [ta for ta in args]
        kwargs = {tk: tv for tk, tv in kwargs.items()}
        argsIsEsc = [(n, arg) for n, arg in enumerate(args) if isinstance(arg, Scan)]
        kwargsIsEsc = {
            key: kwarg for key, kwarg in kwargs.items() if isinstance(kwarg, Scan)
        }
        allEscs = [a for n, a in argsIsEsc]
        allEscs.extend(kwargsIsEsc.values())

        scan_lens = []
        for tscan in allEscs:
            scan_lens.append(len(tscan))
        if not np.unique(scan_lens):
            raise Exception(
                "Scan instances do not have same length, bring them on the same scan parameter set, e.g. using the digitize method!"
            )
        scan_len = scan_lens[0]
        args_same_len = []
        for ta in args:
            try:
                tlen = len(ta)
            except TypeError:
                continue
            if tlen == scan_len:
                args_same_len.append(ta)
        kwargs_same_len = {}
        for tn, ta in kwargs.items():
            try:
                tlen = len(ta)
            except TypeError:
                continue
            if tlen == scan_len:
                kwargs_same_len[tn] = ta
        res = []
        for i_step in range(scan_len):
            targs = []
            for ta in args:
                if id(ta) in [id(tmp) for tmp in args_same_len]:
                    targs.append(ta[i_step])
                else:
                    targs.append(ta)
            tkwargs = {}
            for tn, ta in kwargs.items():
                if id(ta) in [id(tmp) for tmp in kwargs_same_len.values()]:
                    tkwargs[tn] = ta[i_step]
                else:
                    tkwargs[tn] = ta
            tres = func(*targs, **tkwargs)
            if not isinstance(tres, tuple):
                tres = (tres,)
            res.append(tres)
        fres = []
        for tres in zip(*res):
            if isinstance(tres[0], Array):
                fres.append(concatenate(list(tres)))
            else:
                fres.append(tres)

        if len(fres) > 1:
            return tuple(fres)
        else:
            return fres[0]

    return wrapped

    # if escSorter is "first":
    # if len(allEscs) > 0:
    # sorter = allEscs[0]
    # else:
    # sorter = None
    # print(
    # "Did not find any Array instance \
    # in input parameters!"
    # )
    # else:
    # sorter = escSorter
    # if not sorter is None:
    # ixsorter = allEscs.index(sorter)
    # allEscs.pop(ixsorter)
    # ixmaster, ixslaves, stepLengthsNew = match_indexes(
    # sorter.index, [t.index for t in allEscs]
    # )
    # ixslaves.insert(ixsorter, ixmaster)
    # ids_res = sorter.index[ixmaster]
    # for n, arg in argsIsEsc:
    # args.pop(n)
    # args.insert(n, arg.data[ixslaves.pop(0)])
    # for key, kwarg in kwargsIsEsc.items():
    # kwargs.pop(key)
    # kwargs[key] = kwarg.data[ixslaves.pop(0)]
    # output = func(*args, **kwargs)
    # if not type(output) is tuple:
    # output = (output,)
    # output = list(output)
    # if convertOutput2EscData:
    # stepLengths, scan = get_scan_step_selections(
    # ixmaster, sorter.scan.step_lengths, scan=sorter.scan
    # )
    # if convertOutput2EscData == "auto":
    # convertOutput2EscData = []
    # for i, toutput in enumerate(output):
    # try:
    # lentoutput = len(toutput)
    # if len(ids_res) == len(toutput):
    # convertOutput2EscData.append(i)
    # except TypeError:
    # pass

    # for n in convertOutput2EscData:
    # toutput = output.pop(n)
    # output.insert(
    # n,
    # Array(
    # data=toutput,
    # index=ids_res,
    # step_lengths=stepLengths,
    # parameter=scan.parameter,
    # index_dim=0,
    # ),
    # )

    # if len(output) == 1:
    # output = output[0]
    # elif len(output) == 0:
    # output = None
    # return output

    # return wrapped


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
        Array,
        "__%s__" % opJoin.__name__.strip("_"),
        escaped(opJoin, convertOutput2EscData=[0]),
    )
    if (
        True
    ):  # any(top in opJoin.__name__ for top in ["add", "sub", "mul", "div", "mod"]):
        setattr(
            Array,
            "__r%s__" % opJoin.__name__.strip("_"),
            escaped(opJoin, convertOutput2EscData=[0]),
        )


for opSing, symbol in _operatorsSingle:
    setattr(
        Array,
        "__%s__" % opSing.__name__.strip("_"),
        escaped(opSing, convertOutput2EscData=[0]),
    )


def match_scans(a0, a1, parameters=[]):
    """Match scans of two escape arrays."""
    s0 = a0.scan
    s1 = a1.scan
    s0arr = np.asarray(
        [
            value["values"]
            for name, value in s0.parameter.items()
            if ((not parameters) or (name in parameters))
        ]
    ).T
    s1arr = np.asarray(
        [
            value["values"]
            for name, value in s1.parameter.items()
            if ((not parameters) or (name in parameters))
        ]
    ).T

    s0_sel = ((np.isin(s0arr, s1arr).all(axis=1))).nonzero()[0]
    s1_sel = ((np.isin(s1arr, s0arr).all(axis=1))).nonzero()[0]
    return get_step_indexes(s0, s0_sel), get_step_indexes(s1, s1_sel)


class Scan:
    def __init__(self, parameter={}, step_lengths=None, array=None):
        self.step_lengths = step_lengths
        if parameter:
            for par, pardict in parameter.items():
                if not len(pardict["values"]) == len(self):
                    raise Exception(
                        f"Parameter array length of {par} does not fit the defined steps."
                    )
        else:
            parameter = {"none": {"values": [1] * len(step_lengths)}}
        self.parameter = parameter
        self._array = array
        # self._add_methods()
        self.tools = ScanTools(self)

    def append_parameter(self, parameter: {"par_name": {"values": list}}):
        for par, pardict in parameter.items():
            if not len(pardict["values"]) == len(self):
                lenthis = len(pardict["values"])
                raise Exception(
                    f"Parameter array length of {par} ({lenthis}) does not fit the defined steps ({len(self)})."
                )
        self.parameter.update(parameter)

    @property
    def par_steps(self):
        data = {name: value["values"] for name, value in self.parameter.items()}
        data.update({"step_length": self.step_lengths})
        return pd.DataFrame(data, index=list(range(len(self))))

    def nansum(self, *args, **kwargs):
        return [step.nansum(*args, **kwargs) for step in self]

    def nanmean(self, *args, **kwargs):
        return [step.nanmean(*args, **kwargs) for step in self]

    def nanstd(self, *args, **kwargs):
        return [step.nanstd(*args, **kwargs) for step in self]

    def nanmedian(self, *args, **kwargs):
        return [step.nanmedian(*args, **kwargs) for step in self]

    def nanpercentile(self, *args, **kwargs):
        return [step.nanpercentile(*args, **kwargs) for step in self]

    def nanquantile(self, *args, **kwargs):
        return [step.nanquantile(*args, **kwargs) for step in self]

    def nanmin(self, *args, **kwargs):
        return [step.nanmin(*args, **kwargs) for step in self]

    def nanmax(self, *args, **kwargs):
        return [step.nanmax(*args, **kwargs) for step in self]

    def sum(self, *args, **kwargs):
        return [step.sum(*args, **kwargs) for step in self]

    def mean(self, *args, **kwargs):
        return [step.mean(*args, **kwargs) for step in self]

    def average(self, *args, **kwargs):
        return [step.average(*args, **kwargs) for step in self]

    def std(self, *args, **kwargs):
        return [step.std(*args, **kwargs) for step in self]

    def median(self, *args, **kwargs):
        return [step.median(*args, **kwargs) for step in self]

    def min(self, *args, **kwargs):
        return [step.min(*args, **kwargs) for step in self]

    def max(self, *args, **kwargs):
        return [step.max(*args, **kwargs) for step in self]

    def any(self, *args, **kwargs):
        return [step.any(*args, **kwargs) for step in self]

    def all(self, *args, **kwargs):
        return [step.all(*args, **kwargs) for step in self]

    def count(self):
        return [len(step) for step in self]

    def nancount(self): ...

    # TODO

    def median_and_mad(self, axis=None, k_dist=1.4826, norm_samples=False):
        """Calculate median and median absolute deviation for steps of a scan.

        Args:
            axis (int, sequence of int, None, optional): axis argument for median calls.
            k_dist (float, optional): distribution scale factor, should be
                1 for real MAD.
                Defaults to 1.4826 for gaussian distribution.
        """
        # if self._array.is_dask_array():
        #     absfoo = da.abs
        # else:
        #     absfoo = np.abs

        med = [step.median(axis=axis) for step in self]
        mad = [
            (((step - tmed).abs()) * k_dist).median(axis=axis)
            for step, tmed in zip(self, med)
        ]
        if norm_samples:
            mad = [tmad / da.sqrt(ct) for tmad, ct in zip(mad, self.count())]
        return med, mad

    # def weighted_median_and_mad(self, weights=None, axis=None, k_dist=1.4826, norm_samples=False):
    #     """Calculate median and median absolute deviation for steps of a scan.

    #     Args:
    #         axis (int, sequence of int, None, optional): axis argument for median calls.
    #         k_dist (float, optional): distribution scale factor, should be
    #             1 for real MAD.
    #             Defaults to 1.4826 for gaussian distribution.
    #     """
    #     # if self._array.is_dask_array():
    #     #     absfoo = da.abs
    #     # else:
    #     #     absfoo = np.abs
    #     utilities.weighted_quantiles(0.5)
    #     med = [step.median(axis=axis) for step in self]
    #     mad = [
    #         (((step - tmed).abs()) * k_dist).median(axis=axis)
    #         for step, tmed in zip(self, med)
    #     ]
    #     if norm_samples:
    #         mad = [tmad / da.sqrt(ct) for tmad, ct in zip(mad, self.count())]
    #     return med, mad

    def weighted_avg_and_std(self, weights=None, norm_samples=False):
        avg = []
        std = []
        for step in self:
            if weights:
                (ta, tw) = match_arrays(step, weights)
                (tavg, tstd) = utilities.weighted_avg_and_std(ta.data, tw.data)
            else:
                (tavg, tstd) = utilities.weighted_avg_and_std(step.data, weights)
            avg.append(tavg)
            std.append(tstd)
        if norm_samples:
            std = [tstd / da.sqrt(ct) for tstd, ct in zip(std, self.count())]
        return np.asarray(avg), np.asarray(std)

    def correlation_analysis_to(self, ref, *args, **kwargs):
        (td, tr) = match_arrays(self._array, ref)
        return [
            step.correlation_analysis_to(tref, *args, **kwargs)
            for step, tref in zip(td.scan, tr.scan)
        ]

    def corr_ana_plot(self, referece, scanpar_name=None, axis=None):
        if not scanpar_name:
            names = list(self.parameter.keys())
            scanpar_name = names[0]
        x = np.asarray(self.parameter[scanpar_name]["values"]).ravel()
        corres = self.correlation_analysis_to(referece)

        if not axis:
            axis = plt.gca()

        std = [tc[0] for tc in corres]
        std_fx = [tc[1] for tc in corres]

        ordercolors = ["b", "r"]
        for to, toc in zip([0, 1], ordercolors):
            axis.plot(
                x,
                [tc[to] for tc in std],
                toc + "--" + ".",
                label=f"poly. order:{to+1}; zero free",
            )
        for to, toc in zip([0, 1], ordercolors):
            axis.plot(
                x,
                [tc[to] for tc in std_fx],
                toc + "-" + "o",
                label=f"poly. order:{to+1}; zero fixed",
            )

    def plot(
        self,
        weights=None,
        scanpar_name=None,
        norm_samples=True,
        axis=None,
        use_quantiles=True,
        *args,
        **kwargs,
    ):
        if not scanpar_name:
            names = list(self.parameter.keys())
            scanpar_name = names[0]
        x = np.asarray(self.parameter[scanpar_name]["values"]).ravel()

        if self._array.ndim_nonzero == 1:
            if not weights:
                if use_quantiles:
                    tmp = np.asarray(
                        self.nanquantile(
                            [0.5, 0.5 - 0.682689492137 / 2, 0.5 + 0.682689492137 / 2],
                            # axis=0,
                        )
                    )
                    y = tmp[:, 0]
                    ystd = np.diff(tmp[:, 1:], axis=1)[:, 0] / 2
                else:
                    y = np.asarray(self.nanmean(axis=0)).ravel()
                    ystd = np.asarray(self.nanstd(axis=0)).ravel()
            else:
                if use_quantiles:
                    print(
                        "Cannot use quantile derivation with weights, using average/std instead!"
                    )
                y, ystd = self.weighted_avg_and_std(weights)
            if norm_samples:
                yerr = ystd / np.sqrt(np.asarray(self.count()))
            else:
                yerr = ystd
            if not axis:
                axis = plt.gca()
            axis.errorbar(x, y, yerr=yerr, *args, **kwargs)
            axis.set_xlabel(scanpar_name)
            if self._array.name:
                axis.set_ylabel(self._array.name)
        elif self._array.ndim_nonzero == 2:
            if use_quantiles:
                tmp = np.asarray(
                    self.nanquantile(
                        [0.5, 0.5 - 0.682689492137 / 2, 0.5 + 0.682689492137 / 2],
                        axis=0,
                    )
                )

                ic = tmp[
                    :,
                    0,
                    :,
                ]
                icstd = np.diff(tmp[:, 1:, :], axis=1)[:, 0, :] / 2
                # return ic, icstd
                if not axis:
                    axis = plt.gca()
                y = np.arange(ic.shape[1])
                ih = plot2D(x, y, ic.T, ax=axis, *args, **kwargs)
                axis.set_xlabel(scanpar_name)
                plt.colorbar(ih, ax=axis, label=self._array.name)
                axis.set_ylabel("Median step waveform")
                return ic, icstd

    def hist(
        self,
        cut_percentage=0,
        bins="auto",
        normalize_to=None,
        scanpar_name=None,
        plot_results=True,
        plot_axis=None,
        **kwargs,
    ):
        if self._array.is_dask_array():
            raise Exception(
                "escape array needs to be numpy type for histogramming, compute first."
            )
        if not scanpar_name:
            names = list(self.parameter.keys())
            for scanpar_name in names:
                if not np.isnan(
                    np.asarray(self.parameter[scanpar_name]["values"]).ravel()
                ).all():
                    break
        x_scan = np.asarray(self.parameter[scanpar_name]["values"]).ravel()

        [hmin, hmax] = np.nanpercentile(
            self._array.data.ravel().astype(float),
            [cut_percentage, 100 - cut_percentage],
        )
        # hbins = np.linspace(hmin, hmax, N_intervals + 1)
        hbins = np.histogram_bin_edges(
            self._array.data.ravel(), bins, range=[hmin, hmax]
        )

        hdat = [np.histogram(td.data.ravel(), bins=hbins)[0] for td in self]
        if normalize_to == "max":
            hdat = [td / td.max() for td in hdat]
        elif normalize_to == "sum":
            hdat = [td / td.sum() for td in hdat]
        hdat = np.asarray(hdat)
        if plot_results:
            if not plot_axis:
                plot_axis = plt.gca()
            # utilities.plot2D(x_scan, utilities.edges_to_center(hbins), hdat.T, **kwargs)
            plt.pcolormesh(x_scan, utilities.edges_to_center(hbins), hdat.T, **kwargs)
            plt.xlabel(scanpar_name)
        return x_scan, hbins, hdat

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

    def get_step_indexes(self, ix_step):
        """ "array getter for multiple steps, more efficient than get_step_array"""
        ix_to = np.cumsum(self.step_lengths)
        ix_from = np.hstack([np.asarray([0]), ix_to[:-1]])
        index_sel = np.concatenate(
            [
                self._array.index[fr:to]
                for fr, to in zip(ix_from[ix_step], ix_to[ix_step])
            ],
            axis=0,
        )
        return self._array[np.in1d(self._array.index, index_sel).nonzero()[0]]

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
        try:
            scan_group = group.require_group("scan", track_order=True)
        except:
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
                    if isinstance(attvalue, str):
                        attvalue = str(attvalue)
                    tpg["attributes"][attname] = attvalue

    @staticmethod
    def _load_from_h5(group):
        if "scan" not in group.keys():
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

    def __repr__(self):
        s = "Scan over {} steps".format(len(self))
        s += "\n"
        s += "Parameters {}".format(", ".join(self.parameter.keys()))
        return s

    # def __add__(self,other):
    # return scan_escaped(operator.add)(self,other)


_operatorsJoin = [
    (operator.add, "+"),
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
    (operator.ne, "!="),
    (operator.ge, ">="),
    (operator.gt, ">"),
    # (operator.contains, "in"),
    # (operator.eq, "=="),
]


_operatorsSingle = [
    (operator.invert, "~"),
    (operator.neg, "-"),
    (operator.not_, "not"),
    (operator.pos, "pos"),
]
for opJoin, symbol in _operatorsJoin:
    setattr(Scan, "__%s__" % opJoin.__name__.strip("_"), scan_escaped(opJoin))

for opSing, symbol in _operatorsSingle:
    setattr(Scan, "__%s__" % opSing.__name__.strip("_"), scan_escaped(opSing))


def to_dataframe(*args):
    """work in progress"""
    for arg in args:
        if not np.prod(arg.shape) == len(arg):
            raise (
                NotImplementedError("Only 1D Arrays can be converted to dataframes.")
            )
    dfs = [
        ddf.from_dask_array(arg.data.ravel(), columns=[arg.name], index=arg.index)
        for arg in args
    ]
    return ddf.concat(dfs, axis=0, join="outer", interleave_partitions=False)


@escaped
def match_arrays(*args):
    return args


weighted_avg_and_std = escaped(utilities.weighted_avg_and_std)


def compute(*args):
    """compute multiple escape arrays or dask arrays. Interesting when calculating multiple small arrays
    from the same ancestor dask based array"""
    argtypes = []
    argcollection = []
    for arg in args:
        targtype = []
        if isinstance(arg,Array):
            targtype.append('esc-array')
            if arg.is_dask_array():
                targtype.append('dask_array')
                argcollection.append(arg.data)
        elif isinstance(arg,DaskCollection):
            targtype.append('daskcollection')
            if isinstance(arg,da.Array):
                targtype.append('dask_array')
                argcollection.append(arg)
        else:
            targtypes.append('nodask')
        argtypes.append(targtype)

    with ProgressBar():
        res = da.compute(*argcollection)
    next_dask_index = 0
    out = []
    for ta,argtype in zip(args,argtypes):
        if ('esc-array' in argtype) and ('dask_array' in argtype):
            out.append(
                Array(
                    data=res[next_dask_index],
                    index=ta.index,
                    step_lengths=ta.scan.step_lengths,
                    parameter=ta.scan.parameter,
                    index_dim=ta.index_dim,
                )
            )
            next_dask_index += 1
        elif ('daskcollection' in argtype) and ('dask_array' in argtype):
            out.append(res[next_dask_index])
            next_dask_index += 1
        else:
            out.append(ta)
    return tuple(out)


def store(arrays, lock="auto", **kwargs):
    """
    Storing of multiple escape arrays (as iterable, list or similar), efficient when they originate from the same ancestor
    """
    if lock == "auto":
        lock = get_lock()
    prep = [
        array.h5.append(array.data, array.index, prep_run="store_numpy")
        for array in arrays
    ]
    # return prep
    if not any(prep):
        print("Nothing to append")
        # arrays_store = arrays
    else:
        arrays_store, ndatas, dsets, n_news = zip(
            *[(tarray, *tprep) for tarray, tprep in zip(arrays, prep) if tprep]
        )
        with ProgressBar():
            da.store(ndatas, dsets, lock=lock, **kwargs)
        for array, n_new in zip(arrays_store, n_news):
            array.h5._n_i.append(n_new)
            array.h5._n_d.append(n_new)
    for array in arrays:
        array._data = array.h5.get_data_da()
        array._index = array.h5.index
        array.scan._save_to_h5(array.h5.grp)


def get_lock():
    if escape.STORAGE_LOCK:
        return escape.STORAGE_LOCK
    else:
        return True


def concatenate(arraylist):
    if all([ta.is_dask_array() for ta in arraylist]):
        data = da.concatenate([array.data for array in arraylist], axis=0)
    else:
        data = np.concatenate([array.data for array in arraylist], axis=0)
    index = np.concatenate([array.index for array in arraylist])
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
    validsteps = ~(stepLengths == 0)
    stepLengths = stepLengths[validsteps]
    if scan:
        scan = Scan(
            parameter=scan.get_parameter_selection(validsteps), step_lengths=stepLengths
        )
    return stepLengths, scan


def escaped_FuncsOnEscArray(array, inst_funcs, *args, **kwargs):
    # TODO
    for inst, func in inst_funcs:
        if isinstance(array.data, inst):
            return escaped(func, *args, **kwargs)


def digitize(
    array,
    bins,
    include_outlier_bins=False,
    sort_groups_by_index=True,
    right=False,
    foo=np.digitize,
    **kwargs,
):
    """Digitization function for escape arrays according to numpy.digitize.
    Works for 1D arrays only.

    Args:
        array (escape.Array): the escape array holding data that are supposed
            to be sorted/digitized.
        bins (array_kile): array of bins, has to be 1-dimensional and monotonic.
        include_outlier_bins (bool/'right'/'left' optional): option to include
            outliers of described bin edges on either or both siges of the bins
            array. Defaults to False.
        sort_groups_by_index (bool, optional): sorting escape.Array data within
            bins according to their index value. Defaults to True.
        right (bool, optional): Indicating whether the intervals include the
            right or the left bin edge. Default behavior is (right==False)
            indicating that the interval does not include the right edge. The
            left bin end is open in this case, i.e., bins[i-1] <= x < bins[i]
            is the default behavior for monotonically increasing bins.
            Defaults to False.
        foo (function, optional): option to modify the digitisation function,
            needs still to behave closely to np digitize. Defaults to
            np.digitize.

    Raises:
        NotImplementedError: error if no 1d escape.Array is provided as array
            argument.

    Returns:
        escape.Array: Digitized/ resorted escape.Array
    """

    if not np.prod(np.asarray(array.shape)) == array.shape[array.index_dim]:
        raise NotImplementedError(
            "Only 1d escape arrays can be digitized in a sensible way."
        )
    darray = array.data.ravel()
    if include_outlier_bins:
        direction = np.sign(bins[-1] - bins[0])
        if include_outlier_bins == "right":
            bins = np.concatenate(
                [
                    bins,
                    np.atleast_1d(direction * -np.inf),
                ]
            )
        elif include_outlier_bins == "left":
            bins = np.concatenate(
                [
                    np.atleast_1d(direction * np.inf),
                    bins,
                ]
            )
        else:
            bins = np.concatenate(
                [
                    np.atleast_1d(direction * np.inf),
                    bins,
                    np.atleast_1d(direction * -np.inf),
                ]
            )
    if foo is np.digitize:
        kwargs["right"] = right
    inds = foo(darray, bins, **kwargs)
    ix = inds.argsort()[(0 < inds) & (inds < len(bins))]
    bin_nos, counts = np.unique(
        inds[(0 < inds) & (inds < len(bins))] - 1, return_counts=True
    )
    if sort_groups_by_index:
        for n, bin_no in enumerate(bin_nos):
            tmn = sum(counts[:n])
            tmx = sum(counts[: n + 1])
            tix = array.index[ix[tmn:tmx]].argsort()
            ix[tmn:tmx] = ix[tmn:tmx][tix]
    bin_left = bins[1:]
    bin_right = bins[:-1]
    bin_center = (bin_left + bin_right) / 2

    parameter = {
        f"bin_center_{array.name}": {
            "values": bin_center[bin_nos],
            "attributes": kwargs,
        },
        f"bin_left_{array.name}": {"values": bin_left[bin_nos], "attributes": kwargs},
        f"bin_right_{array.name}": {"values": bin_right[bin_nos], "attributes": kwargs},
    }

    return Array(
        data=array.data[ix],
        index=array.index[ix],
        parameter=parameter,
        step_lengths=counts,
    )


def filter(array, *args, foos_filtering=[operator.ge, operator.le], **kwargs):
    """general filter function for escape arrays. checking for 1D arrays, applies
    arbitrary number of
    filter functions that take one argument as input and"""
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
    stepLengths, scan = get_scan_step_selections(
        ix, array.scan.step_lengths, scan=array.scan
    )
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
        try:
            self.grp = parent[name]
        except:
            self.grp = parent.require_group(name)
        if (
            "esc_type" in self.grp.attrs.keys()
            and self.grp.attrs["esc_type"] == "array_dataset"
        ):
            pass
        else:
            try:
                self.grp.attrs["esc_type"] = "array_dataset"
            except:
                print("Could not put esc_type metadata.")

        self._data_finder = re.compile("^data_[0-9]{4}$")
        self._index_finder = re.compile("^index_[0-9]{4}$")
        self._check_stored_data()

    def _check_stored_data(self):
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

    def clear_stored_data(self):
        for key in self.grp.keys():
            try:
                del self.grp[key]
            except:
                print(f"Did not succeed to delete key {key}!")

    @property
    def index(self):
        if self._n_i:
            return np.concatenate(
                [np.asarray(self.grp[f"index_{n:04d}"][:]) for n in self._n_i], axis=0
            )
        else:
            return np.asarray([], dtype=int)

    def append(self, data, event_ids, scan=None, prep_run=False, lock="auto", **kwargs):
        """
        expects to extend a former dataset, i.e. data includes data already existing,
        this will likely change in future to also allow real appending of entirely new data.
        """
        if lock == "auto":
            lock = get_lock()
        n_new = len(self._n_i)
        ids_stored = self.index
        in_previous_indexes = np.in1d(event_ids, ids_stored)
        if ~in_previous_indexes.any():
            # real appending data
            new_event_ids = event_ids
            new_data = data
        elif in_previous_indexes.all():
            # real extending of data

            if len(event_ids) < len(ids_stored):
                raise Exception("fewer event_ids to append than already stored!")
            if not (event_ids[: len(ids_stored)] == ids_stored).all():
                raise Exception("new event_ids don't extend existing ones!")
            if len(event_ids) == len(ids_stored):
                print("Nothing new to append.")
                return

            new_event_ids = event_ids[len(ids_stored) :]
            new_data = data[len(ids_stored) :, ...]

        self.grp[f"index_{n_new:04d}"] = new_event_ids

        if isinstance(data, np.ndarray):
            if prep_run:
                if prep_run == "store_numpy":
                    pass
                else:
                    raise Exception(
                        "Trying dry_run on numpy array data on {self.grp.name}."
                    )
            self.grp[f"data_{n_new:04d}"] = new_data
        elif isinstance(data, da.Array):
            # ToDo, smarter chunking when writing small data
            new_chunks = tuple(c[0] for c in new_data.chunks)
            dset = self.grp.create_dataset(
                f"data_{n_new:04d}",
                shape=new_data.shape,
                chunks=new_chunks,
                dtype=new_data.dtype,
            )
            if prep_run:
                return new_data, dset, n_new
            da.store(new_data, dset, lock=lock, **kwargs)
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
                    np.dtype(ds.dtype).itemsize * np.prod(ds.shape[1:]) / 1024**2
                )

                chunk_size[0] = int(memlimit_MB // size_element)

            allarrays.append(da.from_array(ds, chunks=chunk_size))
        # if len(allarrays) < 1:
        #     print(ds, ds.shape, chunk_size)
        if len(allarrays) == 0:
            return None
        else:
            return da.concatenate(allarrays)

    def create_array(self):
        return Array(data=self.get_data_da(), index=self.index)


class ArrayH5File:
    def __init__(self, file_name, parent_group_name="", name=None):
        self.file_name = Path(file_name)
        self.parent_group_name = parent_group_name
        self.name = name
        self.require_group()
        self.update_dataset_status()

    @property
    def group_name(self):
        return (Path(self.parent_group_name) / Path(self.name)).as_posix()

    def require_group(self):
        with h5py.File(self.file_name, "a") as f:
            f[self.parent_group_name].require_group(self.name)

    def update_dataset_status(self):
        # self.grp = parent.require_group(name)
        self._data_finder = re.compile("^data_[0-9]{4}$")
        self._index_finder = re.compile("^index_[0-9]{4}$")
        self._n_d = []
        self._n_i = []
        with h5py.File(self.file_name, "r") as f:
            keys = f[self.group_name].keys()
            for key in keys:
                if self._data_finder.match(key):
                    self._n_d.append(int(key[-4:]))
                if self._index_finder.match(key):
                    self._n_i.append(int(key[-4:]))
        self._n_d.sort()
        self._n_i.sort()
        if not self._n_d == self._n_i:
            raise Exception(
                "Corrupt escape ArrayH5Dataset, not equally sized data and index sub-datasets!"
            )

    @property
    def index(self):
        if self._n_i:
            with h5py.File(self.file_name, "r") as f:
                return np.concatenate(
                    [
                        np.asarray(f[self.group_name][f"index_{n:04d}"][:])
                        for n in self._n_i
                    ],
                    axis=0,
                )
        else:
            return np.asarray([], dtype=int)

    def append(self, data, event_ids, scan=None, prep_run=False):
        """
        expects to extend a former dataset, i.e. data includes data already existing,
        this will likely change in future to also allow real appending of entirely new data.
        """
        n_new = len(self._n_i)
        ids_stored = self.index
        in_previous_indexes = np.in1d(event_ids, ids_stored)
        if ~in_previous_indexes.any():
            # real appending data
            new_event_ids = event_ids
            new_data = data
        elif in_previous_indexes.all():
            # real extending of data

            if len(event_ids) < len(ids_stored):
                raise Exception("fewer event_ids to append than already stored!")
            if not (event_ids[: len(ids_stored)] == ids_stored).all():
                raise Exception("new event_ids don't extend existing ones!")
            if len(event_ids) == len(ids_stored):
                print("Nothing new to append.")
                return

            new_event_ids = event_ids[len(ids_stored) :]
            new_data = data[len(ids_stored) :, ...]

        with h5py.File(self.file_name, "a") as f:
            f[self.group_name + f"/index_{n_new:04d}"] = new_event_ids

        if isinstance(data, np.ndarray):
            if prep_run:
                if prep_run == "store_numpy":
                    pass
                else:
                    raise Exception(
                        "Trying dry_run on numpy array data on {self.grp.name}."
                    )
            with h5py.File(self.file_name, "a") as f:
                f[self.group_name + f"/data_{n_new:04d}"] = new_data
        elif isinstance(data, da.Array):
            # ToDo, smarter chunking when writing small data
            new_chunks = tuple(c[0] for c in new_data.chunks)
            location = {
                "file_name": self.file_name,
                "dataset_name": self.group_name + f"/data_{n_new:04d}",
            }
            if prep_run:
                return new_data, location, n_new
            else:
                data.to_hdf5(self.file_name, location["dataset_name"])
        if scan:
            with h5py.File(self.file_name, "a") as f:
                scan._save_to_h5(f[self.group_name])
        self._n_i.append(n_new)
        self._n_d.append(n_new)

    def get_data_da(self, memlimit_MB=50):
        allarrays = []
        for n in self._n_i:
            ds_name = self.group_name + f"/data_{n:04d}"
            h5store = self.analyze_h5_dataset(ds_name, memlimit_MB=memlimit_MB)
            tda = self.h5store_to_da(h5store=h5store)
            allarrays.append(tda)

        return da.concatenate(allarrays)

    def create_array(self):
        return Array(data=self.get_data_da(), index=self.index)

    # @dask.delayed
    def analyze_h5_dataset(self, dataset_path, memlimit_MB=100):
        """Data parser assuming the standard swissfel h5 format for raw data"""
        with h5py.File(self.file_name, mode="r") as fh:
            ds_data = fh[dataset_path]
            if memlimit_MB:
                dtype = np.dtype(ds_data.dtype)
                size_element = (
                    np.dtype(ds_data.dtype).itemsize
                    * np.prod(ds_data.shape[1:])
                    / 1024**2
                )
                chunk_length = int(memlimit_MB // size_element)
                dset_size = ds_data.shape

                chunk_shapes = []
                slices = []
                for chunk_start in range(0, dset_size[0], chunk_length):
                    slice_0dim = [
                        chunk_start,
                        min(chunk_start + chunk_length, dset_size[0]),
                    ]
                    chunk_shape = list(dset_size)
                    chunk_shape[0] = slice_0dim[1] - slice_0dim[0]
                    slices.append(slice_0dim)
                    chunk_shapes.append(chunk_shape)

            h5store = {
                "file_path": self.file_name,
                "dataset_name": ds_data.name,
                "dataset_shape": ds_data.shape,
                "dataset_dtype": dtype,
                "dataset_chunks": {"slices": slices, "shapes": chunk_shapes},
            }
        return h5store

    @dask.delayed
    def read_h5_chunk(self, ds_path, slice_args):
        with h5py.File(self.file_name, "r") as fh:
            dat = fh[ds_path][slice(*slice_args)]
        return dat

    def h5store_to_da(self, h5store):
        arrays = [
            dask.array.from_delayed(
                self.read_h5_chunk(h5store["dataset_name"], tslice),
                tshape,
                dtype=h5store["dataset_dtype"],
            )
            for tslice, tshape in zip(
                h5store["dataset_chunks"]["slices"], h5store["dataset_chunks"]["shapes"]
            )
        ]
        data = dask.array.concatenate(arrays, axis=0)
        return data


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
