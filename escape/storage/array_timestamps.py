from numbers import Number
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import h5py
import escape
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def get_lock():
    if escape.STORAGE_LOCK:
        return escape.STORAGE_LOCK
    else:
        return True


class ArrayTimestamps:
    def __init__(
        self, data, timestamps, timestamp_intervals=None, parameter=None, name="none"
    ):
        self.data = np.asarray(data)
        self.timestamps = np.asarray(timestamps)
        self.name = name
        self.scan = ScanTimestamps(
            parameter=parameter, timestamp_intervals=timestamp_intervals, array=self
        )
        self._append_methods()

    @property
    def shape(self, *args, **kwargs):
        return self.data.shape

    @property
    def ndim(self, *args, **kwargs):
        return self.data.ndim

    @property
    def ndim_nonzero(self, *args, **kwargs):
        return len(np.asarray(self.shape)[np.nonzero(self.shape)[0]])

    def __len__(self):
        return len(self.timestamps)

    #   >> storing

    def store(self, parent_h5py=None, name=None, unit=None, lock="auto", **kwargs):
        """a way to store data, especially expensively computed data, into a new file."""
        if lock == "auto":
            lock = get_lock()
        if not hasattr(self, "h5"):
            self.h5 = ArrayH5Dataset(parent_h5py, name)

        with ProgressBar():
            self.h5.append(self.data, self.timestamps, self.scan, lock=lock, **kwargs)
        self._data = self.h5.get_data_da()
        self._timestamps = self.h5.timestamps
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
            self.h5.append(self.data, self.timestamps, self.scan)
        self._data = self.h5.get_data_da()
        self._timestamps = self.h5.index

    def set_h5_storage_file(self, file_name, parent_group_name, name=None):
        if not hasattr(self, "h5"):
            if not name:
                name = self.name
            self.h5 = ArrayH5File(file_name, parent_group_name, name)
        else:
            logger.info(f"h5 storage already set at {name} in {self.h5.file_name}")

    def _apply_method(self, method, *args, **kwargs):
        return method(self.data, *args, **kwargs)

    def _append_methods(self):
        for name in ["mean", "std", "min"]:
            method = np.__dict__[name]

            def tmet(self, *args, **kwargs):
                return self._apply_method(method, *args, **kwargs)

            setattr(type(self), name, tmet)

    @classmethod
    def load_from_h5(cls, parent_h5py, name):
        h5 = ArrayH5Dataset(parent_h5py, name)
        try:
            parameter, timestamp_intervals = ScanTimestamps._load_from_h5(
                parent_h5py[name]
            )
        except:
            # print(f"could not read scan metadata of {name}")
            parameter = None
            timestamp_intervals = None

        data = h5.get_data_da()
        if data is None:
            return None
        else:
            return cls(
                timestamps=h5.timestamps,
                data=data,
                parameter=parameter,
                timestamp_intervals=timestamp_intervals,
                name=name,
            )

    #    << storing


class ScanTimestamps:
    def __init__(self, parameter={}, timestamp_intervals=None, array=None, data=None):
        self.timestamp_intervals = timestamp_intervals
        if parameter:
            for par, pardict in parameter.items():
                if not len(pardict["values"]) == len(self):
                    raise Exception(
                        f"Parameter array length of {par} does not fit the defined steps."
                    )
        else:
            parameter = {"none": {"values": [1] * len(timestamp_intervals)}}
        self.parameter = parameter
        self._array = array
        # self._add_methods()

        if data is not None:
            self._data = data

    def append_parameter(self, parameter: {"par_name": {"values": list}}):
        for par, pardict in parameter.items():
            if not len(pardict["values"]) == len(self):
                lenthis = len(pardict["values"])
                raise Exception(
                    f"Parameter array length of {par} ({lenthis}) does not fit the defined steps ({len(self)})."
                )
        self.parameter.update(parameter)

    def count(self):
        return [len(step) for step in self]

    @property
    def par_steps(self):
        data = {name: value["values"] for name, value in self.parameter.items()}
        data.update({"timestamp_start": self.timestamp_intervals[:, 0]})
        data.update({"timestamp_stop": self.timestamp_intervals[:, 1]})
        return pd.DataFrame(data, index=list(range(len(self))))

    def append_step(self, parameter, timestamp_interval):
        self.timestamp_intervals.append(timestamp_interval)
        for par, pardict in parameter:
            self.parameter[par]["values"].append(pardict["values"])

    def __len__(self):
        return len(self.timestamp_intervals)

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
        if n == 0 and self.timestamp_intervals is None:
            data = self._array.data[:]
            timestamps = self._array.timestamps[:]
            timestamp_intervals = self._array.timestamp_intervals
            # parameter = self._array.parameter

        # assert not self.step_lengths is None, "No step sizes defined."
        elif not n < len(self.timestamp_intervals):
            raise IndexError(f"Only {len(self.timestamp_intervals)} steps")
        else:
            interval = self.timestamp_intervals[n]
            inds_all = np.searchsorted(interval, self._array.timestamps)
            indmin = (inds_all == 0).nonzero()[0][-1]
            if 2 in inds_all:
                indmax = (inds_all == 2).nonzero()[0][0]
            else:
                indmax = len(self._array.timestamps)
            inds = (indmin, indmax)
            data = self._array.data[slice(*inds)]
            timestamps = self._array.timestamps[slice(*inds)]
            timestamp_intervals = [self.timestamp_intervals[n]]
            parameter = {}
            for par_name, par in self.parameter.items():
                parameter[par_name] = {}
                parameter[par_name]["values"] = [par["values"][n]]
                if "attributes" in par.keys():
                    parameter[par_name]["attributes"] = par["attributes"]
        return ArrayTimestamps(
            data=data,
            timestamps=timestamps,
            parameter=parameter,
            timestamp_intervals=timestamp_intervals,
        )

    # def get_step_indexes(self, ix_step):  # TODO
    #     """ "array getter for multiple steps, more efficient than get_step_array"""
    #     ix_to = np.cumsum(self.step_lengths)
    #     ix_from = np.hstack([np.asarray([0]), ix_to[:-1]])
    #     index_sel = np.concatenate(
    #         [
    #             self._array.index[fr:to]
    #             for fr, to in zip(ix_from[ix_step], ix_to[ix_step])
    #         ],
    #         axis=0,
    #     )
    #     return self._array[np.isin(self._array.index, index_sel).nonzero()[0]]

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

    def plot(
        self,
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

        if use_quantiles:
            tmp = np.asarray(
                [
                    np.nanquantile(
                        tstep.data,
                        [
                            0.5,
                            0.5 - 0.682689492137 / 2,
                            0.5 + 0.682689492137 / 2,
                        ],
                    )
                    for tstep in self
                ]
            )
            y = tmp[:, 0]
            ystd = np.diff(tmp[:, 1:], axis=1)[:, 0] / 2
        else:
            y = np.asarray([np.nanmean(tstep.data, axis=0) for tstep in self]).ravel()
            ystd = np.asarray([np.nanstd(tstep.data, axis=0) for tstep in self]).ravel()
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

    # def nansum(self, *args, **kwargs):
    #     return [step.nansum(*args, **kwargs) for step in self]

    def nanmean(self, *args, **kwargs):
        return [step.nanmean(*args, **kwargs) for step in self]

    def nanstd(self, *args, **kwargs):
        return [step.nanstd(*args, **kwargs) for step in self]

    def nanmedian(self, *args, **kwargs):
        return [step.nanmedian(*args, **kwargs) for step in self]

    # def nanpercentile(self, *args, **kwargs):
    #     return [step.nanpercentile(*args, **kwargs) for step in self]

    # def nanquantile(self, *args, **kwargs):
    #     return [step.nanquantile(*args, **kwargs) for step in self]

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

    # def nancount(self): ...

    # # TODO

    # def median_and_mad(self, axis=None, k_dist=1.4826, norm_samples=False):
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

    #     med = [step.median(axis=axis) for step in self]
    #     mad = [
    #         (((step - tmed).abs()) * k_dist).median(axis=axis)
    #         for step, tmed in zip(self, med)
    #     ]
    #     if norm_samples:
    #         mad = [tmad / da.sqrt(ct) for tmad, ct in zip(mad, self.count())]
    #     return med, mad

    def _save_to_h5(self, group):
        self._check_consistency()
        if "scan" in group.keys():
            del group["scan"]
        try:
            scan_group = group.require_group("scan", track_order=True)
        except:
            scan_group = group.require_group("scan")
        scan_group["timestamp_intervals"] = self.timestamp_intervals
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
        step_lengths = group["scan"]["timestamp_intervals"][()]
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


class ArrayH5Dataset:
    def __init__(self, parent, name):
        self.parent = parent
        try:
            self.grp = parent[name]
        except:
            self.grp = parent.require_group(name)
        if (
            "esc_type" in self.grp.attrs.keys()
            and self.grp.attrs["esc_type"] == "array_timestamps_dataset"
        ):
            pass
        else:
            try:
                self.grp.attrs["esc_type"] = "array_timestamps_dataset"
            except:
                print("Could not put esc_type metadata.")

        self._data_finder = re.compile("^data_[0-9]{4}$")
        self._ts_finder = re.compile("^timestamps_[0-9]{4}$")
        self._check_stored_data()

    def _check_stored_data(self):
        self._n_d = []
        self._n_t = []
        for key in self.grp.keys():
            if self._data_finder.match(key):
                self._n_d.append(int(key[-4:]))
            if self._ts_finder.match(key):
                self._n_t.append(int(key[-4:]))
        self._n_d.sort()
        self._n_t.sort()
        # print(self._n_t,self._n_d)
        if not self._n_d == self._n_t:
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
    def timestamps(self):
        if self._n_t:
            return np.concatenate(
                [np.asarray(self.grp[f"timestamps_{n:04d}"][:]) for n in self._n_t],
                axis=0,
            )
        else:
            return np.asarray([], dtype=int)

    def append(
        self, data, timestamps, scan=None, prep_run=False, lock="auto", **kwargs
    ):
        """
        expects to extend a former dataset, i.e. data includes data already existing,
        this will likely change in future to also allow real appending of entirely new data.
        """
        if lock == "auto":
            lock = get_lock()
        n_new = len(self._n_t)
        timestamps_stored = self.timestamps
        in_previous_timestamps = np.isin(timestamps, timestamps_stored)
        if ~in_previous_timestamps.any():
            # real appending data
            new_timestamps = timestamps
            new_data = data
        elif in_previous_timestamps.all():
            # real extending of data

            if len(timestamps) < len(timestamps_stored):
                raise Exception("fewer event_ids to append than already stored!")
            if not (timestamps[: len(timestamps_stored)] == timestamps_stored).all():
                raise Exception("new event_ids don't extend existing ones!")
            if len(timestamps) == len(timestamps_stored):
                print("Nothing new to append.")
                return

            new_timestamps = timestamps[len(timestamps_stored) :]
            new_data = data[len(timestamps_stored) :, ...]

        self.grp[f"timestamps_{n_new:04d}"] = new_timestamps

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

            try:
                if "default_dataset_compression" in self.grp.file.attrs:
                    compression = self.grp.file.attrs["default_dataset_compression"]
                else:
                    compression = None

                if "default_dataset_compression_opts" in self.grp.file.attrs:
                    compression_opts = self.grp.file.attrs[
                        "default_dataset_compression_opts"
                    ]
                else:
                    compression_opts = None

                dset = self.grp.create_dataset(
                    f"data_{n_new:04d}",
                    shape=new_data.shape,
                    chunks=new_chunks,
                    dtype=new_data.dtype,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            except:
                compression = None
                compression_opts = None

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
        self._n_t.append(n_new)
        self._n_d.append(n_new)

    def get_data_da(self, memlimit_MB=50):
        allarrays = []
        for n in self._n_t:
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
        return ArrayTimestamps(data=self.get_data_da(), index=self.timestamps)


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
        self._timestamps_finder = re.compile("^timestamps_[0-9]{4}$")
        self._n_d = []
        self._n_t = []
        with h5py.File(self.file_name, "r") as f:
            keys = f[self.group_name].keys()
            for key in keys:
                if self._data_finder.match(key):
                    self._n_d.append(int(key[-4:]))
                if self._timestamps_finder.match(key):
                    self._n_t.append(int(key[-4:]))
        self._n_d.sort()
        self._n_t.sort()
        if not self._n_d == self._n_t:
            raise Exception(
                "Corrupt escape ArrayH5Dataset, not equally sized data and index sub-datasets!"
            )

    @property
    def timestamps(self):
        if self._n_t:
            with h5py.File(self.file_name, "r") as f:
                return np.concatenate(
                    [
                        np.asarray(f[self.group_name][f"timestamps_{n:04d}"][:])
                        for n in self._n_t
                    ],
                    axis=0,
                )
        else:
            return np.asarray([], dtype=int)

    def append(self, data, timestamps, scan=None, prep_run=False):
        """
        expects to extend a former dataset, i.e. data includes data already existing,
        this will likely change in future to also allow real appending of entirely new data.
        """
        n_new = len(self._n_t)
        ids_stored = self.timestamps
        in_previous_timestamps = np.isin(timestamps, ids_stored)
        if ~in_previous_timestamps.any():
            # real appending data
            new_timestamps = timestamps
            new_data = data
        elif in_previous_timestamps.all():
            # real extending of data

            if len(timestamps) < len(ids_stored):
                raise Exception("fewer event_ids to append than already stored!")
            if not (timestamps[: len(ids_stored)] == ids_stored).all():
                raise Exception("new event_ids don't extend existing ones!")
            if len(timestamps) == len(ids_stored):
                print("Nothing new to append.")
                return

            new_timestamps = timestamps[len(ids_stored) :]
            new_data = data[len(ids_stored) :, ...]

        with h5py.File(self.file_name, "a") as f:
            f[self.group_name + f"/timestamps_{n_new:04d}"] = new_timestamps

        if isinstance(data, np.ndarray):
            if prep_run:
                if prep_run == "store_numpy":
                    print("this should happen")
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
        self._n_t.append(n_new)
        self._n_d.append(n_new)

    def get_data_da(self, memlimit_MB=50):
        allarrays = []
        for n in self._n_t:
            ds_name = self.group_name + f"/data_{n:04d}"
            h5store = self.analyze_h5_dataset(ds_name, memlimit_MB=memlimit_MB)
            tda = self.h5store_to_da(h5store=h5store)
            allarrays.append(tda)

        return da.concatenate(allarrays)

    def create_array(self):
        return ArrayTimestamps(data=self.get_data_da(), timestamps=self.timestamps)

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
