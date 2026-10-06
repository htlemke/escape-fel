"""Contract tests for escape results files: filename completion, overwrite
behaviour, which ``data`` types can be stored, failure atomicity, and the
caller patterns eco uses (counter, status server, archiver, bs-stream).

Headless, no EPICS; everything is written under pytest's ``tmp_path``.
Run just this file:
    python3 -m pytest tests/storage/test_result_files.py
"""
import io
import sys
import warnings

import dask.array as da
import h5py
import numpy as np
import pytest

import escape
from escape import Array, ArrayTimestamps, DataSet
from escape.storage.dataset import check_result_file, normalize_result_filepath
from escape.storage.storage_timestamps import ArrayH5Dataset

TS = np.arange(10.0)
INTERVALS = [(0.0, 5.0), (5.0, 10.0)]
PARAM = {"x": {"values": [0, 1]}}


def mk(data, ts=TS, name="m"):
    return ArrayTimestamps(
        data=data,
        timestamps=ts,
        timestamp_intervals=INTERVALS,
        parameter=PARAM,
        name=name,
    )


def store(path, arrays):
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        for name, a in arrays.items():
            ds.append(a, name=name)
            a.store()


def user_warnings(caught):
    return [w for w in caught if issubclass(w.category, UserWarning)]


# ------------------------------------------------------------ filenames

SUFFIX_CASES = [
    # given, file that must be used, warns?
    ("run", "run.esc.h5", False),
    ("scan_0.5V", "scan_0.5V.esc.h5", False),
    ("run.h5", "run.esc.h5", True),
    ("run.esc", "run.esc.h5", True),
    ("run.zarr", "run.esc.zarr", True),
    ("run.esc.h5", "run.esc.h5", False),
    ("run.esc.zarr", "run.esc.zarr", False),
]


@pytest.mark.parametrize("given,expect,warns", SUFFIX_CASES)
def test_normalize_result_filepath(given, expect, warns):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = normalize_result_filepath(given)
    assert out.name == expect
    caught = user_warnings(caught)
    assert len(caught) == (1 if warns else 0), [str(w.message) for w in caught]
    if warns:
        assert expect in str(caught[0].message)
        assert caught[0].filename == __file__  # points at the caller


@pytest.mark.parametrize(
    "given,result_type,expect,warns",
    [
        ("run", "zarr", "run.esc.zarr", False),
        ("run.esc.h5", "zarr", "run.esc.zarr", True),
        ("run.esc.zarr", "zarr", "run.esc.zarr", False),
    ],
)
def test_normalize_result_filepath_explicit_type(given, result_type, expect, warns):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = normalize_result_filepath(given, result_type)
    assert out.name == expect
    assert len(user_warnings(caught)) == (1 if warns else 0)


@pytest.mark.parametrize("entry", ["create", "init", "load"])
@pytest.mark.parametrize("given,expect,warns", SUFFIX_CASES)
def test_entry_points_complete_suffix(tmp_path, entry, given, expect, warns):
    if expect.endswith(".zarr"):
        pytest.importorskip("zarr")
    if entry == "load":
        DataSet.create_with_new_result_file(
            tmp_path / expect, force_overwrite=True
        ).close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entry == "create":
            ds = DataSet.create_with_new_result_file(
                tmp_path / given, force_overwrite=True
            )
        elif entry == "init":
            ds = DataSet(results_file=tmp_path / given, mode="w")
        else:
            ds = DataSet.load_from_result_file(tmp_path / given)
    ds.close()
    assert ds.results_filepath == tmp_path / expect
    assert (tmp_path / expect).exists()
    caught = user_warnings(caught)
    assert len(caught) == (1 if warns else 0), [str(w.message) for w in caught]
    if warns:
        assert caught[0].filename == __file__


def test_open_file_object_is_left_alone(tmp_path):
    f = h5py.File(tmp_path / "raw.h5", "w")
    ds = DataSet(results_file=f, mode="w")
    assert ds.results_file is f and ds.results_filepath is None
    f.close()


def test_unknown_type_after_esc_raises_valueerror(tmp_path):
    from escape.storage.dataset import filespec_to_file

    with pytest.raises(ValueError, match="'.esc.h5'"):
        filespec_to_file(tmp_path / "run.esc.txt", mode="w")


# ------------------------------------------------------------ overwrite


def test_existing_file_non_interactive_raises(tmp_path, monkeypatch):
    target = tmp_path / "exists.esc.h5"
    target.write_bytes(b"keep")
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(FileExistsError):
        DataSet.create_with_new_result_file(tmp_path / "exists")
    assert target.read_bytes() == b"keep"


class FakeTty(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("answer", ["n\n", "\n", ""])
def test_existing_file_declined_raises(tmp_path, monkeypatch, answer):
    target = tmp_path / "exists.esc.h5"
    target.write_bytes(b"keep")
    monkeypatch.setattr(sys, "stdin", FakeTty(answer))
    with pytest.raises(FileExistsError):
        DataSet.create_with_new_result_file(tmp_path / "exists")
    assert target.read_bytes() == b"keep"


def test_existing_file_confirmed_overwrites(tmp_path, monkeypatch):
    target = tmp_path / "exists.esc.h5"
    target.write_bytes(b"keep")
    monkeypatch.setattr(sys, "stdin", FakeTty("y\n"))
    DataSet.create_with_new_result_file(tmp_path / "exists").close()
    assert h5py.is_hdf5(target)


# ------------------------------------------------------------ data types

DATA_CASES = {
    "ndarray": (np.arange(10.0), np.arange(10.0)),
    "dask": (da.arange(10.0, chunks=4), np.arange(10.0)),
    "list": (list(np.arange(10.0)), np.arange(10.0)),
    "tuple": (tuple(range(10)), np.arange(10)),
    "list_of_lists": ([[i, i + 1] for i in range(10)], np.c_[TS, TS + 1]),
    "list_with_none": ([None] + list(range(1, 10)), np.r_[np.nan, 1:10]),
    "callable_list": (lambda: list(range(10)), np.arange(10)),
}


@pytest.mark.parametrize("case", sorted(DATA_CASES))
def test_data_roundtrip(tmp_path, case):
    data, expected = DATA_CASES[case]
    path = tmp_path / "rt.esc.h5"
    store(path, {"m": mk(data)})
    with DataSet.load_from_result_file(path) as ds:
        out = ds.datasets["m"]
        np.testing.assert_array_equal(np.asarray(out.data), expected)
        np.testing.assert_array_equal(out.timestamps, TS)


def test_empty_placeholder_is_accepted():
    a = mk(np.array([]), ts=np.array([]), name="empty")
    assert len(a.data) == 0


def test_list_timestamps_are_accepted(tmp_path):
    path = tmp_path / "ts.esc.h5"
    store(path, {"m": mk(list(range(10)), ts=list(TS))})
    with DataSet.load_from_result_file(path) as ds:
        np.testing.assert_array_equal(ds.datasets["m"].timestamps, TS)


def test_dask_and_callable_stay_lazy():
    a = mk(da.arange(10.0, chunks=5))
    assert isinstance(a._data, da.Array)
    calls = []
    b = mk(lambda: calls.append(1) or list(range(10)))
    assert calls == []
    assert isinstance(b.data, np.ndarray) and calls == [1]


def test_data_setter_coerces_and_is_idempotent():
    a = mk(np.arange(10.0))
    a.data = list(range(10))
    assert isinstance(a.data, np.ndarray)
    a.data = np.asarray(a.data)
    assert isinstance(a.data, np.ndarray)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="7.*10"):
        mk(np.arange(7.0))


def test_ragged_lists_raise():
    with pytest.raises(ValueError, match="ragged"):
        mk([[1, 2]] * 9 + [[1]])


def test_update_with_lists():
    c = mk(list(range(10))).update(mk(list(range(10, 20)), ts=TS + 20))
    assert len(c.data) == 20


def test_event_array_accepts_lists():
    a = Array(data=list(range(5)), index=np.arange(5), name="e")
    assert isinstance(a.data, (np.ndarray, da.Array))


# ------------------------------------------------------------ atomicity


def slot_keys(path, name="m"):
    with h5py.File(path) as f:
        return sorted(
            k for k in f[name].keys() if k.startswith(("timestamps_", "data_"))
        )


def test_object_dtype_raises_and_leaves_no_orphans(tmp_path):
    path = tmp_path / "obj.esc.h5"
    ds = DataSet.create_with_new_result_file(path, force_overwrite=True)
    with pytest.raises(TypeError, match="'/m'"):
        with ds:
            a = mk(np.array([object()] * 10))
            ds.append(a, name="m")
            a.store()
    assert not ds.results_file  # closed by the context manager
    assert slot_keys(path) == []


def test_failure_halfway_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / "half.esc.h5"

    def boom(*args, **kwargs):
        raise RuntimeError("killed half-way")

    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        a = mk(da.arange(10.0, chunks=5))
        ds.append(a, name="m")
        monkeypatch.setattr(da, "store", boom)
        with pytest.raises(RuntimeError, match="half-way"):
            a.store()
    assert slot_keys(path) == []


def test_partial_overlap_raises_without_orphans(tmp_path):
    path = tmp_path / "ov.esc.h5"
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        a = mk(np.arange(10.0))
        ds.append(a, name="m")
        a.store()
        with pytest.raises(ValueError, match="overlap"):
            ArrayH5Dataset(ds.results_file, "m").append(np.arange(10.0), TS + 5)
    assert slot_keys(path) == ["data_0000", "timestamps_0000"]


def test_extend_appends_only_new_values(tmp_path):
    path = tmp_path / "ext.esc.h5"
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        a = mk(np.arange(10.0))
        ds.append(a, name="m")
        a.store()
        ArrayH5Dataset(ds.results_file, "m").append(
            np.arange(15.0), np.arange(15.0)
        )
    with DataSet.load_from_result_file(path) as ds:
        np.testing.assert_array_equal(ds.datasets["m"].data, np.arange(15.0))


# ------------------------------------------------------------ diagnostics


def test_closed_file_says_so(tmp_path):
    path = tmp_path / "c.esc.h5"
    store(path, {"m": mk(np.arange(10.0))})
    r = DataSet.load_from_result_file(path)
    arr = r.datasets["m"]
    r.close()
    with pytest.raises(RuntimeError, match="closed"):
        np.asarray(arr.data)


def test_materialize_outlives_file(tmp_path):
    path = tmp_path / "mat.esc.h5"
    store(path, {"m": mk(np.arange(10.0))})
    r = DataSet.load_from_result_file(path)
    arr = r.datasets["m"].materialize()
    r.close()
    np.testing.assert_array_equal(arr.data, np.arange(10.0))
    assert len(arr.scan) == 2


def make_orphan_file(path):
    with h5py.File(path, "w") as f:
        g = f.require_group("m")
        g.attrs["esc_type"] = "array_timestamps_dataset"
        g["timestamps_0000"] = TS
        g2 = f.require_group("good")
        g2.attrs["esc_type"] = "array_timestamps_dataset"
        g2["timestamps_0000"] = TS
        g2["data_0000"] = TS * 2


def test_corrupt_group_names_file_and_group(tmp_path):
    path = tmp_path / "bad.esc.h5"
    make_orphan_file(path)
    with DataSet.load_from_result_file(path) as ds:
        with pytest.raises(ValueError, match=r"'/m' in results file .*bad\.esc\.h5"):
            ds.datasets["m"].data
        np.testing.assert_array_equal(ds.datasets["good"].data, TS * 2)


def test_check_result_file_and_repair(tmp_path):
    path = tmp_path / "bad.esc.h5"
    make_orphan_file(path)
    assert check_result_file(path) == {"m": "orphan_timestamps", "good": "ok"}
    check_result_file(path, repair=True)
    assert check_result_file(path) == {"good": "ok"}
    with DataSet.load_from_result_file(path) as ds:
        assert list(ds.datasets) == ["good"]


# ------------------------------------------------------------ eco callers


def test_eco_counter_pattern(tmp_path):
    """lists from CA monitors -> new file -> store -> reload -> use."""
    with DataSet.create_with_new_result_file(tmp_path / "myfile") as ds:
        for name in ("a", "b"):
            arr = mk([float(i) for i in range(10)], ts=list(TS), name=name)
            ds.append(arr, name=name)
            arr.store()
    r = DataSet.load_from_result_file(tmp_path / "myfile")
    arrays = {k: v.materialize() for k, v in r.datasets.items()}
    r.close()
    for arr in arrays.values():
        np.testing.assert_array_equal(arr.data, np.arange(10.0))


def test_eco_status_server_pattern(tmp_path):
    f = h5py.File(tmp_path / "status.esc.h5", "w")
    ds = DataSet(results_file=f, mode="w")
    arr = mk([1.0, 2.0], ts=[0.0, 1.0], name="s")
    ds.append(arr, name="s")
    arr.store()
    f.close()
    with DataSet.load_from_result_file(tmp_path / "status.esc.h5") as r:
        np.testing.assert_array_equal(r.datasets["s"].data, [1.0, 2.0])


def test_eco_archiver_pattern(tmp_path):
    path = tmp_path / "arch.esc.h5"
    h5py.File(path, "w").close()
    arr = mk(np.array([]), ts=np.array([]), name="pv")
    arr.set_h5_storage_file(path, "/", name="pv")
    arr.h5.append([1.0, None, 3.0], [0.0, 1.0, 2.0])
    arr.h5.append([1.0, None, 3.0, 4.0, 5.0], np.arange(5.0))  # extends
    # ArrayH5File groups carry no esc_type; eco reads them back through it
    np.testing.assert_array_equal(
        arr.h5.get_data_da().compute(), [1.0, np.nan, 3.0, 4.0, 5.0]
    )
    np.testing.assert_array_equal(arr.h5.timestamps, np.arange(5.0))


def test_eco_event_array_pattern(tmp_path):
    path = tmp_path / "bs.esc.h5"
    with DataSet.create_with_new_result_file(path) as ds:
        a = Array(
            data=np.arange(20.0),
            index=np.arange(20),
            step_lengths=[10, 10],
            parameter={"p": {"values": [1, 2]}},
            name="e",
        )
        ds.append(a, name="e")
        a.store()
    with DataSet.load_from_result_file(path) as r:
        e = r.datasets["e"]
        np.testing.assert_array_equal(np.asarray(e.data), np.arange(20.0))
        assert len(e.scan) == 2


def test_event_array_object_dtype_leaves_no_orphans(tmp_path):
    path = tmp_path / "ev.esc.h5"
    with DataSet.create_with_new_result_file(path) as ds:
        a = Array(data=np.array([object()] * 5), index=np.arange(5), name="e")
        ds.append(a, name="e")
        with pytest.raises(TypeError, match="'/e'"):
            a.store()
    with h5py.File(path) as f:
        assert [k for k in f["e"].keys() if k.startswith(("index_", "data_"))] == []


# ------------------------------------------------------------ old files


def test_reads_hand_written_old_layout(tmp_path):
    """The on-disk layout is unchanged: a file built by hand the way
    escape <= 0.2.14 wrote it (minus the scan group, which then loads as a
    single step) still loads."""
    path = tmp_path / "old.esc.h5"
    with h5py.File(path, "w") as f:
        g = f.require_group("m")
        g.attrs["esc_type"] = "array_timestamps_dataset"
        g["timestamps_0000"] = TS[:5]
        g["data_0000"] = TS[:5] * 2
        g["timestamps_0001"] = TS[5:]
        g["data_0001"] = TS[5:] * 2
    with DataSet.load_from_result_file(path) as ds:
        np.testing.assert_array_equal(ds.datasets["m"].data, TS * 2)
        np.testing.assert_array_equal(ds.datasets["m"].timestamps, TS)


def test_event_array_materialize_outlives_file(tmp_path):
    path = tmp_path / "evm.esc.h5"
    with DataSet.create_with_new_result_file(path) as ds:
        a = Array(data=np.arange(20.0), index=np.arange(20), step_lengths=[10, 10],
                  parameter={"p": {"values": [1, 2]}}, name="e")
        ds.append(a, name="e")
        a.store()
    r = DataSet.load_from_result_file(path)
    e = r.datasets["e"].materialize()
    r.close()
    np.testing.assert_array_equal(e.data, np.arange(20.0))
    np.testing.assert_array_equal(e.scan.nanmean(), [4.5, 14.5])
