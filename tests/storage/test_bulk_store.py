"""Storing many small ArrayTimestamps sharing one scan (eco's status-server
recording: ~10 000 channels, a few points each, one ScanTimestamps):
string channels, hard-linked scan groups and the bulk ``store()`` options.

Headless; everything is written under pytest's ``tmp_path``.
Run just this file:
    python3 -m pytest tests/storage/test_bulk_store.py
"""
import h5py
import numpy as np
import pytest

from escape import Array, ArrayTimestamps, DataSet
from escape.storage.storage_timestamps import ScanTimestamps

TS = np.array([0.1, 0.2])
INTERVALS = np.array([[0.0, 15.0]])
PARAM = {"recording": {"values": [0.0], "attributes": {"note": "window"}}}


def mk(data, name, intervals=INTERVALS):
    return ArrayTimestamps(
        data=data,
        timestamps=TS,
        timestamp_intervals=intervals,
        parameter=PARAM,
        name=name,
    )


def store(path, arrays, **store_kwargs):
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        for name, a in arrays.items():
            ds.append(a, name=name)
            a.store(**store_kwargs)


def objno(group):
    return h5py.h5g.get_objinfo(group.id).objno


def intervals(f, name):
    return ScanTimestamps._load_from_h5(f[name])[1]


# ------------------------------------------------------------ string channels

STRINGS = np.array(["Open", "Closed"]).astype(h5py.string_dtype("utf-8"))


def test_h5py_string_channel_roundtrip(tmp_path):
    path = tmp_path / "s.esc.h5"
    store(path, {"num": mk(np.array([1.0, 2.0]), "num"), "enum": mk(STRINGS, "enum")})
    with DataSet.load_from_result_file(path) as ds:
        out = np.asarray(ds.datasets["enum"].data)
        assert list(out) == ["Open", "Closed"]
        assert all(isinstance(v, str) for v in out)
        np.testing.assert_array_equal(ds.datasets["num"].data, [1.0, 2.0])


def test_h5py_string_event_array_roundtrip(tmp_path):
    path = tmp_path / "e.esc.h5"
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        a = Array(data=STRINGS, index=np.arange(2), name="e")
        ds.append(a, name="e")
        a.store()
    with DataSet.load_from_result_file(path) as ds:
        assert list(np.asarray(ds.datasets["e"].data)) == ["Open", "Closed"]


def test_plain_object_data_still_refused(tmp_path):
    with DataSet.create_with_new_result_file(tmp_path / "o.esc.h5") as ds:
        a = mk(np.array([object(), object()]), "o")
        ds.append(a, name="o")
        with pytest.raises(TypeError, match="no native HDF5"):
            a.store()


# ------------------------------------------------------------ shared scan groups


def test_identical_scans_share_one_group(tmp_path):
    path = tmp_path / "l.esc.h5"
    # separate but equal ScanTimestamps, as ArrayTimestamps builds one each
    store(path, {f"ch{i}": mk(np.arange(2.0) + i, f"ch{i}") for i in range(3)})
    with h5py.File(path, "r") as f:
        assert len({objno(f[f"ch{i}"]["scan"]) for i in range(3)}) == 1
        assert sorted(f.keys()) == ["ch0", "ch1", "ch2"]  # nothing extra at root
    with DataSet.load_from_result_file(path) as ds:
        for i in range(3):
            a = ds.datasets[f"ch{i}"]
            np.testing.assert_array_equal(a.scan.timestamp_intervals, INTERVALS)
            assert a.scan.parameter["recording"]["attributes"]["note"] == b"window"
            np.testing.assert_array_equal(a.data, np.arange(2.0) + i)


def test_different_scans_are_not_shared(tmp_path):
    path = tmp_path / "d.esc.h5"
    store(path, {"a": mk([1.0, 2.0], "a"), "b": mk([1.0, 2.0], "b", INTERVALS + 100)})
    with h5py.File(path, "r") as f:
        assert objno(f["a"]["scan"]) != objno(f["b"]["scan"])
        np.testing.assert_array_equal(intervals(f, "b"), INTERVALS + 100)


def test_shared_scan_safety(tmp_path):
    """The four properties linking relies on, through escape's own writes."""
    path = tmp_path / "s.esc.h5"
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        arrays = {f"ch{i}": mk(np.arange(2.0), f"ch{i}") for i in range(3)}
        for name, a in arrays.items():
            ds.append(a, name=name)
            a.store()
        f = ds.results_file
        assert h5py.h5g.get_objinfo(f["ch0"]["scan"].id).nlink == 3

        # re-storing one array with another scan leaves the others alone
        ScanTimestamps(
            parameter={"recording": {"values": [1.0]}},
            timestamp_intervals=INTERVALS + 100,
        )._save_to_h5(f["ch1"])
        np.testing.assert_array_equal(intervals(f, "ch1"), INTERVALS + 100)
        np.testing.assert_array_equal(intervals(f, "ch0"), INTERVALS)
        np.testing.assert_array_equal(intervals(f, "ch2"), INTERVALS)

        # removing the first writer keeps the shared group readable
        del f["ch0"]
        np.testing.assert_array_equal(intervals(f, "ch2"), INTERVALS)

        # an array stored afterwards still gets the right scan
        late = mk(np.arange(2.0), "late")
        ds.append(late, name="late")
        late.store()
        np.testing.assert_array_equal(intervals(f, "late"), INTERVALS)

    # copying one channel out makes a real copy
    out = tmp_path / "one.h5"
    with h5py.File(path, "r") as f, h5py.File(out, "w") as g:
        f.copy("ch2", g, name="ch2")
    with h5py.File(out, "r") as g:
        np.testing.assert_array_equal(intervals(g, "ch2"), INTERVALS)


def test_restore_first_writer_with_same_scan(tmp_path):
    """Re-storing the array that holds the canonical copy must not leave
    later arrays linked to a group that changed or vanished."""
    path = tmp_path / "r.esc.h5"
    with DataSet.create_with_new_result_file(path, force_overwrite=True) as ds:
        f = ds.results_file
        a = mk([1.0, 2.0], "a")
        ds.append(a, name="a")
        a.store()
        other = ScanTimestamps(
            parameter={"recording": {"values": [9.0]}},
            timestamp_intervals=INTERVALS + 50,
        )
        other._save_to_h5(f["a"])  # a's scan replaced, the old group freed
        b = mk([1.0, 2.0], "b")
        ds.append(b, name="b")
        b.store()
        np.testing.assert_array_equal(intervals(f, "a"), INTERVALS + 50)
        np.testing.assert_array_equal(intervals(f, "b"), INTERVALS)


# ------------------------------------------------------------ store() options


def _file_contents(path):
    out = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda name, obj: out.__setitem__(
                name,
                np.asarray(obj[()]).tolist() if isinstance(obj, h5py.Dataset) else None,
            )
        )
    return out


def test_bulk_options_write_the_same_file(tmp_path):
    def arrays():
        return {f"ch{i}": mk(np.arange(2.0) * i, f"ch{i}") for i in range(4)}

    store(tmp_path / "a.esc.h5", arrays())
    store(tmp_path / "b.esc.h5", arrays(), flush=False, read_back=False)
    assert _file_contents(tmp_path / "a.esc.h5") == _file_contents(tmp_path / "b.esc.h5")


def test_read_back_false_keeps_memory_data(tmp_path):
    with DataSet.create_with_new_result_file(tmp_path / "m.esc.h5") as ds:
        a = mk(np.array([3.0, 4.0]), "m")
        ds.append(a, name="m")
        a.store(read_back=False)
        assert isinstance(a.data, np.ndarray)
        np.testing.assert_array_equal(a.data, [3.0, 4.0])
        b = mk(np.array([3.0, 4.0]), "n")
        ds.append(b, name="n")
        b.store()
        assert not isinstance(b.data, np.ndarray)  # now reads from the file
        np.testing.assert_array_equal(np.asarray(b.data), [3.0, 4.0])
