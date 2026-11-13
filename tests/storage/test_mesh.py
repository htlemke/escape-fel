"""arr.mesh (escape.storage.mesh): every scan's mesh -- a line, a grid, an
adaptive subset of a grid, points anywhere -- with per-node results that carry
their coordinates, binning before and interpolation after a method, and the
recorded order of the steps."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from escape import Array
from escape.storage.mesh import Mesh, MeshArray


def _array(step_values, per_step=4, parameter=None, grid_specs=None, mesh_specs=None):
    """Steps with `per_step` events each, every event of step i = step_values[i]."""
    data = np.concatenate([np.full(per_step, v, dtype=float) for v in step_values])
    return Array(
        data=data, index=np.arange(len(data)), step_lengths=[per_step] * len(step_values),
        parameter=parameter, grid_specs=grid_specs, mesh_specs=mesh_specs,
    )


def _infos(**columns):
    keys = list(columns)
    n = len(columns[keys[0]])
    return {"scan_step_info": {"values": [{k: columns[k][i] for k in keys} for i in range(n)]}}


# -- a line ------------------------------------------------------------------------
def test_line_merges_repetitions_and_keeps_the_order():
    x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    a = _array([10 * v + i for i, v in enumerate(x)], parameter={"x": {"values": x}})
    m = a.mesh
    assert m.kind == "line" and m.is_grid and len(m) == 3
    assert list(m.step_nodes) == [0, 1, 2, 0, 1, 2]
    assert list(m.visits()) == [2, 2, 2]
    assert m.node_steps == [[0, 3], [1, 4], [2, 5]]
    r = m.mean()
    assert isinstance(r, MeshArray) and r.kind == "grid"
    assert list(r) == [1.5, 12.5, 23.5]  # events of both visits pooled
    assert list(r.coords[0]) == [0.0, 1.0, 2.0] and r.dimension_names == ["x"]
    steps = m.mean(merge=False)
    assert steps.kind == "steps" and list(steps) == [0, 11, 22, 3, 14, 25]
    assert list(m.step_coords[:, 0]) == x
    assert list(m.count()) == [8, 8, 8]
    assert a.scan.mean() == [0, 11, 22, 3, 14, 25]  # the scan: unchanged


def test_readbacks_are_not_dimensions():
    a = _array([1, 2], parameter={"x": {"values": [0.0, 1.0]}, "x_readback": {"values": [0.01, 0.99]}})
    assert a.mesh.dimension_names == ["x"]


# -- a grid ------------------------------------------------------------------------
GRID = {"shape": [2, 3], "positions": [[0.0, 1.0], [10.0, 20.0, 30.0]],
        "grid_dimension_names": ["y", "x"]}


def _grid_array(cells=((0, 0), (0, 1), (0, 2), (1, 0), (1, 2))):
    values = [100 * i + j for i, j in cells]
    return _array(values, parameter=_infos(grid_index=[list(c) for c in cells]), grid_specs=GRID)


def test_grid_results_are_nd_with_nan_where_not_measured():
    r = _grid_array().mesh.mean()
    assert r.shape == (2, 3) and r.kind == "grid"
    assert np.isnan(r[1, 1]) and r[1, 2] == 102
    assert [list(c) for c in r.coords] == GRID["positions"]


def test_grid_selection_and_result_slicing_keep_coordinates():
    a = _grid_array()
    line = a.mesh[:, 2].mean()
    assert list(line) == [2, 102] and line.dimension_names == ["y"]
    assert list(line.coords[0]) == [0.0, 1.0]
    r = a.mesh.mean()
    row = r[0]
    assert list(row.coords[0]) == [10.0, 20.0, 30.0] and row.dimension_names == ["x"]
    assert (r * 2).coords is r.coords  # element-wise keeps them
    assert type(r.sum(axis=0)) is np.ndarray and not isinstance(r.mean(), MeshArray)
    assert type(r[[0, 1]]) is np.ndarray  # fancy indexing: plain
    single = a.mesh[1, 2]  # one grid point: its steps as an Array
    assert isinstance(single, Array) and float(single.mean()) == 102


def test_a_mesh_that_is_not_a_grid_says_how_to_select():
    specs = {"dimension_names": ["a", "b"], "units": ["mm", "mm"], "kind": "scattered",
             "resolution": None, "points": [[0, 0], [1, 0.2], [0.3, 1], [1, 1]]}
    a = _array([1, 2, 3, 4], parameter=_infos(mesh_index=[0, 1, 2, 3]), mesh_specs=specs)
    with pytest.raises(TypeError, match="as_grid"):
        a.mesh[:, 0]
    r = a.mesh.mean()
    assert r.kind == "nodes" and r.shape == (4,) and r.coords[0].shape == (4, 2)
    sub = a.mesh.select({"a": (0.5, 2)})
    assert list(sub.mean()) == [2, 4]
    assert list(a.mesh.select(b=(0.5, 2)).mean()) == [3, 4]


# -- eco's adaptive scans: a refined subset of a grid ------------------------------------
def test_adapted_mesh_from_eco_and_its_original_grid():
    specs = {"dimension_names": ["mot"], "units": ["mm"], "kind": "adapted",
             "resolution": [0.5], "points": [[0.0], [2.0], [1.0], [1.5]]}
    # step order as measured, the second visit of node 1 (x = 2) last
    a = _array([0, 2, 1, 1.5, 2.2], parameter=_infos(mesh_index=[0, 1, 2, 3, 1]), mesh_specs=specs)
    m = a.mesh
    assert m.kind == "adapted" and m.is_grid and m.units == ["mm"]
    assert list(m.mean()) == [0, 1, 1.5, 2.1]  # sorted along the line, x = 2 pooled
    g = m.as_grid("auto")  # the axis grid at the resolution: 0, 0.5, ..., 2
    r = g.mean()
    assert list(r.coords[0]) == [0, 0.5, 1, 1.5, 2] and np.isnan(r[1])


# -- binning: before the method ---------------------------------------------------------
def test_bins_pool_the_events_and_link_given_edges():
    points = [[0.1, 0.1], [0.2, 0.15], [0.8, 0.9], [0.9, 0.7]]
    specs = {"dimension_names": ["a", "b"], "points": points}
    a = _array([1, 3, 10, 20], per_step=2, parameter=_infos(mesh_index=[0, 1, 2, 3]),
               mesh_specs=specs)
    edges_a, edges_b = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])
    r = a.mesh.mean(bins=[edges_a, edges_b])
    assert r.kind == "binned" and r.shape == (2, 2)
    assert r[0, 0] == 2 and r[1, 1] == 15 and np.isnan(r[0, 1])
    assert r.edges[0] is edges_a and r.edges[1] is edges_b  # links, not copies
    assert list(r.coords[0]) == [0.25, 0.75]
    assert a.mesh.as_grid(bins=2)[1, :].mean().shape == (2,)  # then a grid line
    counts = a.mesh.count(bins=2)
    assert counts[0, 0] == 4 and np.isnan(counts[0, 1])  # NaN: nothing measured


# -- interpolation: after the method -------------------------------------------------------
def test_interpolate_onto_a_grid_masked_far_from_nodes():
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 1, size=(40, 2))
    specs = {"dimension_names": ["a", "b"], "points": points.tolist()}
    plane = 2 * points[:, 0] + 3 * points[:, 1]
    a = _array(plane, parameter=_infos(mesh_index=list(range(40))), mesh_specs=specs)
    xs, ys = np.linspace(0.2, 0.8, 7), np.linspace(0.2, 0.8, 5)
    r = a.mesh.mean(interpolate=[xs, ys])
    assert r.kind == "interpolated" and r.shape == (7, 5)
    assert r.coords[0] is xs
    expected = 2 * xs[:, None] + 3 * ys[None, :]
    assert np.allclose(r[~r.mask], expected[~r.mask])
    far = a.mesh.mean(interpolate={"positions": [np.array([5.0]), np.array([5.0])]})
    assert far.mask.all() and np.isnan(far).all()
    with pytest.raises(ValueError):
        a.mesh.mean(bins=2, interpolate=True)


def test_interpolate_a_line():
    a = _array([0, 10, 20], parameter={"x": {"values": [0.0, 1.0, 2.0]}})
    r = a.mesh.mean(interpolate=[np.array([0.5, 1.5])])
    assert list(r) == [5, 15]


# -- plots -------------------------------------------------------------------------------
def test_plots_of_every_kind():
    import matplotlib.pyplot as plt

    _grid_array().mesh.mean(plot=True)
    _array([0, 1, 4], parameter={"x": {"values": [0.0, 1.0, 2.0]}}).mesh.mean(plot=True)
    rng = np.random.default_rng(2)
    points = rng.uniform(0, 1, size=(12, 2))
    specs = {"dimension_names": ["a", "b"], "points": points.tolist()}
    a = _array(points.sum(axis=1), parameter=_infos(mesh_index=list(range(12))), mesh_specs=specs)
    ax = a.mesh.mean().plot()
    assert ax.get_xlabel() == "a"
    a.mesh.mean(interpolate=20, plot=True)
    plt.close("all")


# -- persistence and sub-arrays ---------------------------------------------------------------
def test_saved_and_loaded_with_its_mesh(tmp_path):
    from escape import DataSet

    specs = {"dimension_names": ["mot"], "units": ["mm"], "kind": "adapted",
             "resolution": [0.5], "points": [[0.0], [2.0], [1.0]]}
    a = _array([0, 2, 1, 2.2], parameter=_infos(mesh_index=[0, 1, 2, 1]), mesh_specs=specs)
    a.name = "sig"
    expected_nodes, expected = list(a.mesh.step_nodes), list(a.mesh.mean())
    path = tmp_path / "m.esc.h5"
    d = DataSet.create_with_new_result_file(path)
    d.append(a, name="sig")
    a.store()
    d.results_file.close()
    ds = DataSet.load_from_result_file(path)  # keeps the file open for the lazy data
    b = ds.datasets["sig"]
    assert b.mesh.kind == "adapted" and list(b.mesh.step_nodes) == expected_nodes
    assert list(b.mesh.mean()) == expected  # (a itself now reads the closed file)


def test_sub_arrays_keep_the_mesh():
    specs = {"dimension_names": ["a", "b"], "points": [[0, 0], [1, 0], [0, 1]]}
    a = _array([1, 2, 3], parameter=_infos(mesh_index=[0, 1, 2]), mesh_specs=specs)
    sub = a.scan[[1, 2]]
    assert sub.mesh.dimension_names == ["a", "b"] and len(sub.mesh) == 2
    assert list(sub.mesh.mean()) == [2, 3]


# -- the geometry alone ------------------------------------------------------------------
def test_mesh_without_data():
    g = Mesh.from_grid([[0, 1], [5, 6, 7]], dimension_names=["y", "x"])
    assert g.is_grid and g.shape == (2, 3) and len(g) == 6
    p = Mesh.from_points([[0, 0], [1, 0], [0, 1], [1, 0]])
    assert len(p) == 3 and not p.is_grid and p.kind == "scattered"
    assert len(p.triangulation().triangles) == 1


def test_a_2d_grid_plot_draws_a_map():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _grid_array().mesh.mean(plot=ax)
    assert ax.collections or ax.images  # the colour map is there
    assert ax.get_xlabel() == "x" and ax.get_ylabel() == "y"
    plt.close("all")


# -- binning a per-shot channel into a scan axis -------------------------------------
def _digitized_tt_grid(bin_width=50.0, unit="fs"):
    """An energy scan whose shots also carry a jittering arrival time, binned
    into a second axis -- the timetool-binned EXAFS scan, in miniature."""
    from escape.storage.storage import unravel_scans

    rng = np.random.default_rng(0)
    energies = [7100.0, 7101.0, 7102.0]
    per_step = 200
    signal = np.concatenate(
        [np.full(per_step, e - 7100.0) for e in energies]
    )
    arrival = rng.uniform(-100.0, 100.0, size=len(signal))
    index = np.arange(len(signal))
    sig = Array(data=signal, index=index, step_lengths=[per_step] * len(energies),
                parameter={"energy": {"values": energies}})
    tt = Array(data=arrival, index=index, step_lengths=[per_step] * len(energies),
               parameter={"energy": {"values": energies}}, name="tt.edge_fs")
    binned = tt.digitize(bin_width=bin_width, range=(-100.0, 100.0), unit=unit)
    return sig, tt, binned, unravel_scans(sig, binned, categorize_target=sig)


def test_digitize_by_bin_width_is_one_axis_named_after_the_channel():
    _, _, binned, _ = _digitized_tt_grid()
    m = binned.mesh
    # bin_left_/bin_right_ describe the same axis as bin_center_, not two more
    assert m.ndim == 1 and m.kind == "line"
    assert m.dimension_names == ["tt.edge_fs"] and m.units == ["fs"]
    # 50 fs bins on the roundto grid: the centers are the multiples of 50
    assert len(m) == 5
    assert list(m.nodes.ravel()) == [-100.0, -50.0, 0.0, 50.0, 100.0]


def test_bin_edges_sit_on_multiples_of_the_width():
    from escape.storage.storage import bin_edges

    # the default grid is the roundto one: bin centers on the multiples
    assert list(bin_edges([-512.0, 437.0], bin_width=100.0)) == list(
        np.arange(-550.0, 451.0, 100.0)
    )
    assert list(bin_edges([-512.0, 437.0], bin_width=100.0, align="center")) == list(
        np.arange(-550.0, 451.0, 100.0)
    )
    assert list(bin_edges([-512.0, 437.0], bin_width=100.0, align="edge")) == list(
        np.arange(-600.0, 501.0, 100.0)
    )
    assert list(bin_edges(n_bins=4, range=(0.0, 4.0))) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(bin_edges(bin_width=50.0, range=(-100.0, 100.0), align="edge")) == [
        -100.0, -50.0, 0.0, 50.0, 100.0
    ]
    with pytest.raises(ValueError):
        bin_edges([0.0, 1.0], bin_width=1.0, n_bins=2)
    with pytest.raises(ValueError):
        bin_edges(bin_width=1.0)


def test_a_scan_axis_crossed_with_binned_shots_is_a_grid():
    sig, _, _, grid = _digitized_tt_grid()
    m = grid.mesh
    assert m.is_grid and m.shape == (3, 5)
    assert m.dimension_names == ["energy", "tt.edge_fs"]
    assert m.units == ["", "fs"]
    means = m.nanmean()
    assert means.shape == (3, 5)
    # the signal depends on the energy only, so every timetool bin of a row agrees
    for row, energy in zip(means, [7100.0, 7101.0, 7102.0]):
        assert np.allclose(row, energy - 7100.0)
    assert list(means.coords[0]) == [7100.0, 7101.0, 7102.0]
    assert list(means.coords[1]) == [-100.0, -50.0, 0.0, 50.0, 100.0]
    assert m.nanmean().shape == m.count().shape


def test_grid_dimensions_can_be_selected_by_the_channel_name():
    _, _, _, grid = _digitized_tt_grid()
    picked = grid.mesh.select({"tt.edge_fs": (-50.0, 50.0)})
    assert picked.shape == (3, 3)
    assert list(picked.mesh.positions[1]) == [-50.0, 0.0, 50.0]


def test_digitize_refuses_bins_and_a_width_together():
    _, tt, _, _ = _digitized_tt_grid()
    with pytest.raises(ValueError):
        tt.digitize(np.arange(-100.0, 101.0, 50.0), bin_width=50.0)


def test_many_steps_do_not_make_an_endless_repr(capsys):
    """repr is the overview; the per-step picture is what show() is for."""
    _, _, _, grid = _digitized_tt_grid(bin_width=10.0)

    grid = grid.compute()
    assert len(grid.scan) == 63  # 3 energies x 21 timetool bins
    lines = repr(grid).splitlines()
    assert not [l for l in lines if l.startswith("Step ")]
    assert len(lines) < 15
    assert any(".show()" in l for l in lines)

    grid.show()
    shown = capsys.readouterr().out.splitlines()
    assert len([l for l in shown if l.startswith("Step ")]) == 63

    grid.show(max_steps=10)
    shown = capsys.readouterr().out.splitlines()
    assert len([l for l in shown if l.startswith("Step ")]) == 10
    assert any("further steps" in l for l in shown)

    grid.show(per_step=False)
    assert not [l for l in capsys.readouterr().out.splitlines() if l.startswith("Step ")]


def test_selecting_events_keeps_the_name_and_the_grid():
    """A filtered channel is still that channel: `tt[amp > x].digitize(...)`
    must name its axis after the timetool, not after a local variable."""
    _, tt, _, grid = _digitized_tt_grid()
    good = tt[(tt > -60.0) & (tt < 60.0)]
    assert good.name == "tt.edge_fs" and len(good) < len(tt)
    assert good.digitize(bin_width=50.0, unit="fs").mesh.dimension_names == ["tt.edge_fs"]

    kept = grid[grid.index % 2 == 0]
    assert kept.mesh.is_grid and kept.mesh.dimension_names == grid.mesh.dimension_names
    assert kept.mesh.units == ["", "fs"]


def test_compute_keeps_the_name_and_the_grid():
    _, _, _, grid = _digitized_tt_grid()
    grid.name = "mu"
    computed = grid.compute()
    assert computed.name == "mu"
    assert computed.mesh.is_grid and computed.mesh.shape == (3, 5)
    assert computed.mesh.dimension_names == ["energy", "tt.edge_fs"]
    assert np.allclose(np.asarray(computed.mesh.nanmean()),
                       np.asarray(grid.mesh.nanmean()), equal_nan=True)


# -- the fast path must not change any answer ----------------------------------------
@pytest.mark.parametrize("method", [
    "nanmean", "nanstd", "nanmedian", "nansum", "nanmin", "nanmax",
    "mean", "std", "median", "min", "max", "all", "any", "count", "nancount",
])
def test_direct_reduction_agrees_with_the_array_method(method, monkeypatch):
    """Reductions take a shortcut past Array/escaped on numpy data; the values
    must be the ones the ordinary per-step path gives."""
    from escape.storage.mesh import ArrayMesh

    _, _, _, grid = _digitized_tt_grid(bin_width=25.0)
    fast = np.asarray(getattr(grid.mesh, method)(), dtype=float)

    monkeypatch.setattr(ArrayMesh, "_direct_reduction", lambda self, n, k: None)
    slow = np.asarray(getattr(grid.mesh, method)(), dtype=float)

    assert fast.shape == slow.shape
    assert np.allclose(fast, slow, equal_nan=True)


def test_methods_with_arguments_keep_the_ordinary_path():
    """An `axis` or an escape-only method must not be shortcut."""
    from escape.storage.mesh import ArrayMesh

    _, _, _, grid = _digitized_tt_grid()
    mesh = grid.mesh
    assert mesh._direct_reduction("nanmean", {}) is np.nanmean
    assert mesh._direct_reduction("nanmean", {"axis": 0}) is None   # axis given
    assert mesh._direct_reduction("isnan", {}) is None              # element-wise
    assert mesh._direct_reduction("nancount", {}) is None           # escape-only
    # a positional argument is forwarded, and the answer still matches nanmedian
    assert np.allclose(np.asarray(mesh.nanpercentile(50)),
                       np.asarray(mesh.nanmedian()), equal_nan=True)


def test_node_steps_keeps_the_recorded_order():
    x = [0.0, 1.0, 2.0, 1.0, 0.0, 1.0]
    a = _array([10 * v + i for i, v in enumerate(x)], parameter={"x": {"values": x}})
    assert a.mesh.node_steps == [[0, 4], [1, 3, 5], [2]]
    assert a.mesh.node_groups() == a.mesh.node_steps
    assert list(a.mesh.visits()) == [2, 3, 1]


def test_lazy_results_are_batched_and_unchanged(monkeypatch):
    import dask.array as da
    from escape.storage import mesh as mesh_module

    x = [0.0, 1.0, 2.0] * 4
    values = np.concatenate([np.full(5, v) + np.arange(5) for v in x])
    lazy = Array(data=da.from_array(values, chunks=7), index=np.arange(len(values)),
                 step_lengths=[5] * len(x), parameter={"x": {"values": x}})
    eager = Array(data=values, index=np.arange(len(values)),
                  step_lengths=[5] * len(x), parameter={"x": {"values": x}})

    monkeypatch.setattr(mesh_module, "MESH_COMPUTE_BATCH_NODES", 2)   # force batching
    assert np.allclose(np.asarray(lazy.mesh.nanmean()),
                       np.asarray(eager.mesh.nanmean()), equal_nan=True)


# -- the methods arr.grid has, arr.mesh has too ---------------------------------------
def test_mesh_has_every_step_method_grid_has():
    from escape.storage.mesh import ArrayMesh
    from escape.storage.storage import SCAN_STEP_METHODS

    assert not set(SCAN_STEP_METHODS) - set(dir(ArrayMesh))


@pytest.mark.parametrize("method", ["median_and_mad", "weighted_avg_and_std",
                                    "weighted_stat"])
def test_several_values_per_node_come_back_as_several_meshes(method):
    from escape.storage.example_data import make_grid_scan

    sig = make_grid_scan(shape=(4, 5), n_events_per_step=40, seed=1)
    result = getattr(sig.mesh, method)()
    assert isinstance(result, tuple) and len(result) == 2
    for part in result:
        assert isinstance(part, MeshArray) and part.kind == "grid"
        assert part.shape[:2] == (4, 5)          # weighted_stat's error is (4, 5, 2)
        assert len(part.coords[0]) == 4 and len(part.coords[1]) == 5
    # one step per node here, so the mesh and the grid must agree on the value
    # (weighted_stat's error is per-step (2, n) on a Scan, which is the shape
    # errorbar wants and not one the grid can lay out -- the mesh gives it per
    # node instead, as a trailing axis)
    on_grid = getattr(sig.grid, method)()
    pairs = zip(result, on_grid) if method != "weighted_stat" else [(result[0], on_grid[0])]
    for mesh_part, grid_part in pairs:
        assert np.allclose(np.asarray(mesh_part), np.asarray(grid_part), equal_nan=True)


def test_one_value_per_node_still_comes_back_bare():
    from escape.storage.example_data import make_grid_scan

    sig = make_grid_scan(shape=(3, 4), n_events_per_step=20, seed=0)
    assert isinstance(sig.mesh.nanmean(), MeshArray)
    assert isinstance(sig.mesh.count(), MeshArray)


def test_several_values_per_node_on_a_line_and_merged():
    x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    a = _array([10 * v + i for i, v in enumerate(x)], per_step=8,
               parameter={"x": {"values": x}})
    med, mad = a.mesh.median_and_mad()
    assert med.shape == (3,) and mad.shape == (3,)
    assert list(med.coords[0]) == [0.0, 1.0, 2.0]
    # both visits of a node pooled, as for every other mesh method
    assert np.allclose(np.asarray(med), np.asarray(a.mesh.nanmedian()), equal_nan=True)
    per_step = a.mesh.median_and_mad(merge=False)[0]
    assert per_step.shape == (6,)


def test_array_median_and_mad_works_with_its_own_defaults():
    """axis=None is 'over everything', not a broken selector."""
    values = np.array([1.0, 2.0, 3.0, 100.0])
    a = Array(data=values, index=np.arange(4), step_lengths=[4])
    med, mad = a.median_and_mad(k_dist=1.0)
    assert med == 2.5
    assert mad == np.median(np.abs(values - 2.5))
    assert np.isscalar(med) or np.ndim(med) == 0


def test_abs_is_element_wise_and_keeps_the_array():
    a = Array(data=np.array([-1.0, 2.0, -3.0]), index=np.arange(3), step_lengths=[3])
    out = a.abs()
    assert isinstance(out, Array)
    assert list(np.asarray(out.data)) == [1.0, 2.0, 3.0]
    assert list(out.index) == [0, 1, 2]


def test_to_grid_refuses_values_that_are_not_per_step():
    from escape.storage.example_data import make_grid_scan

    sig = make_grid_scan(shape=(3, 4), n_events_per_step=10, seed=0)
    grid = sig.grid
    assert grid.to_grid(np.arange(12.0)).shape == (3, 4)
    with pytest.raises(ValueError):
        grid.to_grid(np.zeros((2, 12)))     # the errorbar layout, not per step


# -- bin edges from a width cover the data -------------------------------------------
def test_bin_edges_cover_the_range_in_both_alignments():
    from escape.storage.storage import bin_edges

    for align in ("edge", "center"):
        for lo, hi, width in [(-512.0, 437.0, 100.0), (-500.0, 400.0, 100.0),
                              (-100.0, 100.0, 50.0), (0.0, 100.0, 25.0),
                              (-0.3, 0.7, 0.1)]:
            edges = bin_edges([lo, hi], bin_width=width, align=align)
            assert edges[0] <= lo and edges[-1] >= hi, (align, lo, hi, width)
            assert np.allclose(np.diff(edges), width)
    # centers of an "center"-aligned grid sit on multiples of the width
    edges = bin_edges([-512.0, 437.0], bin_width=100.0, align="center")
    centers = (edges[1:] + edges[:-1]) / 2
    assert np.allclose(centers / 100.0, np.round(centers / 100.0))


def test_bin_edges_refuses_a_width_in_the_wrong_units():
    from escape.storage.storage import bin_edges, MAX_BINS

    with pytest.raises(ValueError, match="units"):
        bin_edges([-1.0, 8.8], bin_width=20e-15)          # seconds against ps
    assert len(bin_edges([0.0, 1.0], n_bins=MAX_BINS)) == MAX_BINS + 1
    with pytest.raises(ValueError):
        bin_edges([0.0, 1.0], n_bins=MAX_BINS + 1)


def test_timetool_binning_can_cross_the_runs_own_scan():
    """An energy scan binned in arrival time: the delay bins must become a
    second axis, not replace the energy one."""
    from escape.storage.storage import unravel_scans

    sig, tt, binned, by_hand = _digitized_tt_grid()
    by_tool = sig.tools.timetool_binning(
        tt, time_vec=None, time_bins=50.0, time_range=(-100.0, 100.0),
        cross_scan=True, unit="fs",
    )
    assert by_tool.mesh.is_grid and by_tool.mesh.shape == by_hand.mesh.shape
    assert by_tool.mesh.dimension_names == by_hand.mesh.dimension_names
    assert by_tool.mesh.units == by_hand.mesh.units
    # the delay axis positions are readable off the ArrayMesh itself
    assert list(by_tool.mesh.positions[1]) == [-100.0, -50.0, 0.0, 50.0, 100.0]
    assert np.allclose(np.asarray(by_tool.mesh.nanmean()),
                       np.asarray(by_hand.mesh.nanmean()), equal_nan=True)

    # without cross_scan the energy axis is replaced by the delay bins
    flat = sig.tools.timetool_binning(
        tt, time_vec=None, time_bins=50.0, time_range=(-100.0, 100.0), unit="fs",
    )
    assert flat.mesh.ndim == 1
    assert flat.mesh.dimension_names == ["tt.edge_fs"]
    assert len(flat.mesh) == by_hand.mesh.shape[1]


def test_timetool_binning_and_digitize_put_bins_on_the_same_grid():
    """Both bin a width onto the roundto grid, so the two paths agree."""
    sig, tt, binned, _ = _digitized_tt_grid()
    flat = sig.tools.timetool_binning(
        tt, time_vec=None, time_bins=50.0, time_range=(-100.0, 100.0),
    )
    assert list(flat.mesh.nodes.ravel()) == list(binned.mesh.nodes.ravel())


def test_timetool_binning_refuses_a_range_with_explicit_edges():
    sig, tt, _, _ = _digitized_tt_grid()
    with pytest.raises(ValueError, match="time_range"):
        sig.tools.timetool_binning(
            tt, time_vec=None, time_bins=np.arange(-100.0, 101.0, 50.0),
            time_range=(-100.0, 100.0),
        )


def test_timetool_binning_by_width_keeps_every_shot():
    """A width used to generate its edges with an arange that stopped one
    width short of the maximum, dropping the latest-arriving shots."""
    from escape.storage.test_data import get_test_data

    data = get_test_data()
    signal, corrected = data["i"], data["t"]
    binned = signal.tools.timetool_binning(corrected, time_vec=None, time_bins=0.2)
    assert len(binned) == len(signal)
    assert len(binned.scan.step_lengths) > 1
