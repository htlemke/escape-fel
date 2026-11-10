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
