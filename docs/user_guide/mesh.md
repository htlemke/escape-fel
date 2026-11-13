# Mesh — Where Each Step Measured

Every scan measures at points in a space of one or more named dimensions: a
motor position, a delay, two motors of a map. The **mesh** of an Array is that
set of points (its *nodes*) and, for every recorded step, which node it
measured. `arr.mesh` is always there:

* a plain 1-D scan is a 1-D mesh (a *line*);
* a mesh scan is a **grid**: every node sits on a product of one position
  vector per dimension, so results can be laid out as an N-D array;
* an adaptive scan is a refined subset of a grid (`kind="adapted"`);
* points anywhere are a mesh that is not a grid.

Several steps can measure the same node (repetitions, an adaptive scan coming
back); per node, their events are **pooled**. The recorded order stays
available, so nothing depends on the scan's step list.

## One value per node

The step methods of `arr.scan` (`mean`, `std`, `median`, `sum`, `count`,
`median_and_mad`, `weighted_stat`, ...) exist on `arr.mesh` too -- all of them,
so `arr.mesh` really is the general form of `arr.grid` and not a subset:

```python
import numpy as np
import escape

x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]          # an ascan, repeated twice
data = np.concatenate([np.full(100, 10 * v) + np.random.randn(100) for v in x])
arr = escape.Array(data=data, index=np.arange(len(data)),
                   step_lengths=[100] * 6, parameter={"x": {"values": x}})

arr.scan.mean()        # 6 values, one per step
r = arr.mesh.mean()    # 3 values, one per position: both repetitions pooled
r.coords[0]            # array([0., 1., 2.])
arr.mesh.mean(merge=False)   # one per recorded step, in recorded order
arr.mesh.step_nodes          # node of every recorded step: [0, 1, 2, 0, 1, 2]
arr.mesh.visits()            # steps per node: [2, 2, 2]
```

A method that gives several values per group -- `median_and_mad`,
`weighted_avg_and_std`, `weighted_stat`, `correlation_analysis_to` -- gives one
mesh per value:

```python
med, mad = sig.mesh.median_and_mad()    # both (n_delay, n_motor), both with coords
```

The result is a **plain numpy array** that also carries its coordinates
({class}`~escape.storage.mesh.MeshArray`): `.coords` (positions per axis for a
grid, the `(n, ndim)` node coordinates otherwise), `.kind`,
`.dimension_names`, `.units`. For a grid it is N-D, NaN where nothing was
measured; otherwise 1-D, one value per node. Slicing with integers and slices
keeps the coordinates (`r[:, 3].coords`); reductions, fancy indexing and the
like give an ordinary array.

## Grids: index like an array

```python
from escape.storage.example_data import make_grid_scan

sig = make_grid_scan(shape=(5, 8), dim_names=("delay_ps", "motor_mm"),
                     dim_ranges=((-0.5, 2.0), (0.0, 4.0)), n_events_per_step=50)
sig.mesh.shape              # (5, 8)
line = sig.mesh[:, 3]       # the steps of column 3, as a 1-D mesh
line.mean(plot=True)        # one value per delay, plotted against it
sig.mesh.mean(plot=True)    # the 5x8 map
```

`arr.mesh[...]` needs a grid. On a mesh that is not one it says what to use
instead: `.as_grid(...)`, or `.select(...)` by value, which works on any mesh:

```python
arr.mesh.select({"xrd.x": (0, 1)})   # or select(xrd_x=(0, 1))
```

## Bins: before the method

`bins=` groups the *events* into bins first, then applies the method per bin,
so a bin's value comes from all events of all steps that fall into it:

```python
m = arr2d.mesh.mean(bins=[edges_x, edges_y])   # given edges: linked, not copied
m.edges[0] is edges_x                          # True
arr2d.mesh.mean(bins="auto")
arr2d.mesh.as_grid(bins=20)[:, 3].mean()      # then index it like a grid
```

`bins`: `"auto"` (the mesh's resolution if known -- for an adaptive scan its
original axis grid --, else the distinct coordinates if there are few, else a
number of bins), `"nodes"` (the distinct coordinates per dimension), a number
of bins, edges, or one of these per dimension (a list, or a dict by dimension
name).

This bins the mesh's *own* dimensions. To make a new dimension out of a
per-shot channel -- a timetool's arrival time, a pulse energy -- bin that
channel into scan steps with `digitize` and cross it with the scan:

```python
from escape.storage.storage import unravel_scans

tt_bins = tt.digitize(bin_width=50, range=(-400, 400), unit="fs")
grid = unravel_scans(mu, tt_bins, categorize_target=mu)
grid.mesh.dimension_names    # ['...mono...energy', 'bernina.tt_kb.edge_position_fs']
```

See [Making an axis out of a per-shot channel](grid.md#binning-a-per-shot-channel).

## Interpolation: after the method

`interpolate=` applies the method per node first, then interpolates the values
linearly onto a grid through the nodes' triangulation. Points far from any
node (more than twice their typical spacing) or outside the measured area are
NaN and marked in `.mask`:

```python
r = arr2d.mesh.mean(interpolate=100)          # 100 points per dimension
r = arr2d.mesh.mean(interpolate=[xs, ys])     # given positions: linked
r = arr2d.mesh.mean(interpolate={"positions": 50, "max_distance": 3})
```

`bins=` and `interpolate=` together are refused: the order would be ambiguous.

## Plots

`plot=True` (or an Axes or Figure) works on every method, as on `arr.scan`:
a curve in 1-D, a colour map for a 2-D grid, binned or interpolated result,
and a colour map over the triangulation for 2-D nodes. Afterwards,
`result.plot()` does the same.

## The mesh without data

{class}`~escape.storage.mesh.Mesh` is the geometry alone:

```python
from escape.storage.mesh import Mesh

g = Mesh.from_grid([[0, 1, 2], [5, 6]], dimension_names=["y", "x"])
p = Mesh.from_points(points)   # points anywhere
p.triangulation()              # its cells (matplotlib Triangulation in 2-D)
```

## Where it comes from

eco writes `mesh_specs` (dimension names, units, kind, resolution, nodes) with
every scan, and a `mesh_index` per step; both are read when a run is loaded
and stored in escape result files. Older files get their mesh from
`grid_specs` and the steps' `grid_index`, or else from the scan's own
parameter values (the targets, not the `*_readback` copies).

`arr.grid` stays as it was; `arr.mesh` is the general form of it.
