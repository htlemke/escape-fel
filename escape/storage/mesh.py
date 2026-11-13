"""Meshes: where in an N-d space the steps of a scan measured.

A **mesh** is a set of points (its *nodes*) in a space of one or more named
dimensions. A **grid** is a special, structured mesh: every node sits on a
product of one position vector per dimension, so values can be laid out as an
N-d array. A plain 1-d scan is a 1-d mesh (and so always a grid); a mesh scan
is a grid; an adaptive scan is a refined subset of a grid; points anywhere are
a mesh that is not a grid.

:class:`Mesh` is the geometry alone. :class:`ArrayMesh` (``arr.mesh``) binds it
to the steps of an :class:`~escape.Array`: which node every recorded step
measured. Several steps can measure the same node (repetitions, an adaptive
scan going back); per node, their events are pooled. The recorded order stays
available (``step_nodes``, ``step_coords``, ``merge=False``), so the mesh does
not depend on the scan for anything.

The step methods (``mean``, ``std``, ``median``, ... -- the ones
:class:`~escape.storage.storage.Scan` has) exist on ``arr.mesh`` too and return
a :class:`MeshArray`: a plain numpy array that also carries its coordinates.
For a grid it is N-d (NaN where nothing was measured), otherwise 1-d, one value
per node. Keywords on every one of them:

``bins=``
    group the *events* into bins first (see :meth:`ArrayMesh.as_grid`), then
    apply the method per bin: a binned grid.
``interpolate=``
    apply the method per node first, then interpolate the values onto a grid
    (linear, through the nodes' triangulation; masked far from any node).
``merge=False``
    one value per recorded step, in recorded order, instead of per node.
``plot=``
    plot the result (True, an Axes or a Figure), as the Scan methods do.

Vectors given for ``bins`` or ``interpolate`` are kept as they are (linked,
not copied) in the result's ``edges`` / ``coords``.
"""

import logging
from numbers import Integral

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Mesh", "ArrayMesh", "MeshArray"]

_GRID_KINDS = ("grid", "binned", "interpolated")

# How many nodes' lazy results one dask run evaluates. Evaluating them together
# is what makes a reduction over a lazy mesh fast -- one scheduler pass instead
# of one per node, reading each file chunk once for all the nodes that share
# it. Batching rather than one graph for everything is not only a bound on
# memory: graph building and scheduling grow faster than linearly, so a few
# medium graphs beat one big one. Measured on a 4483-node grid: 46 s per-node,
# 19 s in one graph, 11 s in batches of 512 (the optimum is shallow, 256-1024).
MESH_COMPUTE_BATCH_NODES = 512


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
class MeshArray(np.ndarray):
    """A numpy array of results on a mesh, with its coordinates.

    Attributes (all references, nothing is copied):

    coords
        one entry per axis of the array: for a grid-like ``kind`` (``"grid"``,
        ``"binned"``, ``"interpolated"``) the positions of that dimension; for
        ``"nodes"`` and ``"steps"`` the ``(n, ndim)`` node coordinates along
        axis 0. ``None`` for axes of the values themselves (e.g. a spectrum per
        node).
    kind
        ``"grid"``, ``"binned"``, ``"interpolated"``, ``"nodes"`` or ``"steps"``.
    dimension_names, units
        of the mesh's dimensions.
    edges
        bin edges per dimension (``"binned"`` only).
    mask
        True where an interpolated value is too far from any node
        (``"interpolated"`` only; those values are NaN).
    mesh
        the :class:`Mesh` the values belong to.

    Slicing with integers and slices keeps the coordinates of what is left.
    Any other operation that changes the shape (reductions, fancy or boolean
    indexing) returns a plain ndarray: its coordinates would be a guess.
    Element-wise arithmetic keeps them.
    """

    _ATTRS = ("coords", "kind", "dimension_names", "units", "edges", "mask", "mesh")

    def __new__(cls, values, coords=None, kind=None, dimension_names=None, units=None,
                edges=None, mask=None, mesh=None):
        obj = np.asarray(values).view(cls)
        obj.coords = coords
        obj.kind = kind
        obj.dimension_names = dimension_names
        obj.units = units
        obj.edges = edges
        obj.mask = mask
        obj.mesh = mesh
        return obj

    def __array_finalize__(self, obj):
        same = isinstance(obj, MeshArray) and obj.shape == self.shape
        for attr in self._ATTRS:
            setattr(self, attr, getattr(obj, attr, None) if same else None)

    def __array_wrap__(self, out, context=None, return_scalar=False):
        if out.shape != self.shape:
            plain = np.asarray(out).view(np.ndarray)
            return plain[()] if return_scalar else plain
        return super().__array_wrap__(out, context, return_scalar)

    def __getitem__(self, key):
        out = super().__getitem__(key)
        if not isinstance(out, MeshArray):
            return out
        keys = key if isinstance(key, tuple) else (key,)
        if self.coords is None or not all(isinstance(k, (Integral, slice)) for k in keys):
            return out.view(np.ndarray)
        grid_like = self.kind in _GRID_KINDS
        coords, names, units, edges = [], [], [], []
        for axis, c in enumerate(self.coords):
            k = keys[axis] if axis < len(keys) else slice(None)
            on_dimension = grid_like and self.dimension_names and axis < len(self.dimension_names)
            if isinstance(k, Integral):
                continue  # this axis is gone
            coords.append(None if c is None else c[k])
            if on_dimension:
                names.append(self.dimension_names[axis])
                units.append(self.units[axis] if self.units else "")
                if self.edges is not None:
                    e = self.edges[axis]
                    start, stop, step = k.indices(len(c))
                    edges.append(np.asarray(e)[start:stop + 1] if step == 1 else None)
        out.coords = coords
        if grid_like:
            out.dimension_names, out.units = names, units
            out.edges = edges if self.edges is not None else None
        else:
            out.dimension_names, out.units = self.dimension_names, self.units
        out.mask = None if self.mask is None else self.mask[key]
        out.kind, out.mesh = self.kind, self.mesh
        return out

    def plot(self, axis=None, **kwargs):
        """Plot against the coordinates: a curve in 1-d, a colour map for a
        2-d grid, a colour map over the triangulation for 2-d nodes."""
        return plot_mesh_array(self, axis=axis, **kwargs)


def _label(name, unit):
    return f"{name} / {unit}" if unit else str(name)


def plot_mesh_array(res, axis=None, fmt=None, colorbar=True, **kwargs):
    import matplotlib.pyplot as plt

    if axis is None or axis is True:
        axis = plt.gca()
    elif hasattr(axis, "add_subplot") and not hasattr(axis, "plot"):
        axis = axis.add_subplot(111)
    values = np.asarray(res)
    names = list(res.dimension_names or [])
    units = list(res.units or [""] * len(names))
    if res.kind in _GRID_KINDS:
        ndim = len(names)
        if ndim == 1 and values.ndim == 1:
            axis.plot(res.coords[0], values, fmt or ("-" if res.kind == "interpolated" else "o-"), **kwargs)
            axis.set_xlabel(_label(names[0], units[0]))
            return axis
        if ndim == 2 and values.ndim == 2:
            from ..utilities import plot2D

            p = plot2D(np.asarray(res.coords[1]), np.asarray(res.coords[0]), values, axis=axis, **kwargs)
            axis.set_xlabel(_label(names[1], units[1]))
            axis.set_ylabel(_label(names[0], units[0]))
            if colorbar:
                plt.colorbar(p, ax=axis)
            return axis
    elif res.kind in ("nodes", "steps") and values.ndim == 1:
        nodes = np.asarray(res.coords[0])
        if nodes.shape[1] == 1:
            order = np.argsort(nodes[:, 0]) if res.kind == "nodes" else slice(None)
            axis.plot(nodes[order, 0], values[order], fmt or "o-", **kwargs)
            axis.set_xlabel(_label(names[0], units[0]))
            return axis
        if nodes.shape[1] == 2:
            x, y = nodes[:, 0], nodes[:, 1]
            p = None
            try:
                if len(x) >= 3 and res.kind == "nodes":
                    p = axis.tripcolor(x, y, values, shading="gouraud", **kwargs)
            except Exception:  # collinear or too few distinct points
                p = None
            s = axis.scatter(x, y, c=values if p is None else "k", s=12 if p is None else 3)
            axis.set_xlabel(_label(names[0], units[0]))
            axis.set_ylabel(_label(names[1], units[1]))
            if colorbar:
                plt.colorbar(p if p is not None else s, ax=axis)
            return axis
    raise ValueError(
        f"no plot for {values.ndim}-d values of kind {res.kind!r} on "
        f"{len(names)} dimension(s)"
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
class Mesh:
    """The nodes of an N-d space (one or more dimensions), optionally on a grid.

    Parameters
    ----------
    nodes : array-like, (n, ndim) or (n,) for one dimension
    dimension_names, units : list of str, optional
    kind : str, optional
        ``"grid"``, ``"adapted"`` (a refined subset of a grid), ``"line"``,
        ``"path"``, ``"scattered"``, ``"binned"`` -- a description; whether the
        mesh *is* a grid is :attr:`is_grid`.
    resolution : list of float, optional
        Spacing below which two points count as one node.
    positions : list of arrays, optional
        The grid's positions per dimension; with `node_grid_index` (the
        position index of every node per dimension) this makes it a grid. A
        1-d mesh is always a grid (its sorted node coordinates).
    """

    def __init__(self, nodes, dimension_names=None, units=None, kind=None,
                 resolution=None, positions=None, node_grid_index=None, edges=None):
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes[:, None]
        self.nodes = nodes
        ndim = nodes.shape[1]
        self.dimension_names = list(dimension_names) if dimension_names else [
            f"dim_{d}" for d in range(ndim)
        ]
        self.units = list(units) if units else [""] * ndim
        self.resolution = None if resolution is None else [float(r) for r in resolution]
        self.edges = edges
        if positions is None and ndim == 1 and len(nodes):
            # a line is always a grid: its sorted coordinates
            positions = [np.unique(nodes[:, 0])]
            node_grid_index = np.searchsorted(positions[0], nodes[:, 0])[:, None]
            kind = kind or "line"
        self.positions = positions
        self.node_grid_index = None if node_grid_index is None else np.asarray(node_grid_index, int)
        if kind is None:
            kind = "grid" if positions is not None else ("line" if ndim == 1 else "scattered")
        self.kind = kind

    @classmethod
    def from_points(cls, points, **kwargs):
        """A mesh of the given points (merged where equal)."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        nodes = np.unique(points, axis=0)
        return cls(nodes, **kwargs)

    @classmethod
    def from_grid(cls, positions, dimension_names=None, units=None, kind="grid"):
        """The full grid of the given positions per dimension."""
        positions = list(positions)
        index = np.array(list(np.ndindex(*[len(p) for p in positions])), dtype=int)
        nodes = np.column_stack(
            [np.asarray(p, float)[index[:, d]] for d, p in enumerate(positions)]
        )
        return cls(nodes, dimension_names, units, kind, positions=positions,
                   node_grid_index=index)

    @property
    def ndim(self):
        return self.nodes.shape[1]

    @property
    def is_grid(self):
        return self.positions is not None and self.node_grid_index is not None

    @property
    def shape(self):
        """The grid's shape, None if the mesh is not a grid."""
        return tuple(len(p) for p in self.positions) if self.is_grid else None

    def __len__(self):
        return len(self.nodes)

    def triangulation(self):
        """The mesh's cells: a matplotlib ``Triangulation`` in 2-d, a scipy
        ``Delaunay`` in N-d (computed on first use)."""
        if getattr(self, "_cells", None) is None:
            if self.ndim == 2:
                from matplotlib.tri import Triangulation

                self._cells = Triangulation(self.nodes[:, 0], self.nodes[:, 1])
            else:
                from scipy.spatial import Delaunay

                self._cells = Delaunay(self.nodes)
        return self._cells

    def __repr__(self):
        dims = ", ".join(
            _label(n, u) for n, u in zip(self.dimension_names, self.units)
        )
        grid = f", grid {'x'.join(map(str, self.shape))}" if self.is_grid else ""
        return f"<Mesh {self.kind}: {len(self)} nodes in [{dims}]{grid}>"


# ---------------------------------------------------------------------------
# a mesh bound to the steps of an Array
# ---------------------------------------------------------------------------
def _is_lazy(value):
    """Whether *value* -- or any part of a tuple of them -- still needs dask."""
    if isinstance(value, tuple):
        return any(hasattr(v, "compute") for v in value)
    return hasattr(value, "compute")


def _as_values(value):
    """One group's result as numpy: an array, or a tuple of them."""
    if isinstance(value, tuple):
        return tuple(np.asarray(v) for v in value)
    return np.asarray(value)


def _step_infos(scan):
    infos = (scan.parameter.get("scan_step_info") or {}).get("values") or []
    return infos if len(infos) == len(scan) else [{}] * len(scan)


def _from_parameters(scan):
    """(names, per-step coordinates, units) from the scan's own parameter
    values: the targets (not the ``*_readback`` copies), numeric, one value
    per step.

    A :func:`~escape.storage.storage.digitize` result describes each of its
    steps with three parameters -- ``bin_center_x``, ``bin_left_x`` and
    ``bin_right_x`` -- which are one axis, not three: only the centers become
    a mesh dimension, named and given a unit by what was binned.
    """
    names, columns, units = [], [], []
    parameters = dict(scan.parameter)
    binned = {
        name[len("bin_center_"):]
        for name in parameters
        if name.startswith("bin_center_")
    }
    for name, par in parameters.items():
        if name == "scan_step_info" or name.endswith("_readback") or not isinstance(par, dict):
            continue
        if any(
            name == prefix + quantity
            for quantity in binned
            for prefix in ("bin_left_", "bin_right_")
        ):
            continue
        try:
            values = np.asarray(par["values"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if values.shape == (len(scan),) and not np.all(np.isnan(values)):
            attributes = par.get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            names.append(attributes.get("axis_name", name))
            units.append(attributes.get("unit") or "")
            columns.append(values)
    if not columns:
        return ["step"], np.arange(len(scan), dtype=float)[:, None], [""]
    return names, np.column_stack(columns), units


def _merge_points(points, resolution=None):
    """(unique nodes, node index per point): equal points -- or, with a
    resolution, points rounding to the same multiple of it -- are one node."""
    if resolution:
        res = np.where(np.asarray(resolution, float) > 0, resolution, 1.0)
        keys = np.round(points / res).astype(np.int64)
    else:
        keys = points
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return points[first], np.asarray(inverse).ravel()


class ArrayMesh:
    """``arr.mesh``: the :class:`Mesh` of an Array's scan and which node each of
    its steps measured. See the module docstring for what it offers."""

    def __init__(self, array, mesh, step_nodes, steps=None):
        self._array = array
        self.mesh = mesh
        self.step_nodes = np.asarray(step_nodes, dtype=int)
        self.steps = np.arange(len(self.step_nodes)) if steps is None else np.asarray(steps, int)

    # -- building -------------------------------------------------------------
    @classmethod
    def from_array(cls, array):
        """From eco's ``mesh_specs`` + per-step ``mesh_index``, else from
        ``grid_specs`` + ``grid_index``, else from the scan's parameter values."""
        scan = array.scan
        infos = _step_infos(scan)
        specs = getattr(array, "_mesh_specs", None)
        grid_specs = getattr(array, "_grid_specs", None)
        mesh_index = [i.get("mesh_index") if isinstance(i, dict) else None for i in infos]
        grid_index = [i.get("grid_index") if isinstance(i, dict) else None for i in infos]
        has_grid = bool(grid_specs) and all(g is not None for g in grid_index)

        if specs and specs.get("points") and all(m is not None for m in mesh_index):
            nodes = np.asarray(specs["points"], dtype=float)
            step_nodes = np.asarray(mesh_index, int)
            positions = node_grid_index = None
            if has_grid and len(grid_specs["shape"]) == nodes.shape[1]:
                positions = [np.asarray(p) for p in grid_specs["positions"]]
                node_grid_index = np.zeros(nodes.shape, int)
                for node, gi in zip(step_nodes, grid_index):
                    node_grid_index[node] = gi
            mesh = Mesh(nodes, specs.get("dimension_names"), specs.get("units"),
                        specs.get("kind"), specs.get("resolution"), positions, node_grid_index)
        elif has_grid:
            keys = np.asarray(grid_index, int)
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            positions = [np.asarray(p) for p in grid_specs["positions"]]
            nodes = np.column_stack(
                [np.asarray(p, float)[unique[:, d]] for d, p in enumerate(positions)]
            )
            mesh = Mesh(nodes, grid_specs.get("grid_dimension_names"),
                        grid_specs.get("grid_units"), "grid",
                        positions=positions, node_grid_index=unique)
            step_nodes = np.asarray(inverse).ravel()
        else:
            names, points, units = _from_parameters(scan)
            nodes, step_nodes = _merge_points(points)
            # several parameters without a grid: moved together, a path
            mesh = Mesh(nodes, names, units if any(units) else None,
                        kind=None if len(names) == 1 else "path")
        return cls(array, mesh, step_nodes)._compact()

    def _compact(self):
        """Only the nodes some step measured, renumbered."""
        used = np.unique(self.step_nodes)
        if len(used) == len(self.mesh) and np.array_equal(used, np.arange(len(used))):
            return self
        remap = np.full(len(self.mesh), -1, int)
        remap[used] = np.arange(len(used))
        m = self.mesh
        mesh = Mesh(m.nodes[used], m.dimension_names, m.units, m.kind, m.resolution,
                    m.positions, None if m.node_grid_index is None else m.node_grid_index[used],
                    m.edges)
        return ArrayMesh(self._array, mesh, remap[self.step_nodes], self.steps)

    # -- what it is -----------------------------------------------------------
    @property
    def nodes(self):
        """(n, ndim) node coordinates."""
        return self.mesh.nodes

    @property
    def dimension_names(self):
        return self.mesh.dimension_names

    @property
    def units(self):
        return self.mesh.units

    @property
    def positions(self):
        """The grid positions per dimension, or ``None`` when the nodes do not
        lie on a grid."""
        return self.mesh.positions

    @property
    def kind(self):
        return self.mesh.kind

    @property
    def is_grid(self):
        return self.mesh.is_grid

    @property
    def ndim(self):
        return self.mesh.ndim

    @property
    def shape(self):
        return self.mesh.shape

    @property
    def step_coords(self):
        """(n_steps, ndim) coordinates of every recorded step, in recorded order."""
        return self.mesh.nodes[self.step_nodes]

    @property
    def node_steps(self):
        """Per node, the (Array) step numbers that measured it, in recorded order."""
        # A stable sort by node keeps each node's steps in recorded order, and
        # the counts say where to cut -- a Python loop over the steps costs
        # noticeably more once a mesh has thousands of nodes.
        order = np.argsort(self.step_nodes, kind="stable")
        counts = np.bincount(self.step_nodes, minlength=len(self.mesh))
        cuts = np.cumsum(counts[:-1]) if len(counts) else []
        return [part.tolist() for part in np.split(self.steps[order], cuts)]

    def visits(self):
        """How many steps measured each node."""
        return np.bincount(self.step_nodes, minlength=len(self.mesh))

    @property
    def array(self):
        """The Array of the selected steps."""
        if len(self.steps) == len(self._array.scan) and np.array_equal(
            self.steps, np.arange(len(self.steps))
        ):
            return self._array
        return self._array_of_steps(self.steps)

    def __len__(self):
        return len(self.mesh)

    def __repr__(self):
        visits = len(self.steps)
        return f"<ArrayMesh {self.mesh!r}, {visits} steps>"

    # -- selecting --------------------------------------------------------------
    def __getitem__(self, sel):
        if not self.is_grid:
            raise TypeError(
                f"this mesh ({self.kind}) is not a grid, so it has no grid index: use "
                ".as_grid(bins=...)[...] or .select({name: (low, high)})"
            )
        ndim = self.mesh.ndim
        sel = sel if isinstance(sel, tuple) else (sel,)
        if Ellipsis in sel:
            i = sel.index(Ellipsis)
            sel = sel[:i] + (slice(None),) * (ndim - len(sel) + 1) + sel[i + 1:]
        if len(sel) > ndim:
            raise IndexError(f"too many indices for a {ndim}-d mesh")
        sel = sel + (slice(None),) * (ndim - len(sel))
        keep_dims, chosen = [], []
        for d, (s, n) in enumerate(zip(sel, self.shape)):
            if isinstance(s, Integral):
                chosen.append([s % n])
            else:
                keep_dims.append(d)
                chosen.append(
                    list(range(*s.indices(n))) if isinstance(s, slice)
                    else [int(i) % n for i in s]
                )
        gi = self.mesh.node_grid_index
        keep = np.all([np.isin(gi[:, d], chosen[d]) for d in range(ndim)], axis=0)
        if not keep.any():
            raise IndexError("the selection contains no measured node")
        if not keep_dims:  # one grid point: its steps, as an Array
            return self._array_of_steps(self.steps[keep[self.step_nodes]])
        renumber = [
            {old: new for new, old in enumerate(chosen[d])} for d in range(ndim)
        ]
        node_grid_index = np.array(
            [[renumber[d][g[d]] for d in keep_dims] for g in gi[keep]], dtype=int
        )
        m = self.mesh
        mesh = Mesh(
            m.nodes[keep][:, keep_dims],
            [m.dimension_names[d] for d in keep_dims],
            [m.units[d] for d in keep_dims],
            m.kind,
            None if m.resolution is None else [m.resolution[d] for d in keep_dims],
            [np.asarray(m.positions[d])[chosen[d]] for d in keep_dims],
            node_grid_index,
        )
        return self._with_nodes(keep, mesh)

    def select(self, ranges=None, **kwargs):
        """The nodes within value ranges, for any mesh:
        ``select({"xrd.x": (0, 1)})`` or ``select(xrd_x=(0, 1))`` (dots as
        underscores). A grid stays a grid (the ranges are per dimension)."""
        ranges = dict(ranges or {})
        names = self.mesh.dimension_names
        for key, value in kwargs.items():
            match = [n for n in names if n == key or n.replace(".", "_") == key]
            if not match:
                raise KeyError(f"{key}: not one of {names}")
            ranges[match[0]] = value
        keep = np.ones(len(self.mesh), bool)
        for name, (low, high) in ranges.items():
            x = self.mesh.nodes[:, names.index(name)]
            keep &= (x >= low) & (x <= high)
        m = self.mesh
        positions, node_grid_index = None, None
        if m.is_grid:
            positions, node_grid_index, index_maps = [], [], []
            for d, p in enumerate(m.positions):
                p = np.asarray(p)
                low, high = ranges.get(names[d], (-np.inf, np.inf))
                inside = np.flatnonzero((p >= low) & (p <= high))
                positions.append(p[inside])
                index_maps.append({old: new for new, old in enumerate(inside)})
            node_grid_index = np.array(
                [[index_maps[d][g[d]] for d in range(m.ndim)] for g in m.node_grid_index[keep]],
                dtype=int,
            ).reshape(-1, m.ndim)
        mesh = Mesh(m.nodes[keep], m.dimension_names, m.units, m.kind, m.resolution,
                    positions, node_grid_index)
        return self._with_nodes(keep, mesh)

    def _with_nodes(self, keep, mesh):
        remap = np.full(len(self.mesh), -1, int)
        remap[np.flatnonzero(keep)] = np.arange(keep.sum())
        on = keep[self.step_nodes]
        return ArrayMesh(self._array, mesh, remap[self.step_nodes[on]], self.steps[on])

    # -- binning ------------------------------------------------------------------
    def _dimension_edges(self, d, spec):
        x = self.mesh.nodes[:, d]
        lo, hi = float(np.min(x)), float(np.max(x))

        def around(centers):
            centers = np.asarray(centers, float)
            if len(centers) == 1:
                half = (self.mesh.resolution or [0.5] * self.mesh.ndim)[d] / 2 or 0.5
                return np.array([centers[0] - half, centers[0] + half])
            mid = (centers[1:] + centers[:-1]) / 2
            return np.concatenate([[2 * centers[0] - mid[0]], mid, [2 * centers[-1] - mid[-1]]])

        if isinstance(spec, str):
            if spec == "nodes":
                return around(np.unique(x))
            if spec != "auto":
                raise ValueError(f"bins: {spec!r} (expected 'auto', 'nodes', a number or edges)")
            res = (self.mesh.resolution or [0] * self.mesh.ndim)[d]
            if res and res > 0:
                n0, n1 = int(np.round(lo / res)), int(np.round(hi / res))
                return around(np.arange(n0, n1 + 1) * res)
            unique = np.unique(x)
            if len(unique) <= 50:
                return around(unique)
            spec = int(max(2, np.sqrt(len(self.mesh))))
        if isinstance(spec, Integral):
            if hi == lo:
                lo, hi = lo - 0.5, hi + 0.5
            return np.linspace(lo, hi, int(spec) + 1)
        return spec  # given edges: kept as they are (a link)

    def _edges(self, bins):
        ndim = self.mesh.ndim
        if isinstance(bins, dict):
            return [self._dimension_edges(d, bins.get(n, "auto"))
                    for d, n in enumerate(self.mesh.dimension_names)]
        if isinstance(bins, (str, Integral)):
            return [self._dimension_edges(d, bins) for d in range(ndim)]
        bins = list(bins)
        if ndim == 1 and all(np.isscalar(b) and not isinstance(b, str) for b in bins):
            return [bins]  # the edges of the one dimension
        if len(bins) != ndim:
            raise ValueError(f"bins: one entry per dimension ({ndim}) expected")
        return [self._dimension_edges(d, b) for d, b in enumerate(bins)]

    def as_grid(self, bins="auto"):
        """The same steps on a grid of bins: every step's events go into the bin
        of its node, so a method then pools all events of a bin.

        `bins`: ``"auto"`` (the mesh's resolution if known -- for an adaptive
        scan its original axis grid --, else the distinct node coordinates if
        few, else a number of bins), ``"nodes"`` (the distinct coordinates of
        each dimension), a number of bins, edges, or one of these per
        dimension (list, or dict by dimension name). Given edges are kept as
        they are (linked) in ``mesh.edges``.
        """
        edges = self._edges(bins)
        index = np.empty(self.mesh.nodes.shape, int)
        inside = np.ones(len(self.mesh), bool)
        for d, e in enumerate(edges):
            e_arr = np.asarray(e, float)
            i = np.searchsorted(e_arr, self.mesh.nodes[:, d], side="right") - 1
            i[self.mesh.nodes[:, d] == e_arr[-1]] = len(e_arr) - 2  # the last edge belongs in
            inside &= (i >= 0) & (i < len(e_arr) - 1)
            index[:, d] = i
        centers = [(np.asarray(e, float)[1:] + np.asarray(e, float)[:-1]) / 2 for e in edges]
        cells, cell_of_node = np.unique(index[inside], axis=0, return_inverse=True)
        node_to_cell = np.full(len(self.mesh), -1, int)
        node_to_cell[np.flatnonzero(inside)] = np.asarray(cell_of_node).ravel()
        nodes = np.column_stack([centers[d][cells[:, d]] for d in range(len(edges))]) if len(cells) \
            else np.empty((0, len(edges)))
        m = self.mesh
        mesh = Mesh(nodes, m.dimension_names, m.units, "binned", m.resolution,
                    positions=centers, node_grid_index=cells, edges=edges)
        on = inside[self.step_nodes]
        return ArrayMesh(self._array, mesh, node_to_cell[self.step_nodes[on]], self.steps[on])

    # -- computing ---------------------------------------------------------------
    def _array_of_steps(self, steps):
        steps = [int(s) for s in steps]
        scan = self._array.scan
        return scan.get_step_array(steps[0]) if len(steps) == 1 else scan[steps]

    def _event_slices(self):
        """``(data, bounds)`` for the fast path, or ``None``.

        The event axis of an Array is partitioned by its step lengths in order,
        so step *n* is ``data[bounds[n]:bounds[n + 1]]``. Only offered for
        numpy-backed data -- a dask array has to go through the lazy path.
        """
        array = self._array
        data = array.data
        if not isinstance(data, np.ndarray):
            return None
        return data, np.asarray(array.scan._step_index_ranges)

    def _direct_reduction(self, name, kwargs):
        """The plain numpy function behind a step method, or ``None``.

        Array's reductions go through :func:`~escape.storage.storage.escaped`,
        which aligns indices and rebuilds an Array for every call. On one slice
        of one numpy array that work is all thrown away again, and with a few
        thousand nodes it costs several times the reduction itself. Without an
        ``axis`` the method is exactly ``np_func(raw_values, ...)``, so for that
        case the function is used directly. Anything else -- an ``axis``, an
        element-wise method, an escape-specific one like ``median_and_mad`` --
        returns ``None`` and takes the ordinary path.
        """
        if "axis" in kwargs:
            return None
        from .storage import _ARRAY_DELEGATE_METHODS

        entry = _ARRAY_DELEGATE_METHODS.get(name)
        if entry is None:
            return None
        np_func, _, is_reduction, _ = entry
        return np_func if (is_reduction and np_func is not None) else None

    def _values_per_node(self, groups, name, args, kwargs):
        """The method applied to every group of steps, as a list over groups.

        Three paths, in order of preference: the plain numpy function on the
        raw values (see :meth:`_direct_reduction`); otherwise the Array method
        per group, with any lazy results evaluated **together** rather than one
        dask run per group -- a few thousand groups re-reading the same file
        chunks is otherwise the whole cost.

        An entry is a tuple when the method gives several values per group
        (``median_and_mad``, ``weighted_stat``, ...).
        """
        values = [None] * len(groups)

        if name == "count":
            for i, steps in enumerate(groups):
                values[i] = int(sum(self._array.scan.step_lengths[s] for s in steps))
            return values

        sliced = self._event_slices()
        np_func = None if sliced is None else self._direct_reduction(name, kwargs)
        if np_func is not None:
            data, bounds = sliced
            for i, steps in enumerate(groups):
                if len(steps) == 1:
                    part = data[bounds[steps[0]]:bounds[steps[0] + 1]]
                else:
                    part = np.concatenate(
                        [data[bounds[s]:bounds[s + 1]] for s in steps]
                    )
                values[i] = np.asarray(np_func(part, *args, **kwargs))
            return values

        lazy_at, lazy = [], []
        for i, steps in enumerate(groups):
            part = self._array_of_steps(steps)
            value = getattr(part, name)(*args, **kwargs)
            if _is_lazy(value):
                lazy_at.append(i)
                lazy.append(value)
            else:
                values[i] = _as_values(value)
        if lazy:
            import dask
            from dask.diagnostics import ProgressBar

            # In batches (see MESH_COMPUTE_BATCH_NODES): one graph for
            # everything is both unbounded and slower than a few medium ones.
            with ProgressBar():
                for start in range(0, len(lazy), MESH_COMPUTE_BATCH_NODES):
                    batch = lazy[start:start + MESH_COMPUTE_BATCH_NODES]
                    for i, value in zip(lazy_at[start:start + len(batch)],
                                        dask.compute(*batch)):
                        values[i] = _as_values(value)
        return values

    def _as_mesh_array(self, values, merge):
        m = self.mesh
        if not merge:
            return MeshArray(values, [self.step_coords], "steps", m.dimension_names, m.units, mesh=m)
        if not m.is_grid:
            return MeshArray(values, [m.nodes], "nodes", m.dimension_names, m.units, mesh=m)
        dtype = np.result_type(values.dtype, float)
        out = np.full(tuple(m.shape) + values.shape[1:], np.nan, dtype=dtype)
        for node, gi in enumerate(m.node_grid_index):
            out[tuple(gi)] = values[node]
        coords = list(m.positions) + [None] * (values.ndim - 1)
        kind = "binned" if m.kind == "binned" else "grid"
        return MeshArray(out, coords, kind, m.dimension_names, m.units,
                         edges=m.edges if kind == "binned" else None, mesh=m)

    def _interpolate(self, values, spec):
        """Node values onto a grid: `spec` True/"auto", a number of points per
        dimension, positions per dimension (kept as given), or a dict with
        ``positions`` and/or ``max_distance`` (in units of the nodes' typical
        spacing, default 2)."""
        m = self.mesh
        max_distance = 2.0
        if isinstance(spec, dict):
            max_distance = spec.get("max_distance", max_distance)
            spec = spec.get("positions", "auto")
        ndim = m.ndim
        lo, hi = m.nodes.min(axis=0), m.nodes.max(axis=0)
        if spec is True or (isinstance(spec, str) and spec == "auto"):
            spec = {1: 200, 2: 100}.get(ndim, 25)
        if isinstance(spec, Integral):
            positions = [np.linspace(lo[d], hi[d], int(spec)) for d in range(ndim)]
        else:
            positions = list(spec) if ndim > 1 else (
                [spec] if np.ndim(spec) == 1 and np.isscalar(np.asarray(spec)[0]) else list(spec)
            )
        values = np.asarray(values, dtype=float)
        tail = values.shape[1:]
        flat = values.reshape(len(values), -1)
        targets = np.stack(
            np.meshgrid(*[np.asarray(p, float) for p in positions], indexing="ij"), axis=-1
        ).reshape(-1, ndim)
        scale = np.where(hi > lo, hi - lo, 1.0)
        if ndim == 1:
            order = np.argsort(m.nodes[:, 0])
            x = m.nodes[order, 0]
            out = np.column_stack(
                [np.interp(targets[:, 0], x, flat[order, j], left=np.nan, right=np.nan)
                 for j in range(flat.shape[1])]
            )
        else:
            from scipy.interpolate import LinearNDInterpolator

            try:
                out = LinearNDInterpolator(m.nodes / scale, flat)(targets / scale)
            except Exception as exc:
                raise ValueError(
                    f"cannot interpolate {len(m)} nodes in {ndim}-d (too few, or all "
                    f"on a line): {exc}"
                ) from None
        mask = None
        if len(m) > 1:
            from scipy.spatial import cKDTree

            tree = cKDTree(m.nodes / scale)
            spacing = np.median(tree.query(m.nodes / scale, k=2)[0][:, 1])
            distance = tree.query(targets / scale)[0]
            far = distance > max_distance * spacing
            out[far] = np.nan
            # also outside the nodes' hull, where linear interpolation has none
            far |= ~np.isfinite(out).all(axis=1)
            mask = far.reshape([len(p) for p in positions])
        out = out.reshape([len(p) for p in positions] + list(tail))
        coords = positions + [None] * len(tail)
        return MeshArray(out, coords, "interpolated", m.dimension_names, m.units, mask=mask, mesh=m)


def _make_mesh_method(name):
    def method(self, *args, merge=True, bins=None, interpolate=None, plot=None,
               plot_kws=None, **kwargs):
        if bins is not None and interpolate is not None:
            raise ValueError("bins= and interpolate= together: bin first, or interpolate the nodes")
        if interpolate is not None and not merge:
            raise ValueError("interpolate= needs one value per node (merge=True)")
        view = self.as_grid(bins) if bins is not None else self
        groups = view.node_groups() if merge else [[s] for s in view.steps]
        per_node = view._values_per_node(groups, name, args, kwargs)

        # Some methods give several values per node (median_and_mad,
        # weighted_avg_and_std, weighted_stat, ...). Each becomes a mesh of its
        # own, returned as a tuple, as the same methods do on arr.scan/arr.grid.
        if per_node and isinstance(per_node[0], tuple):
            columns = [
                np.asarray([node[k] for node in per_node])
                for k in range(len(per_node[0]))
            ]
        else:
            columns = [np.asarray(per_node)]

        results = tuple(
            view._interpolate(values, interpolate) if interpolate is not None
            else view._as_mesh_array(values, merge)
            for values in columns
        )
        result = results[0] if len(results) == 1 else results
        if plot:
            try:
                plot_mesh_array(results[0], axis=plot, **(plot_kws or {}))
            except Exception as exc:  # plotting must not cost the result
                logger.warning("mesh %s: no plot: %s", name, exc)
        return result

    method.__name__ = name
    method.__qualname__ = f"ArrayMesh.{name}"
    method.__doc__ = (
        f"``{name}`` per node of the mesh (the events of all steps on a node "
        "pooled), as a :class:`MeshArray`: N-d for a grid, else one value per "
        "node.\n\n"
        "Keywords besides those of ``Array." + name + "``:\n\n"
        "merge : bool, default True\n"
        "    False: one value per recorded step, in recorded order.\n"
        "bins : see :meth:`ArrayMesh.as_grid`\n"
        "    Bin the events first, then apply per bin.\n"
        "interpolate : True, 'auto', int, positions per dimension, or dict\n"
        "    Apply per node, then interpolate onto a grid (masked far from nodes).\n"
        "plot : bool, Axes or Figure; plot_kws : dict\n"
        "    Plot the result (the first one, for a method giving several).\n"
        "\n"
        "A method that gives several values per group returns one MeshArray "
        "per value, as a tuple.\n"
    )
    return method


def _node_groups(self):
    """Per node, the (Array) step numbers whose events are pooled."""
    return self.node_steps


ArrayMesh.node_groups = _node_groups


def attach_step_methods(names):
    for name in names:
        setattr(ArrayMesh, name, _make_mesh_method(name))
