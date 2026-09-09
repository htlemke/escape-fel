"""Reconstruct and visualize the computation graph behind one or more Streams.

A derived ``Stream`` (arithmetic, ``filter``/``[mask]``, ``element``/``[i]``,
``digitize().categorize()``, ``running_mean`` and friends, ...) is already a
live graph of Python objects -- ``ProcObj.args``/``kwargs`` hold direct
references to the Streams (or plain constants) it was built from. This module
just walks that existing structure into an explicit, inspectable
``networkx.DiGraph`` instead of leaving it implicit in ``_source``/``ProcObj``
chains -- no change to how Streams are built or used (see the docstrings on
``element()``, ``filter()``, ``categorize()``/``categorizeBy()`` and
``_RunningStat`` for the handful of small additions that made a few
previously-invisible dependencies -- e.g. ``element()``'s index, or the fact
that a ``categorize()`` result shares its data with its donor Stream --
walkable at all).

Requires the optional ``networkx`` package (not a core ``escape`` dependency;
imported lazily so merely importing ``escape.stream`` does not require it).

Example
-------
    ratio = i / i0
    smoothed = ratio.running_mean(N_acc=200)
    g = build_graph(smoothed)
    draw(g)
"""

import itertools

import numpy as np

from .escape_stream import EventSource, FilteredEventSource, ProcSource, Stream

# Node "kind" -> a display color for draw(); anything else falls back to gray.
_KIND_COLORS = {
    "channel": "#8ecae6",
    "constant": "#adb5bd",
    "op": "#ffb703",
    "filter": "#fb8500",
    "categorize": "#219ebc",
    "categorize_scan": "#219ebc",
    "categorizeBy": "#219ebc",
    "unknown": "#d90429",
}


def _require_networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "escape.stream.graph requires the optional 'networkx' package "
            "(pip install networkx)."
        ) from exc
    return nx


def _const_label(value):
    if isinstance(value, np.ndarray):
        return f"array{value.shape}"
    r = repr(value)
    return r if len(r) <= 40 else r[:37] + "..."


def _const_node(g, value, counter):
    key = f"const_{next(counter)}"
    g.add_node(key, kind="constant", label=_const_label(value), obj=value)
    return key


def _op_label(proc):
    """Best-effort human label for a ProcObj node.

    Prefers an explicit ``_graph_label()`` hook on ``proc.func`` (used by
    ``_RunningStat`` to report its *current* ``N_acc``, since that lives on
    the result Stream rather than as a ProcObj arg -- see its docstring),
    then falls back to the label already carried in ``returns_names`` (set
    for every arithmetic op and for ``element()``), then the function's own
    name, then a bare repr.
    """
    hook = getattr(proc.func, "_graph_label", None)
    if callable(hook):
        try:
            return hook()
        except Exception:
            pass
    if proc.returns_names:
        return proc.returns_names[0]
    return getattr(proc.func, "__name__", repr(proc.func))


def _parent_label(kind, deps):
    if kind == "categorize":
        n_bins = len(deps["bins"]) - 1
        return f"categorize({deps['data'].name} by {deps['key'].name}, {n_bins} bins)"
    if kind == "categorize_scan":
        return f"categorize_scan({deps['data'].name} by {deps['scan_from'].name})"
    if kind == "categorizeBy":
        return f"categorizeBy({deps['data'].name} by {deps['key'].name})"
    return kind


def _walk_source_node(stream, g, seen, counter):
    """Build the node for a Stream from its `_source` (the default path,
    used whenever there is no `_graph_parent` override -- see `_walk`)."""
    src = stream._source
    key = id(stream)

    if isinstance(src, EventSource):
        g.add_node(key, kind="channel", label=src.name, obj=stream, unit=stream.unit)

    elif isinstance(src, FilteredEventSource):
        g.add_node(key, kind="filter", label=src.name, obj=stream, unit=stream.unit)
        inner_stream = src._inner_stream
        if inner_stream is not None:
            dep_key = _walk(inner_stream, g, seen, counter)
        else:
            # A FilteredEventSource built by hand from a bare Source (not via
            # Stream.filter()) has no Stream to walk -- fall back to a leaf
            # node so the graph stays correct, just less detailed upstream.
            dep_key = f"unknown_{id(src._inner)}"
            g.add_node(dep_key, kind="unknown", label=getattr(src._inner, "name", repr(src._inner)))
        g.add_edge(dep_key, key, role="data")
        mask_key = _walk(src._mask, g, seen, counter)
        g.add_edge(mask_key, key, role="mask")

    elif isinstance(src, ProcSource):
        proc = src.procObj
        g.add_node(key, kind="op", label=_op_label(proc), obj=stream, unit=stream.unit)
        for arg, is_esc in zip(proc.args, proc.args_is_esc):
            dep_key = _walk(arg, g, seen, counter) if is_esc else _const_node(g, arg, counter)
            g.add_edge(dep_key, key, role="arg")
        for name, val in proc.kwargs.items():
            is_esc = proc.kwargs_is_esc[name]
            dep_key = _walk(val, g, seen, counter) if is_esc else _const_node(g, val, counter)
            g.add_edge(dep_key, key, role=name)

    else:
        g.add_node(key, kind="unknown", label=repr(src), obj=stream)

    return key


def _walk(stream, g, seen, counter):
    key = id(stream)
    if key in seen:
        return key
    seen.add(key)

    parent = getattr(stream, "_graph_parent", None)
    if parent is not None:
        # categorize()/categorizeBy() share `_source` with their data donor --
        # walking `_source` here would misrepresent the node as an
        # independent recomputation of the donor's whole upstream chain
        # instead of "donor's data, regrouped". _graph_parent is authoritative
        # in this case; see the note in Stream.categorize()/categorizeBy().
        kind, deps = parent
        g.add_node(key, kind=kind, label=_parent_label(kind, deps), obj=stream,
                   unit=getattr(stream, "unit", None))
        for role, dep in deps.items():
            dep_key = _walk(dep, g, seen, counter) if isinstance(dep, Stream) else _const_node(g, dep, counter)
            g.add_edge(dep_key, key, role=role)
        return key

    return _walk_source_node(stream, g, seen, counter)


def build_graph(*streams):
    """Build a ``networkx.DiGraph`` describing how *streams* were derived.

    Nodes carry a ``kind`` (``"channel"``, ``"constant"``, ``"op"``,
    ``"filter"``, ``"categorize"``/``"categorize_scan"``/``"categorizeBy"``,
    or ``"unknown"``), a human-readable ``label``, and (except for constants)
    an ``obj`` attribute holding the live ``Stream`` itself. Edges carry a
    ``role`` (``"arg"``, a kwarg name, ``"data"``, ``"mask"``, ``"key"``, ...)
    and point from dependency to dependent, so leaves (real channels and
    constants) have no incoming edges.

    Parameters
    ----------
    *streams : Stream
        One or more Streams to use as graph roots. Shared upstream
        dependencies (e.g. two Streams both derived from the same channel)
        collapse to the same node.

    Returns
    -------
    networkx.DiGraph

    See Also
    --------
    draw : Render the graph returned here with matplotlib.
    """
    nx = _require_networkx()
    g = nx.DiGraph()
    seen = set()
    counter = itertools.count()
    for s in streams:
        _walk(s, g, seen, counter)
    return g


def _layered_layout(g):
    nx = _require_networkx()
    pos = {}
    for x, generation in enumerate(nx.topological_generations(g)):
        for y, node in enumerate(sorted(generation, key=str)):
            pos[node] = (x, -y)
    return pos


def draw(g, ax=None):
    """Render a graph built by :func:`build_graph` with matplotlib.

    Layout places real channels/constants (no incoming edges) on the left and
    each derived node one column to the right of its furthest dependency
    (``networkx.topological_generations``), so dependencies always point
    left-to-right.

    Parameters
    ----------
    g : networkx.DiGraph
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    nx = _require_networkx()
    import matplotlib.pyplot as plt

    if ax is None:
        n_gen = len(list(nx.topological_generations(g))) or 1
        _, ax = plt.subplots(figsize=(2.2 * n_gen, 1 + 0.6 * len(g)))

    pos = _layered_layout(g)
    node_colors = [_KIND_COLORS.get(g.nodes[n].get("kind"), "#cccccc") for n in g.nodes]
    labels = {n: g.nodes[n].get("label", str(n)) for n in g.nodes}
    nx.draw_networkx(
        g, pos=pos, ax=ax, labels=labels, node_color=node_colors,
        node_size=1800, font_size=8, arrows=True, edgecolors="black",
    )
    edge_labels = {(u, v): d.get("role", "") for u, v, d in g.edges(data=True)}
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels, ax=ax, font_size=7)
    ax.set_axis_off()
    return ax
