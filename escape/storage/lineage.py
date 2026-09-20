"""Lineage: Arrays remember how they were made, and from which tunable parameters.

Whenever an operation involves a :class:`Param` -- a labeled, changeable value
-- or an Array that itself carries lineage, the resulting Array gets a
``.lineage`` node recording *which function was called on which inputs*. The
Arrays keep their data exactly as before; the lineage is only extra
bookkeeping. Calling ``array.evaluate()`` replays the recorded calls with the
**current** parameter values, recomputing only what became stale, and
``array.plot(live=True)`` (see :mod:`escape.live`) uses that to redraw when a
parameter changes -- e.g. when a span on a histogram is dragged.

Design points
-------------
* **Generic.** A recorded step is just "call this function with these inputs".
  ``filter``/``digitize`` get no special treatment beyond turning their plain
  numeric arguments into labeled Params; any ``escaped()`` function,
  ``map_index_blocks`` call or reduction records the same way. To make a
  custom function tunable, pass it a :class:`Param` -- it receives the plain
  value.
* **Free unless used.** Recording only happens when a Param or an Array with
  lineage is among the inputs (or the operation introduces Params itself, as
  ``filter``/``digitize`` do). Ordinary chains carry no lineage at all.
  ``ENABLED = False`` (or ``ESCAPE_LINEAGE=0`` in the environment) switches
  recording off entirely; Params passed to operations are still resolved to
  their values, so everything keeps working, just without live updates.
* **Nodes cache.** Each node holds its current result plus a stamp of the
  parameter versions / upstream generations it was computed from, so a
  refresh recomputes only stale nodes and several live plots share the work.
* **Lineage keeps inputs alive.** A recorded chain holds the intermediate
  results of its nodes in memory. Fine for data that fits in memory (its
  intended use); switch it off for huge intermediates.
"""

from __future__ import annotations

import functools
import itertools
import logging
import os
import threading
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("ESCAPE_LINEAGE", "1").lower() not in ("0", "false", "off", "no")

_state = threading.local()


def _depth():
    return getattr(_state, "depth", 0)


@contextmanager
def _nested():
    """While active, wrapped operations neither introduce Params nor record
    (they are internals of an outer recorded call, or a replay)."""
    _state.depth = _depth() + 1
    try:
        yield
    finally:
        _state.depth -= 1


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def _same(a, b):
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b)
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


_batch_depth = 0
_batch_pending = []  # observers to call when the outermost batch() ends


def _fire(observers):
    done = []
    for cb in observers:
        if any(cb == d for d in done):
            continue
        done.append(cb)
        try:
            cb()
        except Exception:
            logger.exception("lineage: parameter observer %r failed", cb)


@contextmanager
def batch():
    """Defer observer notifications until the block ends, so setting several
    Params (e.g. both ends of a range) triggers one update, not one each."""
    global _batch_depth
    _batch_depth += 1
    try:
        yield
    finally:
        _batch_depth -= 1
        if _batch_depth == 0 and _batch_pending:
            pending = list(_batch_pending)
            _batch_pending.clear()
            _fire(pending)


class Param:
    """A labeled, changeable value.

    Pass one wherever an operation takes a plain value (a threshold, bin
    edges, a factor for your own ``escaped`` function...); the operation
    receives ``.value``. Results downstream remember the Param, so they can be
    re-evaluated when it changes (:meth:`escape.Array.evaluate`) and exposed
    as an input field by live plots.

    Parameters
    ----------
    value
        Initial value (number, bool, array, ...).
    name : str, optional
        Label shown in panels/`describe`; defaults to ``p1``, ``p2``, ...
    bounds : (float, float), optional
        Lets panels use a slider.
    """

    _counter = itertools.count(1)

    def __init__(self, value, name=None, bounds=None):
        self._value = value
        self.name = name or f"p{next(Param._counter)}"
        self.bounds = bounds
        self.version = 0
        self._observers = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.set(value)

    def set(self, value):
        """Set a new value; notifies observers unless it is unchanged."""
        if _same(self._value, value):
            return
        self._value = value
        self.version += 1
        if _batch_depth:
            _batch_pending.extend(self._observers)
        else:
            _fire(list(self._observers))

    def observe(self, callback):
        """Call ``callback()`` (no arguments) whenever the value changes.
        Returns a function that removes the observer again."""
        self._observers.append(callback)

        def remove():
            if callback in self._observers:
                self._observers.remove(callback)

        return remove

    def __repr__(self):
        return f"Param({self.name!r}, {self._value!r})"

    def __deepcopy__(self, memo):
        return self  # a Param is shared identity, never duplicated

    def __copy__(self):
        return self


# ---------------------------------------------------------------------------
# Lineage nodes
# ---------------------------------------------------------------------------


class Ref:
    """Points at one output of a :class:`Node`, as of generation ``gen``.

    ``Array._lineage`` is a Ref; recorded args reference their upstream Arrays
    through Refs (not the Arrays), so superseded snapshots can be freed.
    """

    __slots__ = ("node", "idx", "gen")

    def __init__(self, node, idx, gen):
        self.node = node
        self.idx = idx
        self.gen = gen

    def get(self):
        v = self.node.get()
        return v if self.idx is None else v[self.idx]

    # Copying an Array shares its lineage; pickling drops it (it holds
    # functions/Params/figures that must not travel).
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (type(None), ())


def _is_array(x):
    from .storage import Array

    return isinstance(x, Array)


# Containers are looked into one level deep for Params/Arrays, but only when
# short: a long list is data (e.g. an index selection), and scanning it on
# every Array.__getitem__ would cost more than the bookkeeping is worth.
_MAX_CONTAINER = 32


def _walk(obj):
    """Yield the leaves of args/kwargs: one level into short lists/tuples/dicts."""
    if isinstance(obj, (list, tuple)) and len(obj) <= _MAX_CONTAINER:
        for x in obj:
            yield x
    elif isinstance(obj, dict) and len(obj) <= _MAX_CONTAINER:
        yield from obj.values()
    else:
        yield obj


def _leaves(args, kwargs):
    for a in itertools.chain(args, kwargs.values()):
        for x in _walk(a):
            yield x


def _scan(args, kwargs):
    """(has a Param, has an Array with lineage) among the operands."""
    has_param = has_lin = False
    for x in _leaves(args, kwargs):
        if type(x) is Param:
            has_param = True
        elif getattr(x, "_lineage", None) is not None:
            has_lin = True
    return has_param, has_lin


def _map(obj, fn):
    if isinstance(obj, list) and len(obj) <= _MAX_CONTAINER:
        return [fn(x) for x in obj]
    if isinstance(obj, tuple) and len(obj) <= _MAX_CONTAINER:
        return tuple(fn(x) for x in obj)
    if isinstance(obj, dict) and len(obj) <= _MAX_CONTAINER:
        return {k: fn(v) for k, v in obj.items()}
    return fn(obj)


def _resolve(args, kwargs):
    """Plain values for a call: Params -> their value."""

    def fn(x):
        return x.value if type(x) is Param else x

    return tuple(_map(a, fn) for a in args), {k: _map(v, fn) for k, v in kwargs.items()}


def _encode(x):
    """Recorded form of an operand: Arrays with lineage become Refs."""
    ref = getattr(x, "_lineage", None)
    if ref is not None:
        return Ref(ref.node, ref.idx, ref.gen)
    return x


class Node:
    """One recorded call: ``func(*args, **kwargs)`` plus its cached result.

    Attributes
    ----------
    func, label
        The function called and a display name.
    value
        Current (cached) result -- an Array or a tuple of outputs.
    tool
        Optional interactive tool (e.g. the histogram selector) that created
        this node's Params.
    """

    def __init__(self, func, args, kwargs, label):
        self.func = func
        self.label = label
        self.args = tuple(_map(a, _encode) for a in args)
        self.kwargs = {k: _map(v, _encode) for k, v in kwargs.items()}
        self.tool = None
        self.value = None
        self.gen = 0
        leaves = list(_leaves(self.args, self.kwargs))
        self.refs = tuple(x for x in leaves if isinstance(x, Ref))
        self.params = tuple(dict.fromkeys(x for x in leaves if type(x) is Param))
        self._stamp = self._current_stamp(at_record=True)

    def _current_stamp(self, at_record=False):
        gens = tuple(r.gen if at_record else r.node.gen for r in self.refs)
        return tuple(p.version for p in self.params), gens

    def get(self):
        """Current result, recomputing this node (and stale upstream nodes)
        with the present Param values if anything changed."""
        for r in self.refs:
            r.node.get()
        stamp = self._current_stamp()
        # The stamp is (Param versions, upstream generations): equal to what
        # the current value was computed from -> nothing to do.
        if stamp != self._stamp:
            self._recompute()
            self._stamp = stamp
        return self.value

    def _resolve_input(self, x):
        if isinstance(x, Ref):
            return x.get()
        if type(x) is Param:
            return x.value
        return x

    def _recompute(self):
        args = tuple(_map(a, self._resolve_input) for a in self.args)
        kwargs = {k: _map(v, self._resolve_input) for k, v in self.kwargs.items()}
        with _nested():
            out = self.func(*args, **kwargs)
        self.value = out
        self.gen += 1
        self._attach(out)

    def _attach(self, out):
        if isinstance(out, tuple):
            for i, o in enumerate(out):
                if _is_array(o):
                    o._lineage = Ref(self, i, self.gen)
        elif _is_array(out):
            out._lineage = Ref(self, None, self.gen)

    def __repr__(self):
        return f"<lineage {describe_node(self)}>"


def record(func, args, kwargs, out, label=None):
    """Attach a lineage node for ``func(*args, **kwargs) -> out`` to every
    Array in ``out``. Never raises (lineage is best-effort bookkeeping)."""
    try:
        outs = out if isinstance(out, tuple) else (out,)
        if not any(_is_array(o) for o in outs):
            return None
        node = Node(func, args, kwargs, label or getattr(func, "__name__", "call"))
        node.value = out
        node._attach(out)
        return node
    except Exception:
        logger.debug("lineage: recording %r failed", func, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# The decorator that hooks storage operations in
# ---------------------------------------------------------------------------

SKIP = object()  # returned by an `introduce` hook: run the call untouched


def recorded(label=None, introduce=None):
    """Decorator: resolve Params in the call and record lineage on Array
    outputs (see the module docstring for when that happens).

    ``introduce(args, kwargs)`` may return ``(args, kwargs)`` with plain
    arguments swapped for labeled Params (this is how ``filter``'s thresholds
    and ``digitize``'s bins become tunable), ``SKIP`` to bypass everything, or
    ``None`` for no change. It only runs for a top-level call while lineage is
    enabled.
    """

    def deco(f):
        name = label or getattr(f, "__name__", "call")

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            top = ENABLED and _depth() == 0
            if introduce is not None and top:
                r = introduce(args, kwargs)
                if r is SKIP:
                    return f(*args, **kwargs)
                if r is not None:
                    args, kwargs = r
            has_param, has_lin = _scan(args, kwargs)
            if not (has_param or has_lin):
                return f(*args, **kwargs)
            rec_args, rec_kwargs = args, kwargs
            if has_param:
                args, kwargs = _resolve(args, kwargs)
            if not top:
                return f(*args, **kwargs)
            with _nested():
                out = f(*args, **kwargs)
            record(f, rec_args, rec_kwargs, out, name)
            return out

        wrapper._recorded_inner = f
        return wrapper

    return deco


def _is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def introduce_filter(args, kwargs):
    """filter(array, lo, hi): plain numeric thresholds become Params."""
    if len(args) < 2:
        return SKIP  # filter() without thresholds -> interactive selector
    array = args[0]
    name = getattr(array, "name", None) or "array"
    thresholds = args[1:]
    names = ("min", "max") if len(thresholds) == 2 else tuple(f"arg{i}" for i in range(len(thresholds)))
    new = [
        Param(t, f"{name} {n}") if _is_number(t) else t
        for t, n in zip(thresholds, names)
    ]
    return (args[0], *new), kwargs


def introduce_digitize(args, kwargs):
    """digitize(array, bins): plain bin edges become one Param."""
    bins = args[1] if len(args) > 1 else kwargs.get("bins")
    if bins is None:
        return SKIP  # digitize() without bins -> interactive selector
    if type(bins) is Param or not hasattr(bins, "__len__"):
        return None
    array = args[0]
    name = getattr(array, "name", None) or "array"
    p = Param(np.asarray(bins), f"{name} bins")
    if len(args) > 1:
        return (args[0], p, *args[2:]), kwargs
    return args, {**kwargs, "bins": p}


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _upstream_nodes(node):
    """Nodes reachable from ``node`` (itself included), upstream first."""
    order, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for r in n.refs:
            visit(r.node)
        order.append(n)

    visit(node)
    return order


def upstream_params(obj):
    """Every Param that influences ``obj`` (an Array, Node, or a live
    ``escape.stream.Stream``), upstream first."""
    stream_params = getattr(obj, "_upstream_params", None)
    if callable(stream_params):  # duck-typed: a Stream walks its own graph
        return stream_params()
    node = obj if isinstance(obj, Node) else getattr(getattr(obj, "_lineage", None), "node", None)
    if node is None:
        return []
    out = []
    for n in _upstream_nodes(node):
        for p in n.params:
            if p not in out:
                out.append(p)
    return out


def _short(x):
    if isinstance(x, Ref):
        return describe_node(x.node)
    if type(x) is Param:
        return f"{x.name}={_short_val(x.value)}"
    if _is_array(x):
        return x.name or "Array"
    return _short_val(x)


def _short_val(v):
    if isinstance(v, np.ndarray):
        return f"array{v.shape}"
    if callable(v):
        return getattr(v, "__name__", "func")
    r = repr(v)
    return r if len(r) < 24 else r[:21] + "..."


def describe_node(node):
    parts = [_short(a) for a in node.args] + [f"{k}={_short(v)}" for k, v in node.kwargs.items()]
    return f"{node.label}({', '.join(parts)})"


def describe(array):
    """One-line text of how ``array`` was made and from which Params."""
    if callable(getattr(array, "_upstream_params", None)):  # a Stream
        return str(getattr(array, "name", "Stream"))
    ref = getattr(array, "_lineage", None)
    if ref is None:
        return f"{getattr(array, 'name', None) or 'Array'} (no lineage)"
    return describe_node(ref.node)
