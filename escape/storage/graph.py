"""Recordable computation graphs for ``escape.storage.Array``.

Build a computation using ordinary escape.storage tools -- arithmetic,
``escaped()``-wrapped functions, ``Array.map_index_blocks()`` -- on
:func:`placeholder` Arrays that carry no real data, just a role name. Every
operation touching a placeholder is **recorded** instead of executed,
producing another placeholder-like ("unbound") Array carrying the recorded
operation. The intended end state (a later step, not implemented by this
module yet) is: serialize the recorded graph, reload it elsewhere, bind each
role to a real ``Array`` (e.g. lazily loaded from a ``DataSet``/HDF5 file --
that half already exists, see ``DataSet(lazy_loading=True)``), and replay
the recorded calls against the real data to get a genuine lazy, dask-backed
``Array`` -- no code generation needed, since replay is just calling the
same functions again with real Arrays substituted for their placeholder
inputs (unlike ``escape.stream.pipeline_codegen``, which has to emit text
because *its* target -- a cam_server script -- isn't itself a live Python
call).

**Recording is strictly opt-in and costs nothing when unused.**
``escaped()`` and ``Array.map_index_blocks()`` only take the recording
branch when a placeholder is among their operands; every other call --
which is to say, all ordinary real-data usage -- proceeds exactly as before,
completely unaffected. See each of their docstrings for the one-line check
that gates this.

Function identity has to survive being written to disk and reloaded in a
different process (possibly a different machine, days later) -- a live
function object isn't trustworthy for that, and unpickling an arbitrary
closure from disk is both fragile across environments and a real code-
execution surface. So every function that appears in a recorded graph --
both the op being called, and any plain function passed as an *argument*
(e.g. ``Array.map_index_blocks(foo, ...)``'s ``foo``) -- must be registered
under a stable name first (see :func:`register`). An unregistered function
raises a clear error rather than being silently (and unsafely) pickled.
"""

# ---------------------------------------------------------------------------
# Function registry
# ---------------------------------------------------------------------------

_REGISTRY = {}  # name -> function
_REGISTRY_NAMES = {}  # id(function) -> name


def register(name=None):
    """Decorator (or plain call, ``register("name")(func)``) making *func*
    recordable in a graph.

    ``@register()`` with no name uses ``func.__module__ + "." +
    func.__qualname__``; pass an explicit short name (``@register("mean")``)
    for something friendlier. Idempotent for the same function/name pair;
    raises if it would give one function two identities, or one name two
    functions -- a graph's replay has to be unambiguous.
    """

    def deco(func):
        nm = name or f"{func.__module__}.{func.__qualname__}"
        existing = _REGISTRY_NAMES.get(id(func))
        if existing is not None and existing != nm:
            raise ValueError(
                f"{func!r} is already registered as {existing!r}; can't also "
                f"register it as {nm!r}."
            )
        if nm in _REGISTRY and _REGISTRY[nm] is not func:
            raise ValueError(f"a different function is already registered as {nm!r}.")
        _REGISTRY[nm] = func
        _REGISTRY_NAMES[id(func)] = nm
        return func

    return deco


def get_name(func):
    """Registered name for *func*, or ``None`` if it isn't registered."""
    return _REGISTRY_NAMES.get(id(func))


def resolve(name):
    """The function registered under *name*. Raises ``KeyError`` if none is."""
    return _REGISTRY[name]


def _register_operator_table():
    """Pre-register the same operator tables ``Array``'s arithmetic dunders
    are generated from, so ``i0 - i`` etc. is recordable with no extra step."""
    from .storage import _operatorsJoin, _operatorsSingle

    for func, _symbol in _operatorsJoin + _operatorsSingle:
        name = f"operator.{func.__name__}"
        if get_name(func) is None:
            register(name)(func)


def _register_common_types():
    """Pre-register the dtype-like types that show up constantly as plain
    kwarg *values* (``dtype=bool``, ``dtype=np.float64``, ...) -- so the
    common case works with no extra step; anything more exotic still needs
    its own explicit @register(), same as a custom function would."""
    import numpy as np

    for t in (bool, int, float, complex, str):
        register(f"type.{t.__name__}")(t)
    for name in (
        "float16", "float32", "float64", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "complex64", "complex128", "bool_",
    ):
        t = getattr(np, name, None)
        if t is not None and get_name(t) is None:
            register(f"type.numpy.{name}")(t)


_register_common_types()


def _call_map_index_blocks(self_arr, foo, *args, **kwargs):
    """Registered stand-in for the ``Array.map_index_blocks`` *method call*
    shape (``self.map_index_blocks(foo, ...)``) -- graph nodes are always
    plain ``func(*args, **kwargs)`` calls, so replaying a method call goes
    through this free function instead of the unbound method itself."""
    return self_arr.map_index_blocks(foo, *args, **kwargs)


register("Array.map_index_blocks")(_call_map_index_blocks)


def _call_categorize(self_arr, other_arr):
    """Registered stand-in for ``Array.categorize``'s method-call shape --
    same reason as ``_call_map_index_blocks`` above."""
    return self_arr.categorize(other_arr)


register("Array.categorize")(_call_categorize)


# ---------------------------------------------------------------------------
# Function references as recordable *values* -- e.g. map_index_blocks(foo, ...)
# passes a plain function as an argument, distinct from the op being called.
# ---------------------------------------------------------------------------


class FuncRef:
    """A registered function, recorded as a graph *value* (an argument) --
    as opposed to the function *being called*, which is recorded as the op
    node itself. See ``Array.map_index_blocks()``'s ``foo`` argument."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"FuncRef({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, FuncRef) and self.name == other.name

    def resolve(self):
        return resolve(self.name)


class StatusRef:
    """A scalar recorded by *name* to be looked up in a run's status dict at
    bind time -- e.g. a per-run DAQ setting (``rounding_factor_keV``,
    ``threshold_min``) that doesn't exist until a real dataset is loaded, so
    it can't be a plain constant when the recipe is defined (typically at
    device-init time, long before any run happens).

    Not an Array, not callable -- ``_as_recordable()`` passes it through
    unchanged as an ordinary recorded value, same as any other constant;
    only :func:`bind` treats it specially (see there).
    """

    __slots__ = ("key",)

    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f"StatusRef({self.key!r})"

    def __eq__(self, other):
        return isinstance(other, StatusRef) and self.key == other.key


class BindError(Exception):
    """Raised by :func:`bind` when a placeholder role or StatusRef key isn't
    available. Its own type (rather than a bare ``KeyError``) so
    :func:`apply_recipes`'s best-effort skip-and-log can catch exactly this
    and let a genuine bug in a registered function surface normally."""


def _as_recordable(value):
    """Classify *value* for recording: an (upstream) Array, a registered-
    function reference, or a plain constant. Raises clearly for an
    unregistered callable -- never silently records something that can't be
    named/replayed later."""
    from .storage import Array

    if isinstance(value, Array):
        return value
    # A class (e.g. `dtype=bool`/`dtype=np.float64`, a common map_index_blocks/
    # dask kwarg) is callable but isn't meant as "call this as the op" here --
    # it's a plain value, just not a JSON-primitive one. Goes through the
    # same FuncRef/registry mechanism as any other callable rather than a
    # separate path -- see _register_common_types() for what's pre-registered
    # (bool/int/float/... and the common numpy dtypes); anything else needs
    # an explicit @register() the same as a custom function would.
    if callable(value):
        name = get_name(value)
        if name is None:
            raise ValueError(
                f"{value!r} is not registered -- decorate it with "
                f"@escape.storage.graph.register() to use it in a recorded "
                f"computation graph (see the module docstring)."
            )
        return FuncRef(name)
    return value


# ---------------------------------------------------------------------------
# Placeholder Arrays
# ---------------------------------------------------------------------------


def _unbound(role):
    """A matched (data, index) callable pair that raise a clear error if
    anything ever tries to materialize them -- reusing Array's *existing*
    `callable(data) or callable(index)` deferred-data support (see
    Array.__init__/`.data`/`.index`) rather than adding a new code path."""

    def _raise_data(data_selector=None):
        raise RuntimeError(
            f"Array role {role!r} is an unbound placeholder -- bind it to "
            f"real data (see escape.storage.graph.bind) before computing."
        )

    def _raise_index():
        raise RuntimeError(
            f"Array role {role!r} is an unbound placeholder -- bind it to "
            f"real data (see escape.storage.graph.bind) before computing."
        )

    return _raise_data, _raise_index


def placeholder(role, shape=None, dtype=None, unit=None):
    """A dummy ``Array`` standing in for real data that doesn't exist yet.

    Build a computation with ordinary escape.storage tools using one or more
    of these; every operation touching a placeholder is recorded instead of
    executed (see the module docstring). Actually computing anything from a
    placeholder (``.data``, ``.index``, ``.compute()``, ...) raises a clear
    error rather than silently doing the wrong thing.

    Parameters
    ----------
    role : str
        Name this placeholder is bound to real data by, later. Also used as
        the resulting Array's ``.name``.
    shape, dtype : optional
        Recorded for later sanity-checking at bind time; not enforced yet.
    unit : str, optional
    """
    from .source import Source
    from .storage import Array

    data_fn, index_fn = _unbound(role)
    src = Source(type="placeholder", role=role, shape=shape, dtype=dtype)
    # step_lengths=[] (not the default None): Array.__init__ defaults a None
    # step_lengths via `[len(index)]` with no guard for a *callable* index
    # (unlike the length-assertion just above it, which does check) -- an
    # explicit, non-None step_lengths sidesteps that without touching
    # Array.__init__ itself.
    return Array(
        data=data_fn, index=index_fn, source=src, name=role, unit=unit, step_lengths=[]
    )


def is_placeholder(obj):
    """True if *obj* is an unbound Array -- a placeholder leaf, or the
    (also-unbound) recorded result of an operation on one."""
    from .storage import Array

    return (
        isinstance(obj, Array)
        # NB: escape.storage.Array's Source lives on `.source` (no leading
        # underscore) -- unlike escape.stream.Stream's `._source`.
        and getattr(obj, "source", None) is not None
        and getattr(obj.source, "type", None) in ("placeholder", "symbolic_op")
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record_op(func, args, kwargs, name=None, unit=None):
    """Return a new unbound Array recording a call to *func* instead of
    executing it -- the shared hook both ``escaped()`` and
    ``Array.map_index_blocks()`` use to build a graph node. *func* must
    already be registered (see :func:`register`); raises clearly if not.
    """
    from .source import Source

    func_name = get_name(func)
    if func_name is None:
        raise ValueError(
            f"{func!r} is not registered -- decorate it with "
            f"@escape.storage.graph.register() to record a computation "
            f"through it (see the module docstring)."
        )
    rec_args = [_as_recordable(a) for a in args]
    rec_kwargs = {k: _as_recordable(v) for k, v in kwargs.items()}

    data_fn, index_fn = _unbound(name or func_name)
    src = Source(type="symbolic_op", func_name=func_name, args=rec_args, kwargs=rec_kwargs)
    from .storage import Array

    # See the identical note in placeholder() re: step_lengths=[].
    return Array(
        data=data_fn, index=index_fn, source=src, name=name or func_name, unit=unit,
        step_lengths=[],
    )


# ---------------------------------------------------------------------------
# Recipe registry -- named recorded computations, predefined once (e.g. by an
# eco device's __init__) and applied against whatever dataset actually
# contains that device's real channels later, at load time.
# ---------------------------------------------------------------------------

_RECIPES = {}  # name -> recorded (unbound) Array


def register_recipe(name, expr):
    """Record *expr* (built from :func:`placeholder`/``StatusRef``-using
    computations) under *name*, for later serialization (:func:`write_recipes`)
    and application (:func:`apply_recipes`).

    Parameters
    ----------
    name : str
        Output field name -- what the computed result is appended into a
        DataSet as, later.
    expr : Array
        Must be unbound (:func:`is_placeholder`) -- there's nothing to
        record about an already-real Array.
    """
    if not is_placeholder(expr):
        raise ValueError(
            f"register_recipe({name!r}): expr is a real Array, not a "
            f"recorded placeholder computation -- nothing to record."
        )
    _RECIPES[name] = expr


def get_recipes():
    """A copy of the current recipe registry (``{name: recorded Array}``)."""
    return dict(_RECIPES)


def clear_recipes():
    """Remove every registered recipe. Mainly for tests -- real devices
    should only ever add recipes, not remove others'."""
    _RECIPES.clear()


# ---------------------------------------------------------------------------
# Serialization -- a recorded graph has to survive being written to disk and
# reloaded in a different process (see the module docstring).
# ---------------------------------------------------------------------------


def _to_json_node(node, nodes_out, seen):
    from .storage import Array

    if isinstance(node, Array):
        key = str(id(node))
        ref = {"__ref__": key}
        if key in seen:
            return ref
        seen.add(key)
        if not is_placeholder(node):
            raise ValueError(
                f"cannot serialize {node!r} -- only placeholder/recorded-op "
                f"Array nodes are recordable, not a real, already-bound one."
            )
        src = node.source
        if src.type == "placeholder":
            nodes_out[key] = {
                "kind": "placeholder",
                "role": src.role,
                "shape": list(src.shape) if src.shape is not None else None,
                "dtype": src.dtype,
            }
        else:  # symbolic_op
            nodes_out[key] = {
                "kind": "op",
                "func_name": src.func_name,
                "args": [_to_json_node(a, nodes_out, seen) for a in src.args],
                "kwargs": {k: _to_json_node(v, nodes_out, seen) for k, v in src.kwargs.items()},
            }
        return ref
    if isinstance(node, FuncRef):
        return {"kind": "funcref", "name": node.name}
    if isinstance(node, StatusRef):
        return {"kind": "statusref", "key": node.key}
    return {"kind": "const", "value": node}


def to_config(recipes):
    """Serialize a ``{name: recorded Array}`` dict (e.g. from
    :func:`get_recipes`) into a plain, JSON-able dict. Shared upstream nodes
    (the same Array reused by several recipes, or within one recipe) are
    stored once and referenced, not duplicated.
    """
    nodes = {}
    seen = set()
    roots = {name: _to_json_node(expr, nodes, seen) for name, expr in recipes.items()}
    return {"nodes": nodes, "roots": roots}


def _from_json_node(rep, nodes, cache):
    if "__ref__" in rep:
        key = rep["__ref__"]
        if key in cache:
            return cache[key]
        node_def = nodes[key]
        if node_def["kind"] == "placeholder":
            obj = placeholder(node_def["role"], shape=node_def.get("shape"), dtype=node_def.get("dtype"))
        else:  # op
            args = [_from_json_node(a, nodes, cache) for a in node_def["args"]]
            kwargs = {k: _from_json_node(v, nodes, cache) for k, v in node_def["kwargs"].items()}
            obj = record_op(resolve(node_def["func_name"]), args, kwargs)
        cache[key] = obj
        return obj
    if rep["kind"] == "funcref":
        return FuncRef(rep["name"])
    if rep["kind"] == "statusref":
        return StatusRef(rep["key"])
    return rep["value"]


def from_config(config):
    """Inverse of :func:`to_config`: returns a ``{name: recorded Array}`` dict.

    Every ``func_name``/``FuncRef`` encountered must already be registered
    in *this* process -- reloading a graph doesn't restore code, only the
    call structure; the functions it names have to be imported/registered
    the same way they were when the graph was first built.
    """
    nodes = config["nodes"]
    cache = {}
    return {name: _from_json_node(ref, nodes, cache) for name, ref in config["roots"].items()}


def _json_default(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(
        f"{obj!r} (type {type(obj).__name__}) isn't JSON-serializable as a "
        f"recipe constant -- use a plain Python/numpy value."
    )


def write_recipes(path, recipes=None):
    """Write *recipes* (default: the whole current registry,
    :func:`get_recipes`) to *path* as JSON."""
    import json

    if recipes is None:
        recipes = get_recipes()
    with open(path, "w") as f:
        json.dump(to_config(recipes), f, indent=2, default=_json_default)


def load_recipes(path):
    """Read a ``{name: recorded Array}`` dict back from a file written by
    :func:`write_recipes`. Does not touch the in-memory registry -- pass the
    result to :func:`apply_recipes` directly, or ``_RECIPES.update(...)`` it
    yourself if you want it merged into the registry."""
    import json

    with open(path) as f:
        return from_config(json.load(f))


# ---------------------------------------------------------------------------
# Binding + replay
# ---------------------------------------------------------------------------


def bind(node, bindings, status=None):
    """Recursively substitute real values for every placeholder/StatusRef/
    FuncRef leaf in a recorded graph, calling each recorded op along the way
    -- returns a real, computed ``Array`` (or a plain value, for a
    constant/FuncRef-resolved leaf).

    Raises :class:`BindError` naming the missing role/key if something isn't
    provided -- an all-or-nothing bind. For a *best-effort* application
    across many independent recipes where not every one necessarily applies
    to a given dataset, see :func:`apply_recipes`, which catches exactly
    this and skips just that recipe.

    Parameters
    ----------
    node : Array | FuncRef | StatusRef | Any
        Typically one recipe's recorded Array (from :func:`get_recipes`/
        :func:`load_recipes`), but any recordable value works recursively.
    bindings : dict
        ``{role: real Array}`` -- role names are exactly what
        :func:`placeholder` was given.
    status : dict, optional
        Looked up by :class:`StatusRef` keys. Omit if the graph has none.
    """
    from .storage import Array

    if isinstance(node, FuncRef):
        return node.resolve()
    if isinstance(node, StatusRef):
        if status is None or node.key not in status:
            raise BindError(f"status key {node.key!r} is not available")
        return status[node.key]
    if not isinstance(node, Array):
        return node  # a plain constant
    if not is_placeholder(node):
        return node  # already real -- e.g. a real Array passed as a plain arg

    src = node.source
    if src.type == "placeholder":
        if src.role not in bindings:
            raise BindError(f"role {src.role!r} is not bound")
        return bindings[src.role]

    # symbolic_op: bind every arg/kwarg, then actually call the function --
    # no code generation needed (unlike escape.stream.pipeline_codegen),
    # since a real escape.storage call already produces a real, lazy Array.
    real_args = [bind(a, bindings, status) for a in src.args]
    real_kwargs = {k: bind(v, bindings, status) for k, v in src.kwargs.items()}
    func = resolve(src.func_name)
    return func(*real_args, **real_kwargs)


def apply_recipes(dataset, recipes=None, status_key="status_run_start", status=None, verbose=True):
    """Bind and replay each of *recipes* against *dataset*, appending every
    one that successfully applies back into *dataset* under its name.

    **Best-effort, multi-pass**: a recipe whose placeholder roles or
    ``StatusRef`` keys aren't available yet is retried on the next pass
    rather than given up on immediately -- one recipe's *result* becomes
    available as a binding for any other recipe once it succeeds (e.g.
    ``det_diff.dap0 = bernina.pump_delayed.categorize(...)`` only resolves
    once ``bernina.pump_delayed``'s own recipe has already been applied),
    so recipes don't need to be registered in dependency order. A recipe
    still missing its roles/status keys once no further progress is made in
    a pass is skipped for good (logged, not raised) -- mirroring the
    ``if k+'.data' in d.datasets``/``try/except`` guards real post-analysis
    code needs today, since not every field exists for every run. A genuine
    bug inside a registered function (as opposed to a missing role/key) is
    *not* retried -- it's logged with its traceback once and that recipe is
    dropped, so one broken recipe can't silently take down the rest of the
    load, but also can't spin forever.

    Parameters
    ----------
    dataset : DataSet
        Bindings are read from ``dataset.datasets`` by name -- a
        placeholder's role must match a real entry's name exactly (e.g. a
        Jungfrau's ``"<name>.data"`` channel). Results are written back via
        ``dataset.append(result, name=name)``.
    recipes : dict, optional
        Default: the whole current registry (:func:`get_recipes`) -- pass
        an explicit dict (e.g. from :func:`load_recipes`) to apply a
        specific serialized set instead.
    status_key : str
        Which entry of ``dataset.datasets`` holds the run's status dict
        (looked up for ``StatusRef``s). Default matches the status-dataset
        convention already used by ``load_dataset_from_scan``.
    status : dict, optional
        Use this instead of ``dataset.datasets[status_key]`` -- e.g. a
        default/fallback status dict for a dataset that doesn't have one
        (mirroring the original snippet's ``status_default`` fallback).
    """
    if recipes is None:
        recipes = get_recipes()
    if status is None:
        try:
            status = dataset.datasets[status_key]
        except Exception:
            status = {}

    bindings = dict(dataset.datasets)
    pending = dict(recipes)
    applied = []
    last_error = {}

    while pending:
        made_progress = False
        for name in list(pending):
            try:
                result = bind(pending[name], bindings, status)
            except BindError as e:
                last_error[name] = str(e)
                continue  # maybe resolvable once another recipe below succeeds
            except Exception:
                if verbose:
                    print(f"apply_recipes: {name!r} failed --")
                    import traceback

                    traceback.print_exc(chain=False)
                del pending[name]  # a real bug, not a missing dependency -- don't retry
                continue
            dataset.append(result, name=name)
            bindings[name] = result
            applied.append(name)
            del pending[name]
            made_progress = True
        if not made_progress:
            break

    if verbose:
        for name in pending:
            print(f"apply_recipes: skipping {name!r} -- {last_error.get(name, 'unresolved')}")
    return applied
