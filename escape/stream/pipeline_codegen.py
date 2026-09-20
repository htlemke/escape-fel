"""Generate a cam_server ``pipeline_type: "stream"`` process() script from a
Stream's computation graph (see :mod:`escape.stream.graph`).

This is the piece that was missing from a plain graph *display*: it doesn't
just show what a Stream depends on, it walks the graph in dependency order
and emits the equivalent per-event Python -- the ``process(data, pulse_id,
timestamp, parameters)`` contract validated against the real Bernina
cam_server (see the pipeline-offload design discussion and
``example_pipeline_offload.ipynb``).

Coverage is deliberately limited to what's actually mechanical:

- real channels, constants, arithmetic operators, ``element()``/indexing,
  and opposite-mask ``filter()``/``[mask]`` pairs -- all *stateless*, computed
  fresh from this event's inputs.
- ``running_mean``/``running_std``/``running_median``/``running_mad`` in
  their plain (unweighted, non-nan-skipping) form -- the one *stateful* op
  in escape.stream today, translated to an explicit module-level
  ``deque`` since cam_server's "stream" pipeline type gives ``process()`` no
  ``init``/state argument of its own (unlike its "custom" type).

Anything else -- ``categorize()``/``digitize()`` (a scan-step concept with no
meaning in a stateless per-event server script), weighted or
nan-skipping running stats, or an arbitrary user function with no
``_graph_codegen`` hook and no known-operator identity -- raises
``NotImplementedError`` naming exactly what's missing, rather than silently
emitting code that doesn't match what the Stream actually computes.
"""

import re

from escape.storage.lineage import Param

from .escape_stream import Stream, _RunningStat, _operatorsCompare, _operatorsJoin, _operatorsSingle

try:
    import operator as _operator
    # Comparisons included, so numeric filters (`stream.filter(lo, hi)`, i.e.
    # `(s >= lo) & (s <= hi)`) offload too; a Param limit is baked in as its
    # current value (see Stream.filter).
    _JOIN_SYMBOLS = dict(_operatorsJoin + _operatorsCompare)
    _SINGLE_SYMBOLS = dict(_operatorsSingle)
    # Stream.__invert__ uses operator.not_ directly (not via _operatorsSingle,
    # see escape_stream.py) -- added here so `~stream` codegens too.
    _SINGLE_SYMBOLS[_operator.not_] = "not"
except ImportError:  # pragma: no cover -- operator is stdlib, always present
    _JOIN_SYMBOLS, _SINGLE_SYMBOLS = {}, {}

_UNWEIGHTED_STATS = ("mean", "std", "median", "mad")


def _safe_name(label, idx):
    base = re.sub(r"\W+", "_", str(label)).strip("_") or "v"
    return f"v{idx}_{base}"[:60]


def _find_pred_var(g, node, role, arg_obj, var):
    """The graph node feeding *node* via edge-role *role* whose `obj` is
    `arg_obj` (arg identity, not just role, since a node can have several
    same-role predecessors, e.g. several plain 'arg' edges)."""
    for u, _, d in g.in_edges(node, data=True):
        if d.get("role") == role and g.nodes[u].get("obj") is arg_obj:
            return var[u]
    raise KeyError((node, role, arg_obj))


def _emit_running_stat(func, stream_obj, arg_vars, args_is_esc, idx, lines, imports, state):
    if not all(args_is_esc) or len(arg_vars) != 1:
        raise NotImplementedError(
            f"generate_process_script: {stream_obj.name!r} is a *weighted* "
            "running_* stat -- only the unweighted form is supported."
        )
    if func.skipnan:
        raise NotImplementedError(
            f"generate_process_script: {stream_obj.name!r} is a running_nan{func.stat} "
            "-- only the non-nan-skipping variants are supported."
        )
    if func.stat not in _UNWEIGHTED_STATS:
        raise NotImplementedError(f"generate_process_script: unknown stat {func.stat!r}")

    imports.add("import statistics")
    imports.add("from collections import deque")
    window = f"_window_{idx}"
    state.append(f"{window} = deque(maxlen={stream_obj.N_acc!r})")

    (input_var,) = arg_vars
    out = _safe_name(f"running_{func.stat}", idx)
    lines.append(f"if {input_var} is not None:")
    lines.append(f"    {window}.append({input_var})")
    lines.append(f"if not {window}:")
    lines.append(f"    {out} = None")
    lines.append("else:")
    if func.stat == "mean":
        lines.append(f"    {out} = statistics.mean({window})")
    elif func.stat == "std":
        lines.append(f"    {out} = statistics.pstdev({window})")
    elif func.stat == "median":
        lines.append(f"    {out} = statistics.median({window})")
    else:  # mad
        lines.append(f"    _med_{idx} = statistics.median({window})")
        lines.append(f"    {out} = statistics.median([abs(_x - _med_{idx}) for _x in {window}])")
    return out


def generate_process_script(stream, output_name=None, header=None):
    """Return cam_server ``pipeline_type: "stream"`` script source for *stream*.

    Parameters
    ----------
    stream : Stream
        The (possibly derived) Stream to translate. Raises
        ``NotImplementedError`` if its graph contains anything not covered
        (see the module docstring) -- never emits code that silently doesn't
        match what the Stream actually computes.
    output_name : str, optional
        Key under which the computed value is published (default:
        ``stream.name``, e.g. ``"(i[~isref] / i[isref]_running_mean(N_acc=5))"``
        -- pass an explicit short name for a nicer output channel).
    header : str, optional
        Overrides the auto-generated module docstring.

    Returns
    -------
    str
        Self-contained script source, ready to hand to
        ``cam_server.PipelineClient.upload_user_script`` /
        ``set_user_script`` and reference as ``"function"`` in a
        ``pipeline_type: "stream"`` instance config.
    """
    from .graph import build_graph
    import networkx as nx  # build_graph already requires it; safe to import here too

    g = build_graph(stream)

    for _, d in g.nodes(data=True):
        if d.get("kind") in ("categorize", "categorize_scan", "categorizeBy"):
            raise NotImplementedError(
                "generate_process_script: scan/categorize structure "
                f"({d.get('label')!r}) has no meaning in a stateless per-event "
                "pipeline script."
            )

    order = list(nx.topological_sort(g))
    var = {}
    lines = []
    imports = set()
    state = []
    channels = []

    for i, node in enumerate(order):
        d = g.nodes[node]
        kind = d["kind"]

        if kind == "channel":
            var[node] = _safe_name(d["label"], i)
            channels.append(d["label"])
            lines.append(f'{var[node]} = data.get({d["label"]!r})')

        elif kind == "constant":
            var[node] = _safe_name("c", i)
            value = d["obj"].value if isinstance(d["obj"], Param) else d["obj"]  # a Param -> its current value
            lines.append(f"{var[node]} = {value!r}")

        elif kind == "filter":
            stream_obj = d["obj"]
            src = stream_obj._source
            data_var = (
                _find_pred_var(g, node, "data", src._inner_stream, var)
                if src._inner_stream is not None
                else None
            )
            mask_var = _find_pred_var(g, node, "mask", src._mask, var)
            if data_var is None:
                raise NotImplementedError(
                    f"generate_process_script: {stream_obj.name!r}'s filter has no "
                    "traceable upstream Stream (built by hand from a bare Source)."
                )
            var[node] = _safe_name("filt", i)
            lines.append(f"{var[node]} = {data_var} if {mask_var} else None")

        elif kind == "op":
            stream_obj = d["obj"]
            proc = stream_obj._source.procObj
            func = proc.func

            arg_vars = [
                _find_pred_var(g, node, "arg", a, var) if is_esc else repr(a.value if isinstance(a, Param) else a)
                for a, is_esc in zip(proc.args, proc.args_is_esc)
            ]

            if isinstance(func, _RunningStat):
                var[node] = _emit_running_stat(
                    func, stream_obj, arg_vars, proc.args_is_esc, i, lines, imports, state
                )
            elif func in _JOIN_SYMBOLS:
                sym = _JOIN_SYMBOLS[func]
                a, b = arg_vars
                var[node] = _safe_name(sym, i)
                lines.append(f"{var[node]} = None if ({a} is None or {b} is None) else ({a} {sym} {b})")
            elif func in _SINGLE_SYMBOLS:
                sym = _SINGLE_SYMBOLS[func]
                (a,) = arg_vars
                py_sym = "not " if sym == "not" else sym
                var[node] = _safe_name(sym, i)
                lines.append(f"{var[node]} = None if {a} is None else ({py_sym}{a})")
            elif hasattr(func, "_graph_codegen"):
                var[node] = _safe_name("op", i)
                expr = func._graph_codegen(arg_vars)
                esc_vars = [v for v, is_esc in zip(arg_vars, proc.args_is_esc) if is_esc]
                guard = " or ".join(f"{v} is None" for v in esc_vars)
                prefix = f"None if ({guard}) else " if guard else ""
                lines.append(f"{var[node]} = {prefix}({expr})")
            else:
                raise NotImplementedError(
                    f"generate_process_script: don't know how to translate "
                    f"{stream_obj.name!r} (function {func!r}) -- give it a "
                    f"_graph_codegen(arg_exprs) hook (see Stream.element()), or "
                    f"write this Stream's script by hand."
                )

        else:
            raise NotImplementedError(
                f"generate_process_script: unsupported node kind {kind!r} ({d.get('label')!r})."
            )

    out_var = var[order[-1]]
    body = "\n".join("    " + l for l in lines)
    # Imports first, then state declarations that use them (deque(...)) -- do
    # not alphabetize the two together, "_window_0" sorts before "from ...".
    setup_block = "\n".join(sorted(imports) + state)
    setup_block = (setup_block + "\n\n") if setup_block else ""
    doc = header or (
        f"Auto-generated by escape.stream.pipeline_codegen.generate_process_script() "
        f"from: {stream.name}\nSubscribe this instance to bsread_channels={channels!r}."
    )
    name = output_name or stream.name

    return (
        f'"""{doc}"""\n'
        f"{setup_block}"
        f"def process(data, pulse_id, timestamp, parameters):\n"
        f"{body}\n"
        f"    if {out_var} is None:\n"
        f"        return None\n"
        f"    return {{{name!r}: {out_var}}}\n"
    )
