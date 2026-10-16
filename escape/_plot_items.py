"""What escape's Select and History toolbar tools treat as *one* plotted
item, and how to turn one into storable data and back.

Some plots are drawn from several matplotlib artists that only make sense
together:

* :func:`escape.plot_utilities.errortube` -- a line plus a shaded band;
* ``ax.errorbar`` (what ``Scan.plot()`` uses) -- a data line plus error-bar
  segments and, with ``capsize``, cap markers (which are themselves
  ``Line2D`` objects in ``ax.lines``).

:func:`collect_items` groups those into one :class:`PlotItem` each, so the
tools can show, hide, save and recall them as a unit. Plain lines are one
item each.

Extra information travels on the artists themselves in a plain dict
attribute, ``escape_meta`` (see :func:`tag`). escape's own plotting
functions fill it in (Array name, scan parameter, run number); the History
tool saves it with the data and restores it on recall.

Not public API apart from :func:`tag` (re-exported as
``escape.plot_utilities.tag``). Imports only numpy/matplotlib.
"""

from __future__ import annotations

import json
import uuid

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
from matplotlib.container import Container, ErrorbarContainer
from matplotlib.lines import Line2D

META_ATTR = "escape_meta"

# keys in escape_meta used internally to group artists; hidden from the
# History tool's metadata column and never copied onto a recalled item
_INTERNAL_KEYS = ("group", "kind")


# ---------------------------------------------------------------------------
# tagging
# ---------------------------------------------------------------------------


def _flatten(obj):
    if obj is None:
        return
    if isinstance(obj, Container):
        yield obj
        for child in obj:
            yield from _flatten(child)
    elif isinstance(obj, (list, tuple)):
        for child in obj:
            yield from _flatten(child)
    else:
        yield obj


def tag(artists, **info):
    """Attach extra information to plotted artists, for escape's History
    and Select toolbar tools.

    ``artists`` can be one artist, a list of them (what ``ax.plot`` returns),
    an ``ErrorbarContainer`` (what ``ax.errorbar`` returns), or a
    ``(line, band)`` pair (what :func:`errortube` returns). All of them end
    up sharing one dict, ``artist.escape_meta``, updated with ``info``;
    calling ``tag`` again adds to it. Returns ``artists`` unchanged, so it can
    wrap a plot call::

        line, = tag(ax.plot(x, y, label="alpha"), sample="LaB6", run=142)

    Keep values simple (numbers, text, lists of those): the History tool
    stores anything else as its printed text.
    """
    targets = [a for a in _flatten(artists) if a is not None]
    meta = next((getattr(a, META_ATTR) for a in targets if isinstance(getattr(a, META_ATTR, None), dict)), None)
    if meta is None:
        meta = {}
    meta.update(info)
    for a in targets:
        try:
            setattr(a, META_ATTR, meta)
        except Exception:
            pass
    return artists


def new_group(artists, kind, **info):
    """Tag ``artists`` as one combined item of ``kind`` (e.g. "errortube")."""
    return tag(artists, group=uuid.uuid4().hex[:10], kind=kind, **info)


def array_meta(array, scan_parameter=None):
    """What escape knows about where an ``Array``'s plot came from -- the
    automatic part of ``escape_meta`` for ``Array.plot``/``Scan.plot``."""
    meta = {}
    name = getattr(array, "name", None)
    if name:
        meta["array"] = str(name)
    if scan_parameter:
        meta["scan_parameter"] = str(scan_parameter)
    try:
        parameter = array.scan.parameter if hasattr(array, "scan") else array.parameter
    except Exception:
        parameter = None
    if isinstance(parameter, dict):
        for key, par in parameter.items():
            if "run" not in str(key).lower():
                continue
            try:
                values = np.unique(np.asarray(par["values"]).ravel())
            except Exception:
                continue
            values = [v.item() if hasattr(v, "item") else v for v in values]
            if len(values) == 1:
                meta["run"] = values[0]
            elif 1 < len(values) <= 20:
                meta["run"] = values
            break
    return meta


def jsonable(value):
    """``value`` as something ``json.dumps`` accepts; text for anything else."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)


def visible_meta(meta):
    """``meta`` without the internal grouping keys."""
    return {k: v for k, v in (meta or {}).items() if k not in _INTERNAL_KEYS}


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

_AUTO_LABEL_PREFIX = "_"


class PlotItem:
    """One logical plotted thing: a line, an errortube (line + band) or an
    errorbar plot (line + bars + caps)."""

    def __init__(self, kind, main, members, container=None):
        self.kind = kind
        self.main = main  # the Line2D carrying label/colour (None: errorbar with fmt="none")
        self.members = [m for m in members if m is not None]
        self.container = container

    # -- identity --------------------------------------------------------
    @property
    def meta(self):
        for obj in (self.main, self.container, *self.members):
            meta = getattr(obj, META_ATTR, None)
            if isinstance(meta, dict):
                return meta
        return {}

    @property
    def label(self):
        if self.container is not None:
            label = self.container.get_label()
            if label and not label.startswith(_AUTO_LABEL_PREFIX):
                return label
        return self.main.get_label() if self.main is not None else (self.container.get_label() or "")

    @property
    def color(self):
        if self.main is not None:
            return self.main.get_color()
        for m in self.members:
            try:
                return m.get_edgecolor()[0] if hasattr(m, "get_edgecolor") else m.get_color()
            except Exception:
                continue
        return "black"

    @property
    def recalled(self):
        return any(getattr(m, "_escape_recalled", False) for m in self.members)

    @property
    def legend_handle(self):
        return self.container if self.container is not None else self.main

    # -- visibility --------------------------------------------------------
    def get_visible(self):
        first = self.main if self.main is not None else (self.members[0] if self.members else None)
        return bool(first.get_visible()) if first is not None else False

    def set_visible(self, state):
        for m in self.members:
            m.set_visible(bool(state))

    def remove(self):
        if self.container is not None:
            try:
                self.container.remove()
                return
            except Exception:
                pass
        for m in self.members:
            try:
                m.remove()
            except Exception:
                pass

    # -- extents, for re-fitting the y-range -------------------------------
    def y_extent(self, xlim=None):
        """(ymin, ymax) of what this item draws, restricted to x in ``xlim``,
        or None."""
        try:
            data = item_arrays(self)
        except Exception:
            return None
        x = data["x"]
        ys = [data["y"]] + [data[k] for k in ("ylo", "yhi") if data.get(k) is not None]
        mask = np.isfinite(x)
        if xlim is not None:
            lo, hi = sorted(xlim)
            mask &= (x >= lo) & (x <= hi)
        values = np.concatenate([np.asarray(y, dtype=float)[mask] for y in ys]) if mask.any() else np.array([])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        return float(values.min()), float(values.max())


def _child_index(ax, artist):
    try:
        return ax.get_children().index(artist)
    except ValueError:
        return 10**9


def collect_items(ax):
    """Every :class:`PlotItem` on ``ax``, in plotting order. Skips escape's
    own overlay artists (``_escape_overlay``) and bands that aren't tagged
    as belonging to a line."""
    items, claimed = [], set()

    for c in ax.containers:
        if not isinstance(c, ErrorbarContainer):
            continue
        data_line, caplines, barlinecols = c.lines
        members = [data_line, *caplines, *barlinecols]
        items.append(PlotItem("errorbar", data_line, members, container=c))
        claimed.update(id(m) for m in members if m is not None)

    groups = {}
    for a in (*ax.lines, *ax.collections, *ax.patches):
        if id(a) in claimed or getattr(a, "_escape_overlay", False):
            continue
        meta = getattr(a, META_ATTR, None)
        group = meta.get("group") if isinstance(meta, dict) else None
        if group:
            groups.setdefault(group, []).append(a)
    for members in groups.values():
        lines = [m for m in members if isinstance(m, Line2D)]
        if not lines:
            continue
        items.append(PlotItem(lines[0].escape_meta.get("kind", "group"), lines[0], members))
        claimed.update(id(m) for m in members)

    for line in ax.lines:
        if id(line) in claimed or getattr(line, "_escape_overlay", False):
            continue
        items.append(PlotItem("line", line, [line]))

    items.sort(key=lambda it: min(_child_index(ax, m) for m in it.members) if it.members else 10**9)
    return items


def data_lines(ax):
    """The ``Line2D`` of each item on ``ax`` -- i.e. ``ax.get_lines()``
    without error-bar caps and escape overlays. What the Fit/Peak/Freq tools
    should consider "the data"."""
    return [it.main for it in collect_items(ax) if it.main is not None]


def all_artists_claimed(ax, items):
    """Whether every visible collection/patch on ``ax`` belongs to one of
    ``items`` (so the y-range can be computed from the items alone)."""
    claimed = {id(m) for it in items for m in it.members}
    return all(id(a) in claimed or not a.get_visible() for a in (*ax.collections, *ax.patches))


def handle_visible(handle):
    """Visibility of a legend handle; containers count as visible if their
    first drawn child is."""
    if isinstance(handle, Container):
        for child in _flatten(list(handle)):
            if hasattr(child, "get_visible"):
                return child.get_visible()
        return True
    return getattr(handle, "get_visible", lambda: True)()


# ---------------------------------------------------------------------------
# turning items into data and back
# ---------------------------------------------------------------------------


def _color(c):
    try:
        return mcolors.to_hex(c, keep_alpha=True)
    except Exception:
        return c if isinstance(c, str) else None


def _line_style(line):
    style = {
        "color": _color(line.get_color()),
        "linestyle": line.get_linestyle(),
        "linewidth": float(line.get_linewidth()),
        "marker": line.get_marker(),
        "markersize": float(line.get_markersize()),
        "markerfacecolor": _color(line.get_markerfacecolor()),
        "markeredgecolor": _color(line.get_markeredgecolor()),
        "markeredgewidth": float(line.get_markeredgewidth()),
        "alpha": line.get_alpha(),
        "drawstyle": line.get_drawstyle(),
        "zorder": float(line.get_zorder()),
    }
    return jsonable(style)


def _errorbar_arrays(container):
    data_line, caplines, barlinecols = container.lines
    cols = list(barlinecols)
    xcol = cols.pop(0) if container.has_xerr and cols else None
    ycol = cols.pop(0) if container.has_yerr and cols else None

    def seg_bounds(col, axis_index):
        segs = col.get_segments()
        lo = np.array([min(s[0][axis_index], s[1][axis_index]) for s in segs], dtype=float)
        hi = np.array([max(s[0][axis_index], s[1][axis_index]) for s in segs], dtype=float)
        pos = np.array([s[0][1 - axis_index] for s in segs], dtype=float)
        return lo, hi, pos

    if data_line is not None:
        x = np.asarray(data_line.get_xdata(orig=False), dtype=float)
        y = np.asarray(data_line.get_ydata(orig=False), dtype=float)
    elif ycol is not None:
        ylo, yhi, x = seg_bounds(ycol, 1)
        y = (ylo + yhi) / 2
    else:
        xlo, xhi, y = seg_bounds(xcol, 0)
        x = (xlo + xhi) / 2

    out = {"x": x, "y": y}
    if ycol is not None:
        lo, hi, _ = seg_bounds(ycol, 1)
        if lo.size == x.size:
            out["ylo"], out["yhi"] = lo, hi
    if xcol is not None:
        lo, hi, _ = seg_bounds(xcol, 0)
        if lo.size == x.size:
            out["xlo"], out["xhi"] = lo, hi
    return out


def item_arrays(item):
    """The numbers needed to redraw ``item``: always ``x`` and ``y``, plus
    ``ylo``/``yhi`` (and ``xlo``/``xhi`` for error bars) where it has them.
    In matplotlib's internal units (dates as day numbers)."""
    if item.kind == "errorbar" and item.container is not None:
        return _errorbar_arrays(item.container)
    x = np.asarray(item.main.get_xdata(orig=False), dtype=float)
    y = np.asarray(item.main.get_ydata(orig=False), dtype=float)
    out = {"x": x, "y": y}
    redraw = getattr(item.main, "_escape_redraw", None)
    if item.kind == "errortube" and redraw and redraw.get("lo") is not None:
        out["ylo"] = np.asarray(redraw["lo"], dtype=float)
        out["yhi"] = np.asarray(redraw["hi"], dtype=float)
    return out


def _is_date_axis(axis):
    get_converter = getattr(axis, "get_converter", None)
    converter = get_converter() if get_converter is not None else getattr(axis, "converter", None)
    if converter is not None and "date" in type(converter).__name__.lower():
        return True
    return isinstance(
        axis.get_major_formatter(), (mdates.AutoDateFormatter, mdates.ConciseDateFormatter, mdates.DateFormatter)
    )


def item_record(item, ax):
    """Everything about ``item`` except its arrays, as JSON-able dict, plus
    the arrays separately: ``(record, arrays)``."""
    arrays = item_arrays(item)
    style = _line_style(item.main) if item.main is not None else {}
    extra = {}
    if item.kind == "errortube":
        redraw = getattr(item.main, "_escape_redraw", None) or {}
        extra = {"step": redraw.get("step"), "falpha": redraw.get("falpha", 0.3)}
    elif item.kind == "errorbar":
        _, caplines, barlinecols = item.container.lines
        if barlinecols:
            col = barlinecols[0]
            try:
                extra["ecolor"] = _color(col.get_colors()[0])
                extra["elinewidth"] = float(np.atleast_1d(col.get_linewidths())[0])
            except Exception:
                pass
        if caplines:
            extra["capsize"] = float(caplines[0].get_markersize()) / 2
            extra["capthick"] = float(caplines[0].get_markeredgewidth())
        extra["fmt_none"] = item.main is None
    x = arrays["x"]
    finite = x[np.isfinite(x)]
    record = {
        "kind": item.kind,
        "label": item.label,
        "meta": jsonable(visible_meta(item.meta)),
        "style": style,
        "extra": jsonable(extra),
        "x_units": "date" if _is_date_axis(ax.xaxis) else None,
        "y_units": "date" if _is_date_axis(ax.yaxis) else None,
        "xrange": [float(finite.min()), float(finite.max())] if finite.size else None,
        "n_points": int(x.size),
    }
    return record, arrays


def _num_to_datetime64(values):
    values = np.asarray(values, dtype=float)
    epoch = np.datetime64(mdates.get_epoch(), "us")
    out = np.full(values.shape, np.datetime64("NaT", "us"))
    ok = np.isfinite(values)
    out[ok] = epoch + (values[ok] * 86400e6).astype("timedelta64[us]")
    return out


def _target_values(values, units, axis):
    """Stored numbers in a form ``axis`` will accept: datetimes for a date
    record unless the axis already holds plain numbers."""
    if units != "date":
        return values
    get_converter = getattr(axis, "get_converter", None)
    converter = get_converter() if get_converter is not None else getattr(axis, "converter", None)
    axis_has_numbers = converter is None and axis.axes.has_data() and not _is_date_axis(axis)
    return values if axis_has_numbers else _num_to_datetime64(values)


_LINE_KWARGS = (
    "color", "linestyle", "linewidth", "marker", "markersize", "markerfacecolor",
    "markeredgecolor", "markeredgewidth", "alpha", "drawstyle", "zorder",
)


def _line_kwargs(style, reference=False):
    kwargs = {k: style[k] for k in _LINE_KWARGS if style.get(k) is not None}
    if reference:
        # dashed, unless it had no connecting line (markers only): then just faded
        if str(kwargs.get("linestyle", "-")).lower() not in ("none", "", " "):
            kwargs["linestyle"] = "--"
        kwargs["alpha"] = 0.5 * (kwargs.get("alpha") or 1.0)
    return kwargs


def redraw(ax, record, arrays, *, label=None, reference=False):
    """Draw a stored item onto ``ax`` again and return the new
    :class:`PlotItem`-able artists (whatever the plot call returned)."""
    kind = record.get("kind", "line")
    style = record.get("style") or {}
    extra = record.get("extra") or {}
    label = record.get("label", "") if label is None else label
    x = _target_values(arrays["x"], record.get("x_units"), ax.xaxis)
    y = _target_values(arrays["y"], record.get("y_units"), ax.yaxis)
    kwargs = _line_kwargs(style, reference)

    if kind == "errortube" and arrays.get("ylo") is not None:
        from escape.plot_utilities import errortube

        y_num = np.asarray(arrays["y"], dtype=float)
        yerr = np.vstack([y_num - arrays["ylo"], arrays["yhi"] - y_num])
        step = extra.get("step")
        if step:
            kwargs.pop("drawstyle", None)
        falpha = float(extra.get("falpha") or 0.3) * (0.5 if reference else 1.0)
        return errortube(x, y, yerr=yerr, axis=ax, falpha=falpha, step=step, label=label, **kwargs)

    if kind == "errorbar":
        y_num = np.asarray(arrays["y"], dtype=float)
        x_num = np.asarray(arrays["x"], dtype=float)
        eb_kwargs = {}
        if arrays.get("ylo") is not None:
            eb_kwargs["yerr"] = np.clip(np.vstack([y_num - arrays["ylo"], arrays["yhi"] - y_num]), 0, None)
        if arrays.get("xlo") is not None:
            eb_kwargs["xerr"] = np.clip(np.vstack([x_num - arrays["xlo"], arrays["xhi"] - x_num]), 0, None)
        for key in ("ecolor", "elinewidth", "capsize", "capthick"):
            if extra.get(key) is not None:
                eb_kwargs[key] = extra[key]
        if extra.get("fmt_none"):
            eb_kwargs["fmt"] = "none"
            kwargs = {k: v for k, v in kwargs.items() if k in ("color", "alpha", "zorder")}
        return ax.errorbar(x, y, label=label, **eb_kwargs, **kwargs)

    return ax.plot(x, y, label=label, **kwargs)


def dumps(obj):
    return json.dumps(jsonable(obj))
