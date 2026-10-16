"""Saved plot curves ("plot history") behind escape's History toolbar button
(:func:`escape.plot_utilities.attach_history_button`).

Each click on *Save* in the History window writes one small HDF5 file
holding the ticked curves: their numbers, their look (colour, line style,
markers, ...), their label and ``escape_meta`` information (see
:func:`escape.plot_utilities.tag`), plus where they came from (figure and
axes titles, axis labels and ranges) and who saved them when, with an
optional note. *Recall* lists those curves again, best matches for the
current plot first, and draws chosen ones back onto any plot.

Where files go, first match wins:

1. the ``directory=`` argument of the functions here;
2. :data:`PLOT_HISTORY_DIR`, if you set it, e.g.
   ``escape.plot_history.PLOT_HISTORY_DIR = "/sf/bernina/data/p12345/res/plot_history"``
   (a shared folder lets colleagues recall each other's curves; every entry
   records who saved it);
3. the ``ESCAPE_PLOT_HISTORY_DIR`` environment variable;
4. ``~/.escape/plot_history``.

The files are plain HDF5 (``h5py``): one group per curve, its description
as JSON in the ``record`` attribute, its numbers as datasets ``x``, ``y``
and, for error bands/bars, ``ylo``/``yhi`` (``xlo``/``xhi``).
"""

from __future__ import annotations

import getpass
import json
import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from escape._plot_items import collect_items, dumps, item_record, jsonable, visible_meta

PLOT_HISTORY_DIR = None
"""Folder for saved curves; ``None`` (default): the ``ESCAPE_PLOT_HISTORY_DIR``
environment variable if set, else ``~/.escape/plot_history``."""

_FORMAT = "escape-plot-history"
_VERSION = 1


def history_dir(directory=None):
    """The folder saved curves go to / are read from (see the module docstring)."""
    for candidate in (directory, PLOT_HISTORY_DIR, os.environ.get("ESCAPE_PLOT_HISTORY_DIR")):
        if candidate:
            return Path(candidate).expanduser()
    return Path.home() / ".escape" / "plot_history"


def _user():
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def axes_context(ax):
    """What identifies the plot ``ax`` belongs to: figure name/title, axes
    title, axis labels, ranges and scales."""
    fig = ax.figure
    suptitle = getattr(fig, "_suptitle", None)
    try:
        window_title = fig.canvas.manager.get_window_title()
    except Exception:
        window_title = ""
    return jsonable(
        {
            "figure_label": fig.get_label() or "",
            "figure_title": suptitle.get_text() if suptitle is not None else "",
            "window_title": window_title or "",
            "axes_title": ax.get_title() or "",
            "xlabel": ax.get_xlabel() or "",
            "ylabel": ax.get_ylabel() or "",
            "xlim": [float(v) for v in ax.get_xlim()],
            "ylim": [float(v) for v in ax.get_ylim()],
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
        }
    )


@dataclass
class HistoryEntry:
    """One saved curve, as listed (its numbers are read only on :meth:`load`)."""

    path: Path
    group: str
    saved_at: datetime
    user: str
    host: str
    note: str
    context: dict
    record: dict
    _arrays: dict = field(default=None, repr=False)

    @property
    def label(self):
        return self.record.get("label", "")

    @property
    def kind(self):
        return self.record.get("kind", "line")

    @property
    def meta(self):
        return self.record.get("meta") or {}

    @property
    def save_id(self):
        """Curves saved with one click share this."""
        return str(self.path)

    def load(self):
        """The stored numbers: dict of arrays (``x``, ``y``, maybe ``ylo``...)."""
        if self._arrays is None:
            import h5py

            with h5py.File(self.path, "r") as f:
                g = f[self.group]
                self._arrays = {k: np.asarray(g[k][()]) for k in g.keys()}
        return self._arrays

    def search_text(self):
        """Everything the Recall search box looks through, lower-case."""
        parts = [
            self.label, self.kind, self.note, self.user, self.host,
            self.saved_at.strftime("%Y-%m-%d %H:%M"),
            *[str(v) for v in self.context.values() if isinstance(v, str)],
            json.dumps(self.meta),
        ]
        return " ".join(parts).lower()


def save_items(items, ax, note="", directory=None):
    """Save plot ``items`` (from :func:`escape._plot_items.collect_items`,
    all on ``ax``) as one history file; returns its path."""
    import h5py

    items = list(items)
    if not items:
        raise ValueError("nothing to save")
    folder = history_dir(directory)
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    user = _user()
    name = f"{now.strftime('%Y%m%d-%H%M%S-%f')}_{user}.h5"
    final = folder / name
    tmp = folder / (name + ".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = _FORMAT
            f.attrs["version"] = _VERSION
            f.attrs["saved_at"] = now.isoformat(timespec="seconds")
            f.attrs["user"] = user
            f.attrs["host"] = socket.gethostname()
            f.attrs["note"] = note or ""
            f.attrs["context"] = dumps(axes_context(ax))
            for i, item in enumerate(items):
                record, arrays = item_record(item, ax)
                g = f.create_group(f"item_{i:03d}")
                g.attrs["record"] = dumps(record)
                for key, values in arrays.items():
                    values = np.asarray(values, dtype=float)
                    opts = {"compression": "gzip"} if values.size > 1000 else {}
                    g.create_dataset(key, data=values, **opts)
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()
    return final


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


def list_entries(directory=None):
    """Every saved curve in the history folder, newest first. Reads only
    each file's descriptions, not the numbers. Unreadable files are skipped
    with a printed note."""
    import h5py

    folder = history_dir(directory)
    if not folder.is_dir():
        return []
    entries = []
    for path in sorted(folder.glob("*.h5")):
        try:
            with h5py.File(path, "r") as f:
                if _decode(f.attrs.get("format", "")) != _FORMAT:
                    continue
                saved_at = datetime.fromisoformat(_decode(f.attrs["saved_at"]))
                common = dict(
                    saved_at=saved_at,
                    user=_decode(f.attrs.get("user", "")),
                    host=_decode(f.attrs.get("host", "")),
                    note=_decode(f.attrs.get("note", "")),
                    context=json.loads(_decode(f.attrs.get("context", "{}"))),
                )
                for name in sorted(f.keys()):
                    record = json.loads(_decode(f[name].attrs["record"]))
                    entries.append(HistoryEntry(path=path, group=name, record=record, **common))
        except Exception as e:
            print(f"[escape] skipping unreadable plot-history file {path.name}: {e}")
    entries.sort(key=lambda e: (e.saved_at, e.group), reverse=True)
    return entries


def delete_entries(entries):
    """Delete saved curves; a file whose curves are all deleted is removed."""
    import h5py

    by_path = {}
    for e in entries:
        by_path.setdefault(Path(e.path), []).append(e.group)
    for path, groups in by_path.items():
        with h5py.File(path, "a") as f:
            for g in groups:
                if g in f:
                    del f[g]
            empty = len(f.keys()) == 0
        if empty:
            path.unlink()


# ---------------------------------------------------------------------------
# matching against the plot it would be recalled into
# ---------------------------------------------------------------------------


def _norm(text):
    return " ".join(str(text or "").lower().split())


def _meaningful_figure_label(label):
    # nfigure's automatic per-cell names and matplotlib's defaults say
    # nothing about what's plotted
    label = _norm(label)
    return "" if (not label or label.startswith("cell-") or label.startswith("figure") or label == "no name") else label


class Matcher:
    """Scores saved entries against one target axes: 0-3 stars."""

    def __init__(self, ax):
        self.context = axes_context(ax)
        self.has_data = ax.has_data()
        items = collect_items(ax)
        self.labels = {_norm(it.label) for it in items if not it.label.startswith("_")}
        metas = [visible_meta(it.meta) for it in items]
        self.arrays = {_norm(m["array"]) for m in metas if m.get("array")}
        self.scan_parameters = {_norm(m["scan_parameter"]) for m in metas if m.get("scan_parameter")}

    def score(self, entry):
        """(points, possible points) for ``entry``."""
        ctx, here = entry.context, self.context
        points = possible = 0

        def check(condition, applicable=True):
            nonlocal points, possible
            if applicable:
                possible += 1
                points += bool(condition)

        for key in ("figure_title", "axes_title", "xlabel", "ylabel"):
            mine = _norm(here.get(key))
            check(mine and mine == _norm(ctx.get(key)), applicable=bool(mine))
        mine = _meaningful_figure_label(here.get("figure_label"))
        check(mine and mine == _meaningful_figure_label(ctx.get("figure_label")), applicable=bool(mine))
        xrange = entry.record.get("xrange")
        if self.has_data and xrange:
            lo, hi = sorted(here["xlim"])
            check(xrange[0] <= hi and xrange[1] >= lo)
        check(_norm(entry.label) in self.labels, applicable=bool(self.labels))
        meta = entry.meta
        check(_norm(meta.get("array")) in self.arrays, applicable=bool(self.arrays))
        check(_norm(meta.get("scan_parameter")) in self.scan_parameters, applicable=bool(self.scan_parameters))
        return points, possible

    def stars(self, entry):
        points, possible = self.score(entry)
        if points == 0 or possible == 0:
            return 0
        return max(1, round(3 * points / possible))

    def x_overlaps(self, entry):
        xrange = entry.record.get("xrange")
        if not xrange or not self.has_data:
            return True
        lo, hi = sorted(self.context["xlim"])
        return xrange[0] <= hi and xrange[1] >= lo


def parse_date(text, end=False):
    """``YYYY-MM-DD`` (optionally ``HH:MM``) as datetime, or None if empty or
    unreadable. ``end=True`` makes a bare date mean the end of that day."""
    text = (text or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y%m%d"):
        try:
            value = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if end and fmt != "%Y-%m-%d %H:%M":
            value = value.replace(hour=23, minute=59, second=59)
        return value
    return None


def filter_entries(entries, search="", date_from=None, date_to=None):
    """``entries`` whose text contains every word of ``search`` and whose
    save time is within ``[date_from, date_to]`` (either may be None)."""
    words = _norm(search).split()
    out = []
    for e in entries:
        if date_from is not None and e.saved_at < date_from:
            continue
        if date_to is not None and e.saved_at > date_to:
            continue
        if words:
            text = e.search_text()
            if not all(w in text for w in words):
                continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# drawing saved curves back
# ---------------------------------------------------------------------------


def recalled_label(entry):
    """Legend label of a recalled curve: its own label plus the save time."""
    label = entry.label
    if not label or label.startswith("_"):
        label = entry.kind
    return f"{label} ({entry.saved_at.strftime('%m-%d %H:%M')})"


def _forget_select_panel(ax):
    # an open Select panel lists the curves present when it was opened;
    # drop it so the next click on Select lists the new set
    gui = getattr(ax, "_escape_select_gui", None)
    if gui is not None:
        try:
            gui.close()
        except Exception:
            pass
        ax._escape_select_gui = None


def _after_change(ax, create_legend):
    from escape.select_gui import rebuild_legend

    # an existing legend is rebuilt with its own look; with none, either
    # create a default one (after recalling: the save-time labels are how
    # recalled curves are told apart) or leave it that way
    rebuild_legend(ax, {} if create_legend else None)
    ax.relim(visible_only=True)
    ax.autoscale_view()
    _forget_select_panel(ax)
    ax.figure.canvas.draw_idle()


def recall(ax, entries, reference=False):
    """Draw saved ``entries`` onto ``ax``, each with its saved look (or, with
    ``reference=True``, dashed and half transparent), its ``escape_meta``
    restored plus where/when it was recalled from, and the save time in its
    legend label. A legend is created if the axes has none. Returns
    ``(drawn, outside)``: what each plot call returned, and the labels of
    curves whose x-range doesn't overlap the current view."""
    from escape._plot_items import _flatten, tag

    matcher = Matcher(ax)
    drawn, outside = [], []
    for entry in entries:
        result = _redraw_entry(ax, entry, reference)
        meta = dict(entry.meta)
        meta.update(
            recalled_from=str(entry.path),
            recalled_item=entry.group,
            saved_at=entry.saved_at.isoformat(timespec="seconds"),
            saved_by=entry.user,
        )
        if entry.note:
            meta.setdefault("note", entry.note)
        tag(result, **meta)
        for artist in _flatten(result):
            try:
                artist._escape_recalled = True
            except Exception:
                pass
        drawn.append(result)
        if not matcher.x_overlaps(entry):
            outside.append(recalled_label(entry))
    _after_change(ax, create_legend=True)
    return drawn, outside


def _redraw_entry(ax, entry, reference):
    from escape._plot_items import redraw

    return redraw(ax, entry.record, entry.load(), label=recalled_label(entry), reference=reference)


def remove_recalled(ax):
    """Remove every recalled curve from ``ax``; returns how many."""
    removed = 0
    for item in collect_items(ax):
        if item.recalled:
            item.remove()
            removed += 1
    if removed:
        _after_change(ax, create_legend=False)
    return removed
