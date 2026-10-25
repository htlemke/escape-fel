"""The window behind escape's History toolbar button
(:func:`escape.plot_utilities.attach_history_button`): a Qt dialog
(:func:`make_qt_history_dialog`) or, for ipympl figures, an ipywidgets
panel below the figure (:func:`make_ipywidgets_history_panel`).

Two tabs:

* **Save** -- tick curves of the current plot, add an optional note, save.
* **Recall** -- every saved curve, best matches for the current plot first;
  search (all words must match: labels, notes, user, figure/axis titles,
  ``escape_meta`` info), optional date range, "only entries matching this
  plot"; *Add to plot* draws the selected ones (original look or a dashed
  "reference" look), *Remove recalled curves* takes them off again,
  *Delete selected...* deletes saved entries.

The storage, matching and redrawing live in :mod:`escape.plot_history`;
this module is only the window. ``qtpy``/``ipywidgets`` are imported
inside the two builders, so importing this module is free.
"""

from __future__ import annotations

from escape import plot_history as ph
from escape._plot_items import collect_items, visible_meta
from escape.select_gui import _display_label, _swatch_icon

_HIDDEN_META = ("recalled_from", "recalled_item")


def _qt(owner, *path):
    """An enum value by its PyQt6-style scoped path, falling back to the
    PyQt5-style short name (``_qt(Qt, "ItemDataRole", "UserRole")``)."""
    obj = owner
    try:
        for name in path:
            obj = getattr(obj, name)
        return obj
    except AttributeError:
        return getattr(owner, path[-1])


def _stars(n):
    return "★" * n + "☆" * (3 - n) if n else ""


def _plot_text(context):
    """Figure title (or a meaningful figure/window name) and axes title."""
    name = context.get("figure_title") or ""
    if not name:
        for key in ("figure_label", "window_title"):
            if ph._meaningful_figure_label(context.get(key)):
                name = context[key]
                break
    return " — ".join(p for p in (name, context.get("axes_title") or "") if p)


def _axes_text(context):
    x, y = context.get("xlabel") or "", context.get("ylabel") or ""
    return f"{x} / {y}" if (x or y) else ""


def _info_text(meta):
    meta = {k: v for k, v in visible_meta(meta).items() if k not in _HIDDEN_META}
    return ", ".join(f"{k}={v}" for k, v in meta.items())


COLUMNS = ["★", "saved", "curve", "type", "plot", "x / y label", "note", "user", "info"]


def _entry_cells(stars, e):
    """The table cells for one saved curve, in :data:`COLUMNS` order."""
    return [
        _stars(stars),
        e.saved_at.strftime("%Y-%m-%d %H:%M"),
        e.label,
        e.kind,
        _plot_text(e.context),
        _axes_text(e.context),
        e.note,
        e.user,
        _info_text(e.meta),
    ]


def _recall_rows(entries, matcher, search, from_text, to_text, only_matching):
    """Filtered, scored and sorted rows for the Recall list, shared by both
    windows: ``(rows, status, bad_dates)`` with ``rows`` a list of
    ``(stars, entry)``."""
    date_from = ph.parse_date(from_text)
    date_to = ph.parse_date(to_text, end=True)
    bad = [name for name, text, value in (("from", from_text, date_from), ("to", to_text, date_to))
           if (text or "").strip() and value is None]
    candidates = ph.filter_entries(entries, search, date_from, date_to)
    scored = [(matcher.stars(e), e) for e in candidates]
    if only_matching:
        scored = [(st, e) for st, e in scored if st > 0]
    scored.sort(key=lambda se: (se[0], se[1].saved_at), reverse=True)
    total = len(entries)
    if not total:
        status = "Nothing saved yet -- use the Save tab."
    elif not scored and only_matching:
        status = f"No saved curve matches this plot -- untick “only entries matching this plot” to see all {total}."
    else:
        status = f"{len(scored)} of {total} saved curves shown."
    if bad:
        status += f" (ignored unreadable {' and '.join(bad)} date: use YYYY-MM-DD)"
    return scored, status, bad


def make_qt_history_dialog(fig, ax):
    """Build, show and return the History window for ``ax`` -- a non-modal
    child of ``fig``'s window. Cached on the axes by the caller
    (:func:`escape.plot_utilities._run_history_button`); showing it again
    re-reads both the plot and the history folder."""
    from qtpy import QtCore, QtGui, QtWidgets

    USER_ROLE = _qt(QtCore.Qt, "ItemDataRole", "UserRole")

    class HistoryDialog(QtWidgets.QDialog):
        def __init__(self):
            super().__init__(fig.canvas.manager.window)
            self.fig, self.ax = fig, ax
            self.setWindowTitle("History")
            self.resize(980, 560)
            # replaceable (tests): returns True to go ahead
            self.confirm = self._ask
            self._items, self._boxes = [], []
            self._entries, self._shown = [], []

            layout = QtWidgets.QVBoxLayout(self)
            self.tabs = QtWidgets.QTabWidget()
            layout.addWidget(self.tabs)
            self.tabs.addTab(self._build_save_tab(), "Save")
            self.tabs.addTab(self._build_recall_tab(), "Recall")
            fig.canvas.mpl_connect("close_event", self._on_fig_close)
            self.refresh()

        # ------------------------------------------------------------------
        @staticmethod
        def guarded(fn):
            # PyQt5 aborts the process if an exception escapes a slot
            def run(*_args):
                try:
                    fn()
                except Exception as e:
                    print(f"[escape] history: {e}")

            return run

        def _ask(self, text):
            yes = _qt(QtWidgets.QMessageBox, "StandardButton", "Yes")
            no = _qt(QtWidgets.QMessageBox, "StandardButton", "No")
            return QtWidgets.QMessageBox.question(self, "History", text, yes | no, no) == yes

        def _on_fig_close(self, _event=None):
            try:
                self.close()
            except RuntimeError:
                pass

        def showEvent(self, event):
            super().showEvent(event)
            self.guarded(self.refresh)()  # a Qt virtual: an exception here aborts too

        def refresh(self):
            self.refresh_save_list()
            self.refresh_recall()

        # ------------------------------------------------------------------
        # Save tab
        def _build_save_tab(self):
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)
            self.save_filter = QtWidgets.QLineEdit()
            self.save_filter.setPlaceholderText("filter curves, e.g. readback")
            self.save_filter.setClearButtonEnabled(True)
            self.save_filter.textChanged.connect(self.guarded(self._refilter_save))
            v.addWidget(self.save_filter)

            self._save_holder = QtWidgets.QWidget()
            self._save_rows = QtWidgets.QVBoxLayout(self._save_holder)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self._save_holder)
            v.addWidget(scroll, 1)

            row = QtWidgets.QHBoxLayout()
            for text, state in (("All", True), ("None", False)):
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(self.guarded(lambda state=state: self._set_save_ticks(state)))
                row.addWidget(b)
            row.addStretch(1)
            v.addLayout(row)

            note_row = QtWidgets.QHBoxLayout()
            note_row.addWidget(QtWidgets.QLabel("note:"))
            self.note_edit = QtWidgets.QLineEdit()
            self.note_edit.setPlaceholderText("optional, e.g. after realignment (searchable later)")
            note_row.addWidget(self.note_edit, 1)
            self.save_button = QtWidgets.QPushButton("Save")
            self.save_button.clicked.connect(self.guarded(self.save_ticked))
            note_row.addWidget(self.save_button)
            v.addLayout(note_row)

            self.save_status = QtWidgets.QLabel("")
            self.save_status.setWordWrap(True)
            v.addWidget(self.save_status)
            return page

        def refresh_save_list(self):
            previous = {id(it.main): box.isChecked() for it, box in zip(self._items, self._boxes)}
            for box in self._boxes:
                box.setParent(None)
                box.deleteLater()
            while self._save_rows.count():
                self._save_rows.takeAt(0)
            self._items = collect_items(self.ax)
            self._boxes = []
            for i, item in enumerate(self._items):
                box = QtWidgets.QCheckBox(_display_label(item, i, with_kind=True))
                box.setIcon(_swatch_icon(QtGui, item.color))
                box.setIconSize(QtCore.QSize(28, 12))
                box.setChecked(previous.get(id(item.main), not item.recalled))
                box.toggled.connect(self.guarded(self._update_save_button))
                self._save_rows.addWidget(box)
                self._boxes.append(box)
            self._save_rows.addStretch(1)
            self._refilter_save()
            self._update_save_button()

        def _refilter_save(self):
            needle = self.save_filter.text().strip().lower()
            for box in self._boxes:
                box.setVisible(needle in box.text().lower())

        def _set_save_ticks(self, state):
            for box in self._boxes:
                if not box.isHidden():
                    box.setChecked(state)

        def _ticked(self):
            return [it for it, box in zip(self._items, self._boxes) if box.isChecked()]

        def _update_save_button(self):
            n = len(self._ticked())
            self.save_button.setText(f"Save {n} curve{'s' if n != 1 else ''}")
            self.save_button.setEnabled(n > 0)

        def save_ticked(self):
            items = self._ticked()
            if not items:
                return
            path = ph.save_items(items, self.ax, note=self.note_edit.text().strip())
            self.save_status.setText(f"Saved {len(items)} curve{'s' if len(items) != 1 else ''} to {path}")
            self.refresh_recall()

        # ------------------------------------------------------------------
        # Recall tab
        def _build_recall_tab(self):
            page = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(page)

            row = QtWidgets.QHBoxLayout()
            self.search_edit = QtWidgets.QLineEdit()
            self.search_edit.setPlaceholderText("search: label, note, user, titles, run, ... (all words must match)")
            self.search_edit.setClearButtonEnabled(True)
            row.addWidget(self.search_edit, 3)
            self.from_edit = QtWidgets.QLineEdit()
            self.from_edit.setPlaceholderText("from YYYY-MM-DD")
            self.to_edit = QtWidgets.QLineEdit()
            self.to_edit.setPlaceholderText("to YYYY-MM-DD")
            for w in (self.from_edit, self.to_edit):
                w.setClearButtonEnabled(True)
                row.addWidget(w, 1)
            v.addLayout(row)

            row = QtWidgets.QHBoxLayout()
            self.only_matching = QtWidgets.QCheckBox("only entries matching this plot")
            self.only_matching.setChecked(True)
            row.addWidget(self.only_matching)
            row.addStretch(1)
            refresh = QtWidgets.QPushButton("Refresh")
            refresh.setToolTip("Re-read the history folder")
            refresh.clicked.connect(self.guarded(self.refresh))
            row.addWidget(refresh)
            v.addLayout(row)

            self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
            self.table.setHorizontalHeaderLabels(COLUMNS)
            self.table.setSelectionBehavior(_qt(QtWidgets.QAbstractItemView, "SelectionBehavior", "SelectRows"))
            self.table.setSelectionMode(_qt(QtWidgets.QAbstractItemView, "SelectionMode", "ExtendedSelection"))
            self.table.setEditTriggers(_qt(QtWidgets.QAbstractItemView, "EditTrigger", "NoEditTriggers"))
            self.table.verticalHeader().setVisible(False)
            header = self.table.horizontalHeader()
            header.setStretchLastSection(True)
            header.setSectionResizeMode(_qt(QtWidgets.QHeaderView, "ResizeMode", "Interactive"))
            self.table.itemDoubleClicked.connect(self.guarded(self.add_selected))
            v.addWidget(self.table, 1)

            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("look:"))
            self.look_original = QtWidgets.QRadioButton("original")
            self.look_reference = QtWidgets.QRadioButton("reference (dashed, faded)")
            self.look_original.setChecked(True)
            row.addWidget(self.look_original)
            row.addWidget(self.look_reference)
            row.addStretch(1)
            self.add_button = QtWidgets.QPushButton("Add to plot")
            self.add_button.clicked.connect(self.guarded(self.add_selected))
            row.addWidget(self.add_button)
            remove = QtWidgets.QPushButton("Remove recalled curves")
            remove.clicked.connect(self.guarded(self.remove_recalled))
            row.addWidget(remove)
            delete = QtWidgets.QPushButton("Delete selected…")
            delete.clicked.connect(self.guarded(self.delete_selected))
            row.addWidget(delete)
            v.addLayout(row)

            self.recall_status = QtWidgets.QLabel("")
            self.recall_status.setWordWrap(True)
            v.addWidget(self.recall_status)
            self.folder_label = QtWidgets.QLabel("")
            self.folder_label.setToolTip(
                "Set escape.plot_history.PLOT_HISTORY_DIR (or the ESCAPE_PLOT_HISTORY_DIR\n"
                "environment variable) to use another folder, e.g. a shared one."
            )
            v.addWidget(self.folder_label)

            for w in (self.search_edit, self.from_edit, self.to_edit):
                w.textChanged.connect(self.guarded(self.refill_table))
            self.only_matching.toggled.connect(self.guarded(self.refill_table))
            return page

        def refresh_recall(self):
            self._entries = ph.list_entries()
            self.folder_label.setText(f"folder: {ph.history_dir()}")
            self.refill_table()

        def refill_table(self):
            scored, status, bad = _recall_rows(
                self._entries, ph.Matcher(self.ax), self.search_edit.text(),
                self.from_edit.text(), self.to_edit.text(), self.only_matching.isChecked(),
            )
            self.from_edit.setStyleSheet("color: crimson;" if "from" in bad else "")
            self.to_edit.setStyleSheet("color: crimson;" if "to" in bad else "")
            self._shown = [e for _, e in scored]
            self.table.setRowCount(len(scored))
            for row, (stars, e) in enumerate(scored):
                tip = f"{e.path.name} / {e.group}\nsaved {e.saved_at:%Y-%m-%d %H:%M:%S} by {e.user} on {e.host}"
                for col, text in enumerate(_entry_cells(stars, e)):
                    cell = QtWidgets.QTableWidgetItem(str(text))
                    cell.setData(USER_ROLE, row)
                    cell.setToolTip(tip)
                    self.table.setItem(row, col, cell)
            self.table.resizeColumnsToContents()
            self.recall_status.setText(status)

        def selected_entries(self):
            rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
            return [self._shown[r] for r in rows if r < len(self._shown)]

        def add_selected(self):
            entries = self.selected_entries()
            if not entries:
                self.recall_status.setText("Select one or more rows first.")
                return
            _, outside = ph.recall(self.ax, entries, reference=self.look_reference.isChecked())
            text = f"Added {len(entries)} curve{'s' if len(entries) != 1 else ''}."
            if outside:
                text += " Outside the current x-range (zoom out / Home to see): " + ", ".join(outside)
            self.refresh_save_list()
            self.refill_table()
            self.recall_status.setText(text)

        def remove_recalled(self):
            n = ph.remove_recalled(self.ax)
            self.refresh_save_list()
            self.refill_table()
            self.recall_status.setText(f"Removed {n} recalled curve{'s' if n != 1 else ''}.")

        def delete_selected(self):
            entries = self.selected_entries()
            if not entries:
                self.recall_status.setText("Select one or more rows first.")
                return
            n = len(entries)
            if not self.confirm(f"Delete {n} saved curve{'s' if n != 1 else ''}? This can't be undone."):
                return
            ph.delete_entries(entries)
            self.refresh_recall()
            self.recall_status.setText(f"Deleted {n} saved curve{'s' if n != 1 else ''}.")

    dialog = HistoryDialog()
    dialog.show()
    return dialog


# ---------------------------------------------------------------------------
# ipywidgets panel (ipympl, ``%matplotlib widget``)
# ---------------------------------------------------------------------------

# column widths (characters) for the monospace Recall list
_WIDTHS = [4, 17, 24, 10, 32, 26, 28, 10, 40]
_ASCII_STARS = {0: "", 1: "*", 2: "**", 3: "***"}


def _fixed(text, width):
    text = " ".join(str(text).split())
    text = text[: width - 2] + "… " if len(text) > width - 1 else text.ljust(width)
    return text.replace(" ", " ")  # a <select> collapses plain spaces


def _row_text(stars, e):
    cells = _entry_cells(stars, e)
    cells[0] = _ASCII_STARS.get(stars, "")  # ★ is often not monospace
    return "".join(_fixed(c, w) for c, w in zip(cells, _WIDTHS))


def make_ipywidgets_history_panel(fig, ax):
    """The History window as an ipywidgets panel, for ipympl
    (``%matplotlib widget``) figures: same Save/Recall tabs and actions as
    :func:`make_qt_history_dialog`, driving the same
    :mod:`escape.plot_history` functions. Displayed on construction;
    :meth:`refresh` re-reads the plot and the history folder (the toolbar
    button calls it when clicked again)."""
    import ipywidgets as widgets
    from IPython.display import display

    from escape.plot_utilities import _make_resizable
    from escape.select_gui import _swatch_html

    Layout = widgets.Layout

    class HistoryPanel(widgets.VBox):
        def __init__(self):
            self.fig, self.ax = fig, ax
            self._items, self._checks, self._rows = [], [], []
            self._entries, self._shown = [], []
            self._pending_delete = None

            # -- Save tab
            self.save_filter = widgets.Text(placeholder="filter curves, e.g. readback", layout=Layout(width="98%"))
            self.save_list = widgets.VBox(layout=Layout(max_height="260px", overflow="auto"))
            all_btn = widgets.Button(description="All", layout=Layout(width="70px"))
            none_btn = widgets.Button(description="None", layout=Layout(width="70px"))
            self.note = widgets.Text(
                description="note:", placeholder="optional, e.g. after realignment (searchable later)",
                layout=Layout(width="70%"),
            )
            self.save_button = widgets.Button(description="Save", button_style="primary")
            self.save_status = widgets.HTML()
            save_page = widgets.VBox([
                self.save_filter, self.save_list, widgets.HBox([all_btn, none_btn]),
                widgets.HBox([self.note, self.save_button]), self.save_status,
            ])

            # -- Recall tab
            self.search = widgets.Text(
                placeholder="search: label, note, user, titles, run, ... (all words must match)",
                layout=Layout(width="55%"),
            )
            self.date_from = widgets.Text(placeholder="from YYYY-MM-DD", layout=Layout(width="20%"))
            self.date_to = widgets.Text(placeholder="to YYYY-MM-DD", layout=Layout(width="20%"))
            self.only_matching = widgets.Checkbox(value=True, description="only entries matching this plot", indent=False)
            refresh_btn = widgets.Button(description="Refresh", tooltip="Re-read the history folder")
            self.header = widgets.HTML()
            self.header.add_class("escape-history-mono")
            self.rows = widgets.SelectMultiple(options=[], rows=12, layout=Layout(width="98%"))
            self.rows.add_class("escape-history-mono")
            self.look = widgets.ToggleButtons(
                options=[("original", "original"), ("reference (dashed, faded)", "reference")], value="original",
            )
            self.add_button = widgets.Button(description="Add to plot", button_style="success")
            remove_btn = widgets.Button(description="Remove recalled curves", layout=Layout(width="auto"))
            self.delete_button = widgets.Button(description="Delete selected…")
            self.recall_status = widgets.HTML()
            self.folder_label = widgets.HTML()
            recall_page = widgets.VBox([
                widgets.HBox([self.search, self.date_from, self.date_to]),
                widgets.HBox([self.only_matching, refresh_btn]),
                self.header, self.rows,
                widgets.HBox([widgets.Label("look:"), self.look]),
                widgets.HBox([self.add_button, remove_btn, self.delete_button]),
                self.recall_status, self.folder_label,
            ])

            self.tabs = widgets.Tab(children=[save_page, recall_page])
            for i, title in enumerate(("Save", "Recall")):
                self.tabs.set_title(i, title)
            css = widgets.HTML(
                "<style>.escape-history-mono, .escape-history-mono select, .escape-history-mono option"
                "{font-family:monospace !important;font-size:12px;}</style>"
            )
            super().__init__(
                [css, widgets.HTML("<b>History</b>"), self.tabs],
                layout=Layout(border="solid 1px #ccc", padding="6px", width="980px"),
            )

            g = self._guarded
            self.save_filter.observe(g(lambda ch: self._refilter_save()), names="value")
            all_btn.on_click(g(lambda b: self._set_ticks(True)))
            none_btn.on_click(g(lambda b: self._set_ticks(False)))
            self.save_button.on_click(g(lambda b: self.save_ticked()))
            for w in (self.search, self.date_from, self.date_to, self.only_matching):
                w.observe(g(lambda ch: self.refill_table()), names="value")
            self.rows.observe(g(lambda ch: self._reset_delete()), names="value")
            refresh_btn.on_click(g(lambda b: self.refresh()))
            self.add_button.on_click(g(lambda b: self.add_selected()))
            remove_btn.on_click(g(lambda b: self.remove_recalled()))
            self.delete_button.on_click(g(lambda b: self.delete_selected()))

            _make_resizable(self)
            self.refresh()
            display(self)

        # ------------------------------------------------------------------
        def _guarded(self, fn):
            # a widget callback runs while handling a comm message: an
            # exception or print there is invisible, so show it in the panel
            def run(*args):
                try:
                    fn(*args)
                except Exception as e:
                    print(f"[escape] history: {e}")
                    status = self.recall_status if self.tabs.selected_index == 1 else self.save_status
                    status.value = f"<span style='color:crimson'>error: {e}</span>"

            return run

        def refresh(self):
            self.refresh_save_list()
            self.refresh_recall()

        # -- Save tab --------------------------------------------------------
        def refresh_save_list(self):
            previous = {id(it.main): cb.value for it, cb in zip(self._items, self._checks)}
            self._items = collect_items(self.ax)
            self._checks, self._rows = [], []
            for i, item in enumerate(self._items):
                cb = widgets.Checkbox(
                    value=previous.get(id(item.main), not item.recalled),
                    description=_display_label(item, i, with_kind=True), indent=False,
                    layout=Layout(width="auto"),
                )
                cb.observe(self._guarded(lambda ch: self._update_save_button()), names="value")
                self._checks.append(cb)
                self._rows.append(widgets.HBox([widgets.HTML(_swatch_html(item.color)), cb]))
            self.save_list.children = self._rows
            self._refilter_save()
            self._update_save_button()

        def _refilter_save(self):
            needle = self.save_filter.value.strip().lower()
            for row, cb in zip(self._rows, self._checks):
                row.layout.display = "" if needle in cb.description.lower() else "none"

        def _set_ticks(self, state):
            for row, cb in zip(self._rows, self._checks):
                if row.layout.display != "none":
                    cb.value = state

        def _ticked(self):
            return [it for it, cb in zip(self._items, self._checks) if cb.value]

        def _update_save_button(self):
            n = len(self._ticked())
            self.save_button.description = f"Save {n} curve{'s' if n != 1 else ''}"
            self.save_button.disabled = n == 0

        def save_ticked(self):
            items = self._ticked()
            if not items:
                return
            path = ph.save_items(items, self.ax, note=self.note.value.strip())
            self.save_status.value = f"Saved {len(items)} curve{'s' if len(items) != 1 else ''} to <code>{path}</code>"
            self.refresh_recall()

        # -- Recall tab ------------------------------------------------------
        def refresh_recall(self):
            self._entries = ph.list_entries()
            self.folder_label.value = (
                f"<small>folder: <code>{ph.history_dir()}</code> -- set "
                "<code>escape.plot_history.PLOT_HISTORY_DIR</code> to use another</small>"
            )
            self.refill_table()

        def refill_table(self):
            scored, status, _bad = _recall_rows(
                self._entries, ph.Matcher(self.ax), self.search.value,
                self.date_from.value, self.date_to.value, self.only_matching.value,
            )
            self._shown = [e for _, e in scored]
            head = "".join(_fixed(c, w) for c, w in zip(COLUMNS, _WIDTHS)).replace("★", "*", 1)
            self.header.value = f"<div style='white-space:pre;font-weight:bold'>{head}</div>"
            self.rows.value = ()
            self.rows.options = [(_row_text(st, e), i) for i, (st, e) in enumerate(scored)]
            self._reset_delete()
            self.recall_status.value = status

        def selected_entries(self):
            return [self._shown[i] for i in self.rows.value if i < len(self._shown)]

        def add_selected(self):
            entries = self.selected_entries()
            if not entries:
                self.recall_status.value = "Select one or more rows first (ctrl/shift-click for several)."
                return
            _, outside = ph.recall(self.ax, entries, reference=self.look.value == "reference")
            text = f"Added {len(entries)} curve{'s' if len(entries) != 1 else ''}."
            if outside:
                text += " Outside the current x-range (zoom out / Home to see): " + ", ".join(outside)
            self.refresh_save_list()
            self.refill_table()
            self.recall_status.value = text

        def remove_recalled(self):
            n = ph.remove_recalled(self.ax)
            self.refresh_save_list()
            self.refill_table()
            self.recall_status.value = f"Removed {n} recalled curve{'s' if n != 1 else ''}."

        def _reset_delete(self):
            self._pending_delete = None
            self.delete_button.description = "Delete selected…"
            self.delete_button.button_style = ""

        def delete_selected(self):
            entries = self.selected_entries()
            if not entries:
                self.recall_status.value = "Select one or more rows first."
                return
            key = tuple((str(e.path), e.group) for e in entries)
            n = len(entries)
            if self._pending_delete != key:
                # no modal dialogs in a notebook: a second click confirms
                self._pending_delete = key
                self.delete_button.description = f"Confirm delete ({n})"
                self.delete_button.button_style = "danger"
                self.recall_status.value = (
                    f"Click “Confirm delete” to delete {n} saved curve{'s' if n != 1 else ''} -- this can't be undone."
                )
                return
            ph.delete_entries(entries)
            self.refresh_recall()
            self.recall_status.value = f"Deleted {n} saved curve{'s' if n != 1 else ''}."

    return HistoryPanel()
