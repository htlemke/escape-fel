"""Tests for plot items / ``escape_meta`` tagging, the plot history store and
the History toolbar window (:mod:`escape._plot_items`,
:mod:`escape.plot_history`, :mod:`escape.history_gui`).

Headless Qt like ``test_select_curves.py``; skipped without qtpy + a Qt
binding.
"""
import os

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("h5py")

import matplotlib
import matplotlib.cbook as cbook
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
cbook._get_running_interactive_framework = lambda: None
matplotlib.use("QtAgg")

try:
    from qtpy import QtWidgets

    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
except Exception as e:
    pytest.skip(f"Qt platform plugin unavailable: {e}", allow_module_level=True)

import matplotlib.pyplot as plt  # noqa: E402

import escape.plot_history as ph  # noqa: E402
from escape._plot_items import collect_items, data_lines, item_arrays  # noqa: E402
from escape.plot_utilities import attach_history_button, attach_select_button, errortube, tag  # noqa: E402
from escape.storage.example_data import make_scan  # noqa: E402

HISTORY_TIP = "History: save curves, recall saved ones"
SELECT_TIP = "Select curves to show"


def pump():
    _APP.processEvents()


@pytest.fixture(autouse=True)
def history_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PLOT_HISTORY_DIR", str(tmp_path / "history"))
    yield tmp_path / "history"
    plt.close("all")


def busy_plot(ax=None):
    """A plain line, a step line, a step errortube and a Scan.plot()."""
    if ax is None:
        _, ax = plt.subplots()
    ax.set_title("demo")
    ax.set_xlabel("delay")
    ax.set_ylabel("signal")
    x = np.linspace(0, 10, 30)
    tag(ax.plot(x, np.sin(x), "o-", color="tab:red", label="plain"), sample="LaB6")
    ax.step(x, np.cos(x), where="mid", label="stepped")
    errortube(x, np.sin(x) + 2, yerr=0.2 * np.ones_like(x), axis=ax, step="mid", label="tube", meta={"run": 7})
    scan = make_scan(n_steps=8, n_events_per_step=50, scan_par_name="delay", name="diode", seed=0)
    scan.scan.plot(axis=ax, capsize=3, label="scan")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# items and tagging


def test_combined_plots_are_one_item_each():
    ax = busy_plot()
    items = collect_items(ax)
    assert [(it.kind, it.label) for it in items] == [
        ("line", "plain"), ("line", "stepped"), ("errortube", "tube"), ("errorbar", "scan"),
    ]
    tube = items[2]
    assert len(tube.members) == 2  # line + band
    # error-bar caps are Line2D objects, but not "data"
    assert len(ax.get_lines()) > len(data_lines(ax)) == 4


def test_tag_shares_and_extends_one_dict():
    _, ax = plt.subplots()
    line, band = errortube([0, 1, 2], [1, 2, 3], yerr=[0.1, 0.1, 0.1], axis=ax, meta={"run": 3})
    tag(line, sample="Si")
    assert band[0].escape_meta is line.escape_meta
    assert line.escape_meta["run"] == 3 and line.escape_meta["sample"] == "Si"


def test_scan_plot_tags_array_and_scan_parameter():
    ax = busy_plot()
    item = collect_items(ax)[3]
    assert item.meta["array"] == "diode" and item.meta["scan_parameter"] == "delay"


# ---------------------------------------------------------------------------
# store


def test_save_list_recall_roundtrip(history_dir):
    ax = busy_plot()
    ph.save_items(collect_items(ax), ax, note="first try")
    entries = ph.list_entries()
    assert {e.label for e in entries} == {"plain", "stepped", "tube", "scan"}
    assert all(e.note == "first try" for e in entries)
    assert next(e for e in entries if e.label == "tube").meta == {"run": 7}

    _, ax2 = plt.subplots()
    drawn, outside = ph.recall(ax2, entries)
    assert not outside
    by_label = {e.label: e for e in entries}
    recalled = collect_items(ax2)
    assert {it.kind for it in recalled} == {"line", "errortube", "errorbar"}
    for item in recalled:
        assert item.recalled
        entry = by_label[item.label.rsplit(" (", 1)[0]]
        stored, again = entry.load(), item_arrays(item)
        assert set(stored) == set(again)
        for key in stored:
            np.testing.assert_allclose(again[key], stored[key], equal_nan=True)
        assert item.meta["recalled_from"] == str(entry.path)
    stepped = next(it for it in recalled if it.label.startswith("stepped"))
    assert stepped.main.get_drawstyle() == "steps-mid"
    plain = next(it for it in recalled if it.label.startswith("plain"))
    assert plain.main.get_marker() == "o" and plain.meta["sample"] == "LaB6"
    assert len(ax2.get_legend().get_texts()) == 4  # a legend was created

    assert ph.remove_recalled(ax2) == 4 and collect_items(ax2) == []


def test_reference_look_is_dashed_and_faded():
    ax = busy_plot()
    ph.save_items(collect_items(ax)[:1], ax)
    _, ax2 = plt.subplots()
    ph.recall(ax2, ph.list_entries(), reference=True)
    line = ax2.get_lines()[0]
    assert line.get_linestyle() == "--" and line.get_alpha() == pytest.approx(0.5)


def test_date_axis_roundtrip():
    t = np.datetime64("2026-10-16T10:00") + np.arange(20) * np.timedelta64(1, "m")
    _, ax = plt.subplots()
    ax.plot(t, np.arange(20.0), label="archiver")
    ph.save_items(collect_items(ax), ax)
    entry = ph.list_entries()[0]
    assert entry.record["x_units"] == "date"
    _, ax2 = plt.subplots()
    ph.recall(ax2, [entry])
    assert np.asarray(ax2.get_lines()[0].get_xdata())[0] == np.datetime64("2026-10-16T10:00")


def test_delete_removes_emptied_files(history_dir):
    ax = busy_plot()
    ph.save_items(collect_items(ax), ax)
    entries = ph.list_entries()
    ph.delete_entries(entries[:2])
    assert len(ph.list_entries()) == 2
    ph.delete_entries(ph.list_entries())
    assert list(history_dir.glob("*.h5")) == []


def test_matcher_ranks_the_same_kind_of_plot_first():
    ax = busy_plot()
    ph.save_items(collect_items(ax)[:1], ax, note="same plot")
    _, other = plt.subplots()
    other.set_title("something else")
    other.plot([100, 200], [1, 2], label="unrelated")
    ph.save_items(collect_items(other), other, note="other plot")

    _, target = plt.subplots()
    target.set_title("demo")
    target.set_xlabel("delay")
    target.plot(np.linspace(0, 10, 5), np.zeros(5), label="plain")
    m = ph.Matcher(target)
    stars = {e.note: m.stars(e) for e in ph.list_entries()}
    assert stars["same plot"] == 3 and stars["other plot"] == 0


def test_filter_by_words_and_dates():
    ax = busy_plot()
    ph.save_items(collect_items(ax), ax, note="after realignment")
    entries = ph.list_entries()
    assert {e.label for e in ph.filter_entries(entries, "tube 7")} == {"tube"}  # meta run=7 is searchable
    assert len(ph.filter_entries(entries, "realignment")) == 4
    assert ph.filter_entries(entries, date_to=ph.parse_date("2000-01-01", end=True)) == []
    assert ph.parse_date("not a date") is None


# ---------------------------------------------------------------------------
# Select with combined items


def test_select_hides_errortube_band_and_refits_with_it():
    ax = busy_plot()
    attach_select_button(ax.figure)
    action = next(a for a in ax.figure.canvas.manager.toolbar.actions() if a.toolTip() == SELECT_TIP)
    action.trigger()
    pump()
    dialog = ax._escape_select_gui
    boxes = dialog.findChildren(QtWidgets.QCheckBox)
    assert [b.text() for b in boxes] == ["plain", "stepped", "tube  [error tube]", "scan  [error bars]"]

    tube = collect_items(ax)[2]
    for i, box in enumerate(boxes):
        box.setChecked(i == 2)  # only the tube
    pump()
    assert all(m.get_visible() for m in tube.members)
    lo, hi = ax.get_ylim()
    band_lo, band_hi = tube.y_extent()
    assert lo <= band_lo and hi >= band_hi and (hi - lo) < 2 * (band_hi - band_lo)
    boxes[2].setChecked(False)
    pump()
    assert not any(m.get_visible() for m in tube.members)


# ---------------------------------------------------------------------------
# History window


def toolbar_actions(fig):
    return fig.canvas.manager.toolbar.actions()


def open_history(fig):
    next(a for a in toolbar_actions(fig) if a.toolTip() == HISTORY_TIP).trigger()
    pump()
    dialog = fig.axes[0]._escape_history_gui
    assert dialog.isVisible()
    return dialog


def test_button_sits_after_save_and_select():
    ax = busy_plot()
    fig = ax.figure
    attach_select_button(fig)
    attach_history_button(fig)
    attach_history_button(fig)  # idempotent
    tb = fig.canvas.manager.toolbar
    acts = toolbar_actions(fig)
    save = acts.index(tb._actions["save_figure"])
    assert [a.toolTip() for a in acts[save + 1:save + 3]] == [SELECT_TIP, HISTORY_TIP]
    assert sum(a.toolTip() == HISTORY_TIP for a in acts) == 1
    assert not acts[save + 2].icon().isNull()


def test_save_then_recall_through_the_window(history_dir):
    ax = busy_plot()
    fig = ax.figure
    attach_history_button(fig)
    dialog = open_history(fig)

    # Save tab: all four ticked; untick "stepped", add a note, save
    boxes = dialog._boxes
    assert len(boxes) == 4 and all(b.isChecked() for b in boxes)
    boxes[1].setChecked(False)
    assert dialog.save_button.text() == "Save 3 curves"
    dialog.note_edit.setText("before beamtime")
    dialog.save_button.click()
    pump()
    assert len(list(history_dir.glob("*.h5"))) == 1

    # Recall tab, on another figure with the same axis labels
    _, ax2 = plt.subplots()
    ax2.set_title("demo")
    ax2.set_xlabel("delay")
    ax2.plot(np.linspace(0, 10, 4), np.zeros(4), label="today")
    attach_history_button(ax2.figure)
    d2 = open_history(ax2.figure)
    assert d2.table.rowCount() == 3
    assert all(d2.table.item(r, 0).text().startswith("★") for r in range(3))
    assert d2.table.item(0, 6).text() == "before beamtime"

    d2.search_edit.setText("tube")
    pump()
    assert d2.table.rowCount() == 1 and d2.table.item(0, 2).text() == "tube"
    d2.table.selectRow(0)
    d2.add_button.click()
    pump()
    items = collect_items(ax2)
    assert [it.kind for it in items] == ["line", "errortube"] and items[1].recalled
    assert items[1].label.startswith("tube (")
    assert len(d2._boxes) == 2  # Save list follows the plot

    d2.search_edit.setText("")
    d2.remove_recalled()
    assert [it.label for it in collect_items(ax2)] == ["today"]

    # delete, with the confirmation answered "yes"
    d2.confirm = lambda text: True
    d2.table.selectAll()
    d2.delete_selected()
    assert ph.list_entries() == [] and list(history_dir.glob("*.h5")) == []


def test_only_matching_hides_unrelated_entries():
    _, other = plt.subplots()
    other.set_title("other")
    other.set_xlabel("energy")
    other.plot([100, 200], [1, 2], label="xas")
    ph.save_items(collect_items(other), other)

    ax = busy_plot()
    attach_history_button(ax.figure)
    dialog = open_history(ax.figure)
    dialog.tabs.setCurrentIndex(1)
    assert dialog.table.rowCount() == 0 and "untick" in dialog.recall_status.text()
    dialog.only_matching.setChecked(False)
    assert dialog.table.rowCount() == 1


def test_history_on_an_empty_figure_recalls_into_it():
    ax = busy_plot()
    ph.save_items(collect_items(ax), ax)
    fig = plt.figure()
    attach_history_button(fig)
    dialog = open_history(fig)  # creates the axes
    dialog.only_matching.setChecked(False)
    dialog.table.selectAll()
    dialog.add_selected()
    assert len(collect_items(fig.axes[0])) == 4
