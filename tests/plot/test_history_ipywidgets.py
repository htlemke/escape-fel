"""The History panel's ipywidgets version (used for ipympl figures),
driven in-process: widgets work without a browser, and setting a value or
calling ``button.click()`` runs the same callbacks a browser event would.

Uses plain Agg figures (``matplotlib.figure.Figure``), so it doesn't touch
pyplot's backend -- the Qt test modules in this folder switch that.
"""
import pytest

pytest.importorskip("ipywidgets")
pytest.importorskip("h5py")

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import escape.plot_history as ph
from escape._plot_items import collect_items
from escape.history_gui import make_ipywidgets_history_panel
from escape.plot_utilities import errortube, tag
from escape.storage.example_data import make_scan


@pytest.fixture(autouse=True)
def history_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ph, "PLOT_HISTORY_DIR", str(tmp_path / "history"))
    return tmp_path / "history"


def new_axes(title="demo", xlabel="delay"):
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return ax


def busy_plot():
    ax = new_axes()
    x = np.linspace(0, 10, 30)
    tag(ax.plot(x, np.sin(x), "o-", label="plain"), sample="LaB6")
    errortube(x, np.sin(x) + 2, yerr=0.2 * np.ones_like(x), axis=ax, step="mid", label="tube", meta={"run": 7})
    make_scan(n_steps=8, n_events_per_step=50, scan_par_name="delay", name="diode", seed=0).scan.plot(
        axis=ax, capsize=3, label="scan"
    )
    ax.legend()
    return ax


def test_save_tab(history_dir):
    ax = busy_plot()
    panel = make_ipywidgets_history_panel(ax.figure, ax)
    assert [cb.description for cb in panel._checks] == ["plain", "tube  [error tube]", "scan  [error bars]"]
    assert panel.save_button.description == "Save 3 curves"

    panel.save_filter.value = "tube"
    assert [r.layout.display for r in panel._rows] == ["none", "", "none"]
    panel._set_ticks(False)  # "None" acts on the filtered rows only
    assert [cb.value for cb in panel._checks] == [True, False, True]
    panel.save_filter.value = ""

    panel.note.value = "before beamtime"
    panel.save_button.click()
    assert len(list(history_dir.glob("*.h5"))) == 1
    assert "Saved 2 curves" in panel.save_status.value
    assert {e.label for e in ph.list_entries()} == {"plain", "scan"}


def test_recall_tab_add_remove_delete(history_dir):
    ax = busy_plot()
    ph.save_items(collect_items(ax), ax, note="first")

    target = new_axes()
    target.plot([0, 10], [0, 1], label="today")
    panel = make_ipywidgets_history_panel(target.figure, target)
    assert len(panel.rows.options) == 3
    assert all(text.startswith("*") for text, _ in panel.rows.options)
    assert " " in panel.rows.options[0][0]  # padded for the monospace columns

    panel.search.value = "run 7"  # metadata is searchable
    assert len(panel.rows.options) == 1 and "tube" in panel.rows.options[0][0]
    panel.rows.value = (0,)
    panel.look.value = "reference"
    panel.add_button.click()
    items = collect_items(target)
    assert [it.kind for it in items] == ["line", "errortube"] and items[1].recalled
    assert items[1].main.get_linestyle() == "--"
    assert len(panel._checks) == 2  # the Save list follows the plot
    assert "Added 1 curve" in panel.recall_status.value

    panel.add_button.click()  # nothing selected after the refill
    assert "Select one or more rows" in panel.recall_status.value

    panel.remove_recalled()
    assert [it.label for it in collect_items(target)] == ["today"]

    panel.search.value = ""
    panel.rows.value = tuple(range(len(panel.rows.options)))
    panel.delete_button.click()  # first click only asks
    assert len(ph.list_entries()) == 3 and panel.delete_button.description == "Confirm delete (3)"
    panel.delete_button.click()
    assert ph.list_entries() == [] and list(history_dir.glob("*.h5")) == []


def test_changing_the_selection_cancels_a_pending_delete():
    ax = busy_plot()
    ph.save_items(collect_items(ax), ax)
    panel = make_ipywidgets_history_panel(ax.figure, ax)
    panel.rows.value = (0,)
    panel.delete_button.click()
    panel.rows.value = (1,)
    assert panel.delete_button.description == "Delete selected…"
    panel.delete_button.click()  # asks again for the new selection
    assert len(ph.list_entries()) == 3


def test_only_matching_and_bad_dates():
    other = new_axes(title="other", xlabel="energy")
    other.plot([100, 200], [1, 2], label="xas")
    ph.save_items(collect_items(other), other)

    ax = busy_plot()
    panel = make_ipywidgets_history_panel(ax.figure, ax)
    assert panel.rows.options == () and "untick" in panel.recall_status.value
    panel.only_matching.value = False
    assert len(panel.rows.options) == 1
    panel.date_from.value = "yesterday-ish"
    assert "unreadable from date" in panel.recall_status.value and len(panel.rows.options) == 1


def test_errors_in_callbacks_show_in_the_panel(monkeypatch):
    ax = busy_plot()
    panel = make_ipywidgets_history_panel(ax.figure, ax)

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ph, "save_items", broken)
    panel.save_button.click()
    assert "disk full" in panel.save_status.value
