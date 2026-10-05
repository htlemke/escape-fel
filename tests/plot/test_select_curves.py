"""Behaviour tests for the "Select curves to show" toolbar button
(:func:`escape.plot_utilities.attach_select_button`, :mod:`escape.select_gui`).

Ported from a standalone headless check written against a tested prototype
in eco's archiver plotting (``eco.dbase.archiver``) -- see
``escape/select_gui.py``'s module docstring for where the engine/front-ends
here came from. Needs ``qtpy`` plus a Qt binding; the whole module is
skipped if either is missing, or if even an offscreen Qt platform plugin
can't start (e.g. no ``PyQt5``/``PySide`` installed at all).

Run just this file directly:
    QT_QPA_PLATFORM=offscreen python3 -m pytest tests/plot/test_select_curves.py
"""
import os

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("matplotlib")

import matplotlib
import matplotlib.cbook as cbook
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Offscreen Qt needs no display, but matplotlib insists on one before it will
# pick a Qt backend or create the QApplication itself: bypass both checks.
# This flips matplotlib's backend process-wide (same as the standalone
# script this was ported from) -- fine for a dedicated test module, but
# don't import this one in the same process as something that needs a
# different backend.
cbook._get_running_interactive_framework = lambda: None
matplotlib.use("QtAgg")

try:
    from qtpy import QtGui, QtWidgets

    _APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
except Exception as e:  # no Qt platform plugin available even offscreen
    pytest.skip(f"Qt platform plugin unavailable: {e}", allow_module_level=True)

import matplotlib.pyplot as plt  # noqa: E402

from escape.plot_utilities import attach_select_button  # noqa: E402

TIP = "Select curves to show"


def make_figure(n_lines=9, legend=True):
    fig, ax = plt.subplots()
    x = np.arange(200.0)
    for i in range(n_lines):
        kind = ("direction", "offset", "readback")[i % 3]
        scale = 100.0 if i == 5 else 1.0  # one curve whose scale dwarfs the rest
        ax.plot(x, scale * (np.sin(x / 20 + i) + i), ".-", label=f"stage.{i // 3}.{kind}")
    if legend:
        ax.legend()
    return fig, ax


def toolbar_of(fig):
    return fig.canvas.manager.toolbar


def select_action(fig):
    return next((a for a in toolbar_of(fig).actions() if a.toolTip() == TIP), None)


def pump():
    _APP.processEvents()


def open_dialog(fig):
    select_action(fig).trigger()
    pump()
    dialog = getattr(fig.axes[0], "_escape_select_gui", None)
    assert dialog is not None and dialog.isVisible(), "dialog did not open"
    return dialog


def widgets_of(dialog):
    boxes = dialog.findChildren(QtWidgets.QCheckBox)
    edit = dialog.findChild(QtWidgets.QLineEdit)
    buttons = {b.text(): b for b in dialog.findChildren(QtWidgets.QPushButton)}
    return boxes, edit, buttons


def ylim_span(ax):
    lo, hi = ax.get_ylim()
    return hi - lo


@pytest.fixture
def closing():
    """Collect figures created during a test and close them afterward --
    matplotlib figures (and their Qt windows) otherwise pile up across
    tests in the same process."""
    figs = []
    yield figs
    for fig in figs:
        plt.close(fig)


def test_button_placement_and_icon(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    tb = toolbar_of(fig)
    acts = tb.actions()
    save = tb._actions["save_figure"]
    action = select_action(fig)
    assert action is not None, "no toolbar action"
    assert acts.index(action) == acts.index(save) + 1, "not right after Save"
    assert tb.widgetForAction(acts[acts.index(action) + 1]) is tb.locLabel
    assert not action.icon().isNull(), "button has no icon"


def test_dialog_lists_all_curves_with_swatches_and_legend(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)
    assert len(boxes) == 9 and all(b.isChecked() for b in boxes)
    assert not boxes[0].icon().isNull(), "rows should carry the line's legend handle"

    w = dialog
    while w is not None and w is not fig.canvas.manager.window:
        w = w.parent()
    assert w is not None, "dialog is not a child of the figure window"
    assert len(ax.get_legend().get_texts()) == 9


def test_unticking_hides_curve_drops_legend_entry_and_refits_y(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)

    before = ylim_span(ax)
    big = next(b for b in boxes if b.text() == "stage.1.readback")  # the x100 curve (i=5)
    big.setChecked(False)
    pump()
    assert not ax.get_lines()[5].get_visible()
    assert len(ax.get_legend().get_texts()) == 8
    assert ylim_span(ax) < before / 10, "y-range was not refit after hiding the big curve"

    big.setChecked(True)
    pump()
    assert len(ax.get_legend().get_texts()) == 9


def test_refit_works_after_a_manual_zoom(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)

    ax.set_ylim(-1, 1)
    assert not ax.get_autoscaley_on()
    boxes[0].setChecked(False)
    pump()
    assert ylim_span(ax) > 2.5, "y-range stayed frozen after a manual zoom"

    boxes[0].setChecked(True)
    pump()


def test_filter_and_all_none_act_on_filtered_rows_only(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)

    edit.setText("readback")
    pump()
    shown = [b for b in boxes if not b.isHidden()]
    assert [b.text() for b in shown] == [b.text() for b in boxes if "readback" in b.text()]
    assert len(shown) == 3

    buttons["None"].click()
    pump()
    vis = [ln.get_visible() for ln in ax.get_lines()]
    assert [v for v, b in zip(vis, boxes) if "readback" in b.text()] == [False] * 3
    assert all(v for v, b in zip(vis, boxes) if "readback" not in b.text()), "None touched filtered-out rows"

    buttons["All"].click()
    pump()
    assert all(ln.get_visible() for ln in ax.get_lines())

    edit.setText("")
    pump()
    assert all(not b.isHidden() for b in boxes)


def test_none_without_filter_empties_plot_and_drops_legend_all_restores(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)

    buttons["None"].click()
    pump()
    assert not any(ln.get_visible() for ln in ax.get_lines()) and ax.get_legend() is None

    buttons["All"].click()
    pump()
    assert len(ax.get_legend().get_texts()) == 9

    # "only the readbacks": None, then filter + All
    buttons["None"].click()
    edit.setText("readback")
    buttons["All"].click()
    edit.setText("")
    pump()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        ln.get_label() for ln in ax.get_lines() if "readback" in ln.get_label()
    ]

    buttons["All"].click()
    pump()


def test_second_click_reuses_same_dialog(closing):
    fig, ax = make_figure()
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)

    dialog.close()
    pump()
    select_action(fig).trigger()
    pump()
    assert getattr(fig.axes[0], "_escape_select_gui", None) is dialog and dialog.isVisible()


def test_closing_figure_closes_dialog():
    fig, ax = make_figure()
    attach_select_button(fig)
    dialog = open_dialog(fig)

    plt.close(fig)
    pump()
    try:
        assert not dialog.isVisible()
    except RuntimeError:
        pass  # C++ object already deleted: also fine


def test_no_legend_created_where_there_was_none(closing):
    fig, ax = make_figure(legend=False)
    closing.append(fig)
    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)
    boxes[0].setChecked(False)
    pump()
    assert ax.get_legend() is None, "a legend must not appear where there was none"


def test_existing_legend_keeps_position_title_and_column_count(closing):
    fig, ax = make_figure()
    closing.append(fig)
    ax.legend(loc="lower left", title="stages", ncol=2)
    loc = ax.get_legend()._loc  # private, but the only handle on where it sits

    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)
    boxes[0].setChecked(False)
    pump()

    legend = ax.get_legend()
    assert legend._loc == loc and legend.get_title().get_text() == "stages"
    assert len(legend.get_texts()) == 8


def test_non_line_labelled_artists_keep_their_legend_entry(closing):
    fig, ax = make_figure()
    closing.append(fig)
    ax.fill_between(np.arange(200.0), -1, 1, alpha=0.2, label="band")
    ax.legend()
    assert len(ax.get_legend().get_texts()) == 10

    attach_select_button(fig)
    dialog = open_dialog(fig)
    boxes, edit, buttons = widgets_of(dialog)
    assert len(boxes) == 9, "only the lines are selectable"
    boxes[0].setChecked(False)
    pump()

    assert len(ax.get_legend().get_texts()) == 9
    assert "band" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_single_curve_gets_no_button(closing):
    fig, ax = make_figure(n_lines=1)
    closing.append(fig)
    attach_select_button(fig)
    assert select_action(fig) is None


def test_button_appears_once_curves_exist_on_a_retry(closing):
    """Documents the deliberate deviation from Fit/Peak/Freq's simple
    idempotent-flag attach: a figure attached to before it had two curves
    gets no button, but calling attach_select_button again later -- once
    there's something to select -- does add it then (see
    attach_select_button's and _select_button_present's docstrings)."""
    fig, ax = make_figure(n_lines=1)
    closing.append(fig)
    attach_select_button(fig)
    assert select_action(fig) is None

    ax.plot(np.arange(10.0), np.arange(10.0), label="second")
    attach_select_button(fig)
    assert select_action(fig) is not None


def test_icon_follows_toolbar_text_colour(closing):
    fig, ax = make_figure()
    closing.append(fig)
    tb = toolbar_of(fig)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 40, 40))
    tb.setPalette(palette)

    attach_select_button(fig)
    img = select_action(fig).icon().pixmap(24, 24).toImage()
    opaque = [img.pixelColor(x, y) for x in range(24) for y in range(24) if img.pixelColor(x, y).alpha() > 200]
    assert opaque and all(c.red() > 200 for c in opaque), "icon is not drawn in the toolbar's text colour"
