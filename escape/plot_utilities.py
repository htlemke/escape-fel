from __future__ import annotations

import hashlib
import matplotlib.pyplot as plt
from time import sleep
import ipywidgets as widgets
import numpy as np
import threading
from threading import Thread
import dask.array as da
from dask.array import ptp
from dataclasses import dataclass
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from IPython import get_ipython
from IPython.display import display

try:
    from sidecar import Sidecar
except ImportError:  # pragma: no cover - optional dependency
    Sidecar = None


class GinputNB:
    """Interactively collect clicked points from a notebook plot.

    Left-click adds a point (drawn/updated live via ``plotspec``), right-click
    removes the last one. Press ``t`` to toggle collecting on/off, ``d`` to
    stop collecting early. Collected coordinates accumulate in ``self.x`` /
    ``self.y``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to collect clicks on. Defaults to the current figure.
    Npts : float
        Number of points to collect before stopping automatically; a
        negative value (the default) collects until stopped with ``d``.
    plotspec : str
        Matplotlib format string for the marker/line showing collected
        points as they're added. Pass ``""``/``None`` to not draw them.
    """

    def __init__(self, fig=None, Npts=-1.0, plotspec="rd-"):
        self.x = []
        self.y = []
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0
        self.get(Npts, plotspec)

    def get(self, Npts=-1.0, plotspec="rd-"):
        self.mcid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.kcid = self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        if plotspec:
            self.line = plt.plot(self.x, self.y, plotspec)[0]
        self.collecting = Npts
        return self

    def toggle_selector(self, event):
        if event.key == "t":
            if self.collecting == 0:
                self.collecting = 1
            if self.collecting == 1:
                self.collecting = 0

    def onclick(self, event):
        if self.collecting == 0:
            return
        if event.button == 1:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            if self.collecting > 0:
                self.collecting -= 1
        else:
            self.x.pop()
            self.y.pop()
            if self.collecting > 0:
                self.collecting += 1
        if self.line:
            self.line.set_data(self.x, self.y)
            if self.collecting == 0:
                self.stop()
            self.line.figure.canvas.draw()

    def stop(self):
        self.fig.canvas.mpl_disconnect(self.mcid)
        self.fig.canvas.mpl_disconnect(self.kcid)
        self.collecting = 0
        donetx = plt.text(self.x[-1], self.y[-1], "Done!", color=[0, 1, 0])
        self.line.figure.canvas.draw()
        sleep(2)
        donetx.remove()

    def onkey(self, event):
        if event.key == "d":
            self.stop()


import ipywidgets as widgets
from matplotlib.widgets import (
    RectangleSelector,
    SpanSelector,
    PolygonSelector,
    LassoSelector,
)
from matplotlib.path import Path as MplPath


def make_box_layout():
    """Standard bordered/padded ``ipywidgets.Layout`` shared by the widgets
    in this module, so their boxes look consistent."""
    return widgets.Layout(
        border="solid 1px black",
        margin="0px 10px 10px 0px",
        padding="5px 5px 5px 5px",
        flex_flow="row wrap",
    )


class SpanSelectNB:
    """Interactive 1-D span (range) selector on a matplotlib axes.

    Wraps ``matplotlib.widgets.SpanSelector``; starts inactive -- press ``t``
    to toggle selection on/off. If ``ax_roi`` is given, the image on ``ax``
    is cropped to the selected span and shown there, sharing ``ax``'s
    colormap limits; ``get_image_roi_data``/``get_image_slice_selection``
    can also be called directly to fetch the cropped data.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure the selector is attached to. Defaults to the current figure.
    ax : matplotlib.axes.Axes, optional
        Axes to select on. Defaults to the current axes.
    ax_roi : matplotlib.axes.Axes, optional
        Axes to draw the selected-region preview into.
    direction : {"horizontal", "vertical"}
        Axis the span is measured along.
    callbacks_changeroi : list[callable]
        Zero-argument callables invoked whenever the selection changes.
    """

    def __init__(
        self,
        fig=None,
        ax=None,
        ax_roi=None,
        direction="horizontal",
        callbacks_changeroi=[],
    ):
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0
        if not ax:
            ax = plt.gca()
        self.ax = ax
        self.ax_roi = ax_roi

        self.selector = SpanSelector(
            ax,
            self.line_select_callback,
            direction=direction,
            # drawtype="box",
            useblit=False,
            button=[1, 3],  # don't use middle button
            # minspan=5,
            # spancoords="data",
            interactive=True,
        )
        fig.canvas.mpl_connect("key_press_event", self.toggle_selector)
        self.selector.set_active(False)
        self.callbacks_changeroi = callbacks_changeroi

    def toggle_selector(self, event):
        if event.key == "t":
            if self.selector.active:
                print("Selector deactivated.")
                self.selector.set_active(False)
                # self.rectangle = Rectangle([10,10],20,20)

            else:
                print("Selector activated.")
                self.selector.set_active(True)

    def line_select_callback(self, eclick, erelease):
        if self.ax_roi:
            roi_data = self.get_image_roi_data()
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(roi_data)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
            else:
                # Create new image with synchronized colormap
                main_image = self.ax.get_images()[0]
                vmin, vmax = main_image.get_clim()
                i = self.ax_roi.imshow(roi_data, vmin=vmin, vmax=vmax)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
        for callback in self.callbacks_changeroi:
            callback()

    def get_image_roi_data(self):
        i = self.ax.get_images()[0]
        return i.get_array()[self.get_image_slice_selection()]

    def get_image_slice_selection(self):
        return slice(int(np.round(self.ymin)), int(np.round(self.ymax))), slice(
            int(np.round(self.xmin)), int(np.round(self.xmax))
        )

    @property
    def vmin(self):
        return self.selector.extents[0]

    @property
    def vmax(self):
        return self.selector.extents[1]


class PolygonSelectNB:
    """Interactive polygon ROI selector over an image axes.

    Wraps ``matplotlib.widgets.PolygonSelector``; starts inactive -- press
    ``t`` to toggle selection on/off. Click to place vertices, close the
    polygon to finish it. If ``ax_roi`` is given, the image on ``ax`` is
    cropped to the polygon's bounding box and shown there; use
    :meth:`get_mask` for the actual boolean polygon mask.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure the selector is attached to. Defaults to the current figure.
    ax : matplotlib.axes.Axes, optional
        Axes to select on. Defaults to the current axes.
    ax_roi : matplotlib.axes.Axes, optional
        Axes to draw the selected-region preview into.
    image_handle : matplotlib.image.AxesImage, optional
        Image whose data :meth:`get_mask` sizes itself to. Defaults to the
        first image on ``ax``.
    callbacks_changeroi : list[callable]
        Zero-argument callables invoked whenever the selection changes.
    """

    def __init__(
        self,
        fig=None,
        ax=None,
        ax_roi=None,
        image_handle=None,
        direction="horizontal",
        callbacks_changeroi=[],
    ):
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0
        if not ax:
            ax = plt.gca()
        self.ax = ax

        if not image_handle:
            image_handle = self.ax.get_images()[0]
        self.image_handle = image_handle

        self.ax_roi = ax_roi

        self.selector = PolygonSelector(
            ax,
            self.line_select_callback,
            # direction=direction,
            # drawtype="box",
            useblit=False,
            # button=[1, 3],  # don't use middle button
            # minspan=5,
            # spancoords="data",
            # interactive=True,
        )
        fig.canvas.mpl_connect("key_press_event", self.toggle_selector)
        self.selector.set_active(False)
        self.callbacks_changeroi = callbacks_changeroi

    def toggle_selector(self, event):
        if event.key == "t":
            if self.selector.active:
                print("Selector deactivated.")
                self.selector.set_active(False)
                # self.rectangle = Rectangle([10,10],20,20)

            else:
                print("Selector activated.")
                self.selector.set_active(True)

    def line_select_callback(self, eclick, erelease):
        if self.ax_roi:
            roi_data = self.get_image_roi_data()
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(roi_data)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
            else:
                # Create new image with synchronized colormap
                main_image = self.ax.get_images()[0]
                vmin, vmax = main_image.get_clim()
                i = self.ax_roi.imshow(roi_data, vmin=vmin, vmax=vmax)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
        for callback in self.callbacks_changeroi:
            callback()

    def get_image_roi_data(self):
        i = self.ax.get_images()[0]
        return i.get_array()[self.get_image_slice_selection()]

    def get_image_slice_selection(self):
        return slice(int(np.round(self.ymin)), int(np.round(self.ymax))), slice(
            int(np.round(self.xmin)), int(np.round(self.xmax))
        )

    def get_mask(self, image=None):
        """Get binary mask of the ROI polygon.

        Parameters
        ----------
        image: numpy array (2D)
            Image that the mask should be based on. Only used for determining
            the shape of the binary mask (which is made equal to the shape of
            the image)

        Returns
        -------
        numpy array (2D)

        """
        if image is None:
            image = self.image_handle.get_array()
        ny, nx = np.shape(image)
        poly_verts = [(self.x[0], self.y[0])] + list(
            zip(reversed(self.x), reversed(self.y))
        )
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        roi_path = MplPath(poly_verts)
        mask = roi_path.contains_points(points).reshape((ny, nx))
        return mask

    @property
    def verts(self):
        return self.selector.verts

    @property
    def x(self):
        return [tv[0] for tv in self.verts]

    @property
    def y(self):
        return [tv[1] for tv in self.verts]


class LassoSelectNB:
    """Interactive freehand lasso ROI selector over an image axes.

    Wraps ``matplotlib.widgets.LassoSelector``; starts inactive -- press
    ``t`` to toggle selection on/off. Drag to trace the ROI boundary. If
    ``ax_roi`` is given, the image on ``ax`` is cropped to the traced
    shape's bounding box and shown there; use :meth:`get_mask` for the
    actual boolean polygon mask.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure the selector is attached to. Defaults to the current figure.
    ax : matplotlib.axes.Axes, optional
        Axes to select on. Defaults to the current axes.
    ax_roi : matplotlib.axes.Axes, optional
        Axes to draw the selected-region preview into.
    image_handle : matplotlib.image.AxesImage, optional
        Image whose data :meth:`get_mask` sizes itself to. Defaults to the
        first image on ``ax``.
    callbacks_changeroi : list[callable]
        Zero-argument callables invoked whenever the selection changes.
    """

    def __init__(
        self,
        fig=None,
        ax=None,
        ax_roi=None,
        image_handle=None,
        direction="horizontal",
        callbacks_changeroi=[],
    ):
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0
        if not ax:
            ax = plt.gca()
        self.ax = ax

        if not image_handle:
            image_handle = self.ax.get_images()[0]
        self.image_handle = image_handle

        self.ax_roi = ax_roi

        self.selector = LassoSelector(
            ax,
            self.line_select_callback,
            # direction=direction,
            # drawtype="box",
            useblit=False,
            # button=[1, 3],  # don't use middle button
            # minspan=5,
            # spancoords="data",
            # interactive=True,
        )
        fig.canvas.mpl_connect("key_press_event", self.toggle_selector)
        self.selector.set_active(False)
        self.callbacks_changeroi = callbacks_changeroi

    def toggle_selector(self, event):
        if event.key == "t":
            if self.selector.active:
                print("Selector deactivated.")
                self.selector.set_active(False)
                # self.rectangle = Rectangle([10,10],20,20)

            else:
                print("Selector activated.")
                self.selector.set_active(True)

    def line_select_callback(self, eclick, erelease):
        if self.ax_roi:
            roi_data = self.get_image_roi_data()
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(roi_data)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
            else:
                # Create new image with synchronized colormap
                main_image = self.ax.get_images()[0]
                vmin, vmax = main_image.get_clim()
                i = self.ax_roi.imshow(roi_data, vmin=vmin, vmax=vmax)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
        for callback in self.callbacks_changeroi:
            callback()

    def get_image_roi_data(self):
        i = self.ax.get_images()[0]
        return i.get_array()[self.get_image_slice_selection()]

    def get_image_slice_selection(self):
        return slice(int(np.round(self.ymin)), int(np.round(self.ymax))), slice(
            int(np.round(self.xmin)), int(np.round(self.xmax))
        )

    def get_mask(self, image=None):
        """Get binary mask of the ROI polygon.

        Parameters
        ----------
        image: numpy array (2D)
            Image that the mask should be based on. Only used for determining
            the shape of the binary mask (which is made equal to the shape of
            the image)

        Returns
        -------
        numpy array (2D)

        """
        if image is None:
            image = self.image_handle.get_array()
        ny, nx = np.shape(image)
        poly_verts = [(self.x[0], self.y[0])] + list(
            zip(reversed(self.x), reversed(self.y))
        )
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        roi_path = MplPath(poly_verts)
        mask = roi_path.contains_points(points).reshape((ny, nx))
        return mask

    @property
    def verts(self):
        return self.selector.verts

    @property
    def x(self):
        return [tv[0] for tv in self.verts]

    @property
    def y(self):
        return [tv[1] for tv in self.verts]


class RectangleSelectNB:
    """Interactive draggable rectangle ROI selector over an image axes.

    Wraps ``matplotlib.widgets.RectangleSelector``; starts inactive -- press
    ``t`` to toggle selection on/off. Unlike a plain ``RectangleSelector``,
    the ROI preview/callbacks also update continuously *during* a drag (not
    only once it's released). If ``ax_roi`` is given, the image on ``ax`` is
    cropped to the rectangle and shown there, sharing ``ax``'s colormap
    limits.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure the selector is attached to. Defaults to the current figure.
    ax : matplotlib.axes.Axes, optional
        Axes to select on. Defaults to the current axes.
    ax_roi : matplotlib.axes.Axes, optional
        Axes to draw the selected-region preview into.
    callbacks_changeroi : list[callable]
        Zero-argument callables invoked whenever the selection changes.
    """

    def __init__(self, fig=None, ax=None, ax_roi=None, callbacks_changeroi=[]):
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0
        if not ax:
            ax = plt.gca()
        self.ax = ax
        self.ax_roi = ax_roi

        self.selector = RectangleSelector(
            ax,
            self.line_select_callback,
            # drawtype="box",
            useblit=False,
            button=[1, 3],  # don't use middle button
            minspanx=5,
            minspany=5,
            spancoords="data",
            interactive=True,
        )
        fig.canvas.mpl_connect("key_press_event", self.toggle_selector)
        self.selector.set_active(False)
        self.callbacks_changeroi = callbacks_changeroi

        # RectangleSelector's own onselect callback (line_select_callback) only
        # fires once a drag/resize is *released* -- by also re-running it on every
        # mouse-move while a drag is in progress (matplotlib already updates
        # self.selector.extents live during the drag itself), the ROI slice/mean
        # updates continuously as you drag, not just when you let go.
        self._last_live_extents = None
        fig.canvas.mpl_connect("motion_notify_event", self._on_drag_motion)

    def _on_drag_motion(self, event):
        if (
            not self.selector.active
            or getattr(self.selector, "_eventpress", None) is None
        ):
            return
        extents = self.selector.extents
        if extents == self._last_live_extents:
            return
        self._last_live_extents = extents
        self.line_select_callback(event, event)

    def toggle_selector(self, event):
        if event.key == "t":
            if self.selector.active:
                print(" RectangleSelector deactivated.")
                self.selector.set_active(False)
                # self.rectangle = Rectangle([10,10],20,20)

            else:
                print(" RectangleSelector activated.")
                self.selector.set_active(True)

    def line_select_callback(self, eclick, erelease):
        if self.ax_roi:
            roi_data = self.get_image_roi_data()
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(roi_data)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
            else:
                # Create new image with synchronized colormap
                main_image = self.ax.get_images()[0]
                vmin, vmax = main_image.get_clim()
                i = self.ax_roi.imshow(roi_data, vmin=vmin, vmax=vmax)
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
        for callback in self.callbacks_changeroi:
            callback()

    def get_image_roi_data(self):
        i = self.ax.get_images()[0]
        return i.get_array()[self.get_image_slice_selection()]

    def get_image_slice_selection(self):
        return slice(int(np.round(self.ymin)), int(np.round(self.ymax))), slice(
            int(np.round(self.xmin)), int(np.round(self.xmax))
        )

    @property
    def xmin(self):
        return float(self.selector.extents[0])

    @property
    def xmax(self):
        return float(self.selector.extents[1])

    @property
    def ymin(self):
        return float(self.selector.extents[2])

    @property
    def ymax(self):
        return float(self.selector.extents[3])


class MultipleRoiSelector(widgets.HBox):
    """Widget managing several rectangle ROIs on one shared image.

    Shows a colormap-range slider, an "Add roi" button, and a tab per ROI
    (each with an editable name and a cropped preview); selecting a tab makes
    only that ROI's rectangle active/visible on the main image. Final ROI
    extents are available via the :attr:`rois` property.

    Parameters
    ----------
    data : array-like, 2-D
        Image to select regions of interest on.
    rois : dict[str, tuple]
        Initial ``{name: (xmin, xmax, ymin, ymax)}`` regions to pre-populate,
        in addition to any added interactively afterwards.
    callbacks_changeanyroi : list[callable(self)]
        Callables invoked (with this selector) whenever any ROI is added,
        moved, or renamed.
    name : str
        Figure name used when this selector creates its own figure (i.e.
        when ``fig``/``ax`` are not given).
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes, optional
        Existing figure/axes already showing ``data``. If given, ROI
        rectangles are drawn directly on ``ax`` instead of a separate copy
        of the image -- used by :class:`StackViewer` so the main image isn't
        duplicated. If omitted (the default), a new figure + image (with its
        own ``Output`` widget) is created.
    convert_rois_to_int : {"inner", "outer", "round", False}
        If not ``False``, every ROI is snapped to integer pixel bounds
        (via :meth:`convert_rois_to_int`, using this as the strategy)
        each time it is added or moved.
    remember_last_roi_in_cell : bool
        If ``True``, the ROIs drawn on this ``data`` are remembered (keyed
        by the current notebook cell and the content of ``data``) and used
        as the initial ``rois`` the next time a selector is created for the
        same data in the same cell -- e.g. after re-running the cell, the
        previous ROIs are still there. If several selectors share a
        cell/data at once, whichever is changed most recently wins.
    detached : bool
        If ``True``, show this widget in a JupyterLab Sidecar panel instead
        of inline. Re-running the same cell (same resolved ``title``)
        replaces that widget's existing sidecar panel rather than opening
        another one. Requires the ``sidecar`` package.
    title : str, optional
        Sidecar panel title. Defaults to ``data.name`` if ``data`` is an
        ``escape.Array`` with one set, else ``name``.
    """

    # Shared across all instances/cells for the lifetime of the kernel; see
    # ``remember_last_roi_in_cell``.
    _remembered_rois = {}

    def __init__(
        self,
        data,
        rois={},
        callbacks_changeanyroi=[],
        name="RoiSelector",
        fig=None,
        ax=None,
        convert_rois_to_int="inner",
        remember_last_roi_in_cell=False,
        detached=False,
        title=None,
    ):
        # super().__init__(layout=widgets.Layout(flex_flow="row wrap"))
        super().__init__()
        self.data = data
        self.name = name
        self._external_ax = ax is not None
        self.roi_selectors = []
        self._tabs_rois = widgets.Tab()

        if convert_rois_to_int not in ("inner", "outer", "round", False, None):
            raise ValueError(
                "convert_rois_to_int must be one of 'inner', 'outer', 'round', or False"
            )
        self.convert_rois_to_int_mode = convert_rois_to_int or False

        self.remember_last_roi_in_cell = remember_last_roi_in_cell
        self._remember_key = None
        if remember_last_roi_in_cell:
            self._remember_key = (
                _auto_cell_name("roiselector") or "__no_cell__",
            ) + self._data_content_key(data)
            remembered = type(self)._remembered_rois.get(self._remember_key, {})
            rois = {**remembered, **rois}

        def f(x):
            self.visi_ind = x["new"]
            try:
                self.set_roi_selection_active(i=self.visi_ind)
                self.set_roi_selection_visible(i=self.visi_ind)
            except IndexError:
                pass

        self._tabs_rois.observe(
            f,
            names="selected_index",
        )

        self._select_buttons = []
        self.debug = widgets.Output()

        self._add_roi_button = widgets.Button(
            description="Add roi",
            disabled=False,
            button_style="primary",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Click me",
            # icon='check' # (FontAwesome names without the `fa-` prefix)
        )

        self._cmap_range = widgets.FloatRangeSlider(
            value=[self.data.min(), self.data.max()],
            description="Colormap range",
            min=self.data.min(),
            max=self.data.max(),
            step=ptp(data) / 5000,
            continuous_update=False,
            disabled=False,
        )

        self._add_roi_button.on_click(self.add_roi)
        self._roi_titles = []

        self.axs_rois = []
        self.figs_rois = []
        self.create_data_plot(fig=fig, ax=ax)

        controls = widgets.HBox([self._clim_slider, self._add_roi_button])
        if self._external_ax:
            # The image is already shown by the caller (e.g. StackViewer's own
            # main plot) -- no separate preview to display here.
            self.children = [controls, self._tabs_rois]
        else:
            self.children = [
                widgets.VBox([controls, self._output_data]),
                self._tabs_rois,
            ]
        self.layout = make_box_layout()
        callbacks_changeanyroi = list(callbacks_changeanyroi)
        if self.convert_rois_to_int_mode or self.remember_last_roi_in_cell:
            callbacks_changeanyroi = [self._on_any_roi_changed] + callbacks_changeanyroi
        self.callbacks_changeanyroi = [
            (lambda: tc(self)) for tc in callbacks_changeanyroi
        ]
        self.result = None  # dummy variable where callbacks can write their results to.
        for roititle, roiextents in rois.items():
            self.add_roi()
            # self.set_roi_selection_active(len(self.roi_selectors) - 1)
            self._roi_titles[-1].value = roititle
            self.roi_selectors[-1].selector.extents = tuple(roiextents)
            self.roi_selectors[-1].callbacks_changeroi = [
                (lambda: tc(self)) for tc in callbacks_changeanyroi
            ]

            self.roi_selectors[-1].line_select_callback(999, 999)

        array_name = getattr(data, "name", None)
        resolved_title = title or array_name or name
        _close_sidecar(name)
        if detached:
            _open_sidecar(name, resolved_title, lambda: display(self))
            _suppress_inline_redisplay(self)

    @property
    def rois(self):
        o = {}
        for tn, ts in zip(self._roi_titles, self.roi_selectors):
            o[tn.value] = (ts.xmin, ts.xmax, ts.ymin, ts.ymax)
        return o

    def _on_any_roi_changed(self, *_args):
        """Hooked into every ROI's change callbacks (see ``__init__``) to
        apply ``convert_rois_to_int``/``remember_last_roi_in_cell`` as
        configured, whenever any ROI is added or moved."""
        if self.convert_rois_to_int_mode:
            kwargs = {"round": False, "outer": False, "inner": False}
            kwargs[self.convert_rois_to_int_mode] = True
            self.convert_rois_to_int(**kwargs)
        if self.remember_last_roi_in_cell and self._remember_key is not None:
            type(self)._remembered_rois[self._remember_key] = dict(self.rois)

    @staticmethod
    def _data_content_key(data):
        arr = np.asarray(data)
        digest = hashlib.sha1(arr.tobytes()).hexdigest()
        return (arr.shape, str(arr.dtype), digest)

    def convert_rois_to_int(self, round=False, outer=False, inner=False):
        """Snap every current ROI's extents to integer pixel bounds, in place.

        Exactly one of the three keywords selects the rounding strategy:

        round
            Round each bound independently to the nearest integer.
        outer
            Grow the ROI outward (floor xmin/ymin, ceil xmax/ymax) so the
            integer region is a superset of the original.
        inner
            Shrink the ROI inward (ceil xmin/ymin, floor xmax/ymax) so the
            integer region is a subset of the original.
        """
        if sum((round, outer, inner)) != 1:
            raise ValueError("exactly one of round/outer/inner must be True")
        for ts in self.roi_selectors:
            xmin, xmax, ymin, ymax = ts.xmin, ts.xmax, ts.ymin, ts.ymax
            if round:
                new_extents = tuple(
                    float(np.round(v)) for v in (xmin, xmax, ymin, ymax)
                )
            elif outer:
                new_extents = (
                    float(np.floor(xmin)),
                    float(np.ceil(xmax)),
                    float(np.floor(ymin)),
                    float(np.ceil(ymax)),
                )
            else:  # inner
                new_extents = (
                    float(np.ceil(xmin)),
                    float(np.floor(xmax)),
                    float(np.ceil(ymin)),
                    float(np.floor(ymax)),
                )
            if new_extents != (xmin, xmax, ymin, ymax):
                ts.selector.extents = new_extents
                ts.line_select_callback(999, 999)

    def create_data_plot(self, fig=None, ax=None):
        if ax is not None:
            # Reuse an existing image (already displayed by the caller) as the
            # ROI-drag target, instead of creating a second copy of it.
            self._output_data = None
            self.fig_data = fig if fig is not None else ax.figure
            self.ax_data = ax
        else:
            self._output_data = widgets.Output()
            with self._output_data:
                plt.close(self.name)
                new_fig, new_ax = plt.subplots(
                    num=self.name,
                    constrained_layout=True,
                    figsize=[4, 4],
                )
                self.fig_data = new_fig
                self.ax_data = new_ax
                ih = self.ax_data.imshow(self.data)
                plt.colorbar(mappable=ih)
                # plt.tight_layout()
                plt.show(self.fig_data)

        mn = np.nanmin(self.data)
        mx = np.nanmax(self.data)
        ptp = mx - mn

        self._clim_slider = widgets.FloatRangeSlider(
            value=[mn, mx],
            min=mn,
            max=mx,
            step=ptp / 200,
            description="Colormap range:",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            # readout_format='.1f',
        )
        # widgets.interact(lambda val:self.set_clim(*val),val=self._clim_slider)
        self._clim_slider.observe(lambda val: self.set_clim(*val["new"]), names="value")

    def set_clim(self, vmin, vmax):
        """Update colormap limits for all plots."""
        # Update main data plot
        main_image = self.ax_data.get_images()[0]
        main_image.set_clim(vmin, vmax)

        # Update all ROI plots
        for ax in self.axs_rois:
            if ax.get_images():
                roi_image = ax.get_images()[0]
                roi_image.set_clim(vmin, vmax)

        # Force redraw of all figures
        plt.draw()

    def synchronize_colormaps(self):
        """Manually synchronize all colormap limits to match the main plot."""
        if self.ax_data.get_images():
            vmin, vmax = self.ax_data.get_images()[0].get_clim()
            self.set_clim(vmin, vmax)

    def add_roi_plot(self):
        ti = len(self.roi_selectors)
        op = widgets.Output()
        with op:
            tfig = plt.figure(constrained_layout=True, figsize=[5, 5])
            self.axs_rois.append(tfig.add_subplot())
            plt.show(tfig)

        # self._select_buttons.append(
        #     widgets.Button(
        #         description="Select",
        #         disabled=False,
        #         button_style="primary",  # 'success', 'info', 'warning', 'danger' or ''
        #         tooltip="Click me",
        #         # icon='check' # (FontAwesome names without the `fa-` prefix)
        #     )
        # )
        self._roi_titles.append(
            widgets.Text(
                value=f"roi{ti}",
                placeholder="ROI title",
                description="Name:",
                disabled=False,
            )
        )

        self._tabs_rois.children += (
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            # self._select_buttons[-1],
                            self._roi_titles[-1]
                        ]
                    ),
                    op,
                ]
            ),
        )

        # self._tabs_rois.set_title(len(self._tabs_rois.children)-1,self._roi_title_input.value)
        def update_tab_title(x):
            self._tabs_rois.set_title(ti, x["new"])
            for callback in self.callbacks_changeanyroi:
                callback()

        self._tabs_rois.set_title(ti, self._roi_titles[ti].value)
        self._roi_titles[ti].observe(update_tab_title, names="value")

    def set_roi_selection_active(self, i=None):
        for rs in self.roi_selectors:
            rs.selector.set_active(False)
        if not i == None:
            self.roi_selectors[i].selector.set_active(True)
        # self.set_roi_selection_visible(i)
        # self.fig_data.canvas.draw()

    def set_roi_selection_visible(self, i=None):
        for rs in self.roi_selectors:
            rs.selector.set_visible(False)
        if not i == None:
            self.roi_selectors[i].selector.set_visible(True)
        # self.fig_data.canvas.draw()

    def get_roi_selection_active(self):
        o = []
        for rs in self.roi_selectors:
            o.append(rs.selector.active)
        return o

    def add_roi(self, *args):
        # self.roi_selectors.append('test')
        ti = len(self.roi_selectors)
        # with self._output_data:
        #     print(ti)
        #     print(self._roi_titles)
        self.add_roi_plot()
        # with self._output_data:
        #     print(ti)
        #     print(self._roi_titles)
        #     print("got here")
        self.roi_selectors.append(
            RectangleSelectNB(
                fig=self.fig_data,
                ax=self.ax_data,
                ax_roi=self.axs_rois[-1],
                callbacks_changeroi=self.callbacks_changeanyroi,
            )
        )
        # self._select_buttons[-1].on_click(lambda dum: self.set_roi_selection_active(ti))

        self._tabs_rois.set_trait("selected_index", len(self.roi_selectors) - 1)
        self.set_roi_selection_active(i=ti)
        self.set_roi_selection_visible(i=ti)

    def update_data(self, data):
        self.data = data
        cmin = self._clim_slider.min
        cmax = self._clim_slider.max
        cval = self._clim_slider.value
        self.ax_data.get_images()[0].set_array(data)
        for ax in self.axs_rois:
            ax.get_images()[0].set_array(data)

        for sel in self.roi_selectors:
            sel.line_select_callback(1, 2)

        self._clim_slider.set_trait("min", min(np.nanmin(self.data), cmin))
        self._clim_slider.set_trait("max", max(np.nanmax(self.data), cmax))
        self._clim_slider.set_trait("value", cval)


import ipywidgets as widgets
from time import sleep
from threading import Thread


class StepViewer(widgets.VBox):
    """Slider viewer over an ``escape.Array``'s scan steps, filled in the
    background.

    Each step's mean image is computed (over ``data_selection`` events, via
    dask) in a background thread, in a randomly shuffled order so the
    "steps done so far" don't just creep up monotonically -- the slider's
    range grows to include a step as soon as its mean is ready.

    Parameters
    ----------
    array : escape.Array
        Array with scan steps to view; each step's mean is shown at its
        slider position.
    data_selection : slice
        Event selection (within each step) averaged for that step's image.
    update_rate : float
        Seconds between polls of the background computation for newly
        finished steps.
    figname : str
        Name of the created matplotlib figure.
    detached : bool
        If ``True``, show this widget in a JupyterLab Sidecar panel instead
        of inline. Re-running with the same ``figname`` replaces that
        widget's existing sidecar panel rather than opening another one.
        Requires the ``sidecar`` package.
    title : str, optional
        Sidecar panel title. Defaults to ``array.name`` if set, else
        ``figname``.
    """

    def __init__(
        self,
        array,
        data_selection=slice(None, 100),
        update_rate=1,
        figname="StepViewer",
        detached=False,
        title=None,
    ):
        super().__init__()
        self.array = array
        self.data_plot = len(self.array.scan) * [np.nan * np.ones(array.shape[1:])]
        self.attr_data_plot = len(self.array.scan) * [None]

        self.output = widgets.Output()
        with self.output:
            plt.close(figname)
            tfig = plt.figure(figname, constrained_layout=True)
            self.ax = tfig.add_subplot()
            plt.show(tfig)

        self.step_order = list(range(len(self.array.scan)))
        np.random.RandomState(0).shuffle(self.step_order)
        self.data_queue = [
            self.array.scan[i][data_selection].nanmean(axis=0).persist()
            for i in self.step_order
        ]
        self.data_plot_done = set()
        steps_done = sorted(list(self.data_plot_done))
        if not steps_done:
            steps_done = [self.step_order[0]]
        self.selector = widgets.SelectionSlider(
            options=steps_done,
            value=self.step_order[0],
            description="Step number",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
        )

        self.step_text = widgets.Output()

        self.update(self.step_order[0])

        self.children = [
            self.output,
            widgets.HBox([self.selector, self.step_text]),
        ]

        array_name = getattr(array, "name", None)
        resolved_title = title or array_name or figname
        _close_sidecar(figname)
        if detached:
            _open_sidecar(figname, resolved_title, lambda: display(self))
            _suppress_inline_redisplay(self)

        self.selector.observe(lambda d: self.update(d["new"]), names="value")

        # @debug.capture(clear_output=False)
        self.uded = 0

        def update_threadfunc():
            while len(self.queue_done()) < len(self.array.scan):
                self.update_data()
                self.update_selector()
                self.uded += 1
                sleep(update_rate)
            self.update_data()
            self.update_selector()
            self.uded += 1

        self.update_thread = Thread(target=update_threadfunc)
        self.update_thread.start()

    def update(self, ix):
        self.ax.cla()
        self.ax.imshow(self.data_plot[ix])
        with self.step_text:
            self.step_text.clear_output()
            print(str(self.array.scan.par_steps.T[ix]))

    def update_data(self):
        for n in self.queue_done():
            if n in self.data_plot_done:
                continue
            else:
                self.data_plot[self.step_order[n]] = self.data_queue[n].compute()
                self.data_plot_done.add(n)

    def update_selector(self):
        value = self.selector.value
        steps_done = sorted(list(self.data_plot_done))
        if not steps_done:
            steps_done = [self.step_order[0]]
        self.selector.set_trait("options", tuple(steps_done))
        self.selector.set_trait("value", value)

    def queue_done(self):
        return np.asarray(
            [list(tmp.dask.values())[0].done() for tmp in self.data_queue]
        ).nonzero()[0]


_AUTO_NAME = object()  # sentinel: distinguishes "not given" from an explicit num=None


def _auto_cell_name(prefix="cell"):
    """Best-effort name that stays the same across re-runs of one notebook
    cell but differs between cells, so auto-named figures don't overwrite
    each other -- unlike ``execution_count``, which increments on every run.

    Prefers the frontend's persistent per-cell id (JupyterLab/Notebook 7 and
    VS Code notebooks report one; it only changes if the cell is deleted and
    recreated). Falls back to a hash of the executing cell's source, which
    is available in any IPython shell (e.g. a plain terminal) even without a
    notebook frontend reporting a cell id -- note this means editing the
    cell's code changes the name. Returns None if there's no IPython session
    at all (e.g. a plain script), so callers can use their own default.
    """
    ip = get_ipython()
    if ip is None:
        return None
    parent = ip.get_parent() or {}
    metadata = parent.get("metadata", {}) or {}
    cell_id = metadata.get("cellId") or metadata.get("cell_id")
    if not cell_id and isinstance(metadata.get("vscode"), dict):
        cell_id = metadata["vscode"].get("cellId")
    if cell_id:
        return f"{prefix}-{cell_id}"

    code = (parent.get("content", {}) or {}).get("code")
    if code:
        digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}-{digest}"

    return None


_SIDECARS = (
    {}
)  # identity key -> live Sidecar, so a repeat call replaces instead of stacking


def _close_sidecar(key):
    """Close and forget any sidecar previously opened under ``key``, if one exists."""
    old = _SIDECARS.pop(key, None)
    if old is not None:
        old.close()


def _open_sidecar(key, title, show_fn):
    """Open a fresh Sidecar titled ``title``, run ``show_fn()`` with an
    ``Output`` widget (already displayed inside the sidecar) as the active
    display target, and remember it under ``key`` -- the next call for the
    same key (see :func:`_close_sidecar`) replaces it instead of piling up
    panels."""
    if Sidecar is None:
        raise ImportError(
            "detached=True needs the 'sidecar' package and the JupyterLab "
            "sidecar extension installed (pip install sidecar)."
        )
    sc = Sidecar(title=title, anchor="split-right")
    out = widgets.Output()
    with sc:
        display(out)
    with out:
        show_fn()
    _SIDECARS[key] = sc


def _suppress_inline_redisplay(obj):
    """Make ``display(obj)``/a bare trailing expression a no-op for ``obj``,
    so a widget already shown in a sidecar doesn't *also* render inline if
    the caller leaves it as a cell's last expression."""
    obj._ipython_display_ = lambda: None


def _detect_plot_backend():
    """"qt", "ipympl", or ``None`` (no toolbar worth attaching to)."""
    backend = plt.get_backend().lower()
    if "qt" in backend:
        return "qt"
    if "ipympl" in backend:
        return "ipympl"
    return None


def _track_active_axes(fig):
    """Remember whichever axes in ``fig`` was last clicked, in
    ``fig._escape_active_ax`` -- ``plt.gca()`` doesn't reliably reflect this
    for a multi-axes figure (it tracks the last axes *created*, not clicked),
    which the fit button (see :func:`attach_fit_button`) needs to know which
    of possibly several subplots to act on."""
    if getattr(fig, "_escape_active_ax_cid", None) is not None:
        return

    def _on_click(event):
        if event.inaxes is not None:
            fig._escape_active_ax = event.inaxes

    fig._escape_active_ax_cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    fig._escape_active_ax = fig.axes[0] if fig.axes else None


def _get_active_axes(fig):
    ax = getattr(fig, "_escape_active_ax", None)
    if ax is not None and ax in fig.axes:
        return ax
    return fig.axes[0] if fig.axes else None


def _run_fit_button(fig):
    """The Fit toolbar button's click handler, shared by the Qt/ipympl
    attachments below. Importing ``escape.fit_gui`` (and so ``lmfit``) is
    deferred to here -- the click -- rather than done at attach time, so
    attaching the button costs nothing up front even without lmfit
    installed."""
    ax = _get_active_axes(fig)
    if ax is None:
        print("[escape] no axes to fit in this figure.")
        return
    try:
        from escape.fit_gui import AxesFitter
    except ImportError as e:
        print(f"[escape] the Fit button needs the optional 'lmfit' dependency: {e}")
        return
    AxesFitter(ax)


def _attach_fit_button_qt(fig):
    toolbar = getattr(fig.canvas.manager, "toolbar", None)
    if toolbar is None or not hasattr(toolbar, "addAction"):
        return  # no toolbar (e.g. rcParams['toolbar'] == 'None') -- nothing to attach to

    icon = None
    try:
        import qtawesome as qta

        icon = qta.icon("mdi.chart-bell-curve")
    except Exception:
        pass

    def _on_click(checked=False):
        _run_fit_button(fig)

    toolbar.addSeparator()
    action = toolbar.addAction(icon, "Fit", _on_click) if icon is not None else toolbar.addAction("Fit", _on_click)
    action.setToolTip("Attach an interactive lmfit fitting panel to the active axes")


def _attach_fit_button_ipympl(fig):
    toolbar = getattr(fig.canvas, "toolbar", None)
    if toolbar is None or not hasattr(toolbar, "toolitems"):
        return

    def _on_click():
        _run_fit_button(fig)

    # a plain function assigned as an *instance* attribute stays unbound (no
    # implicit self) -- exactly the zero-arg callable handle_toolbar_button
    # looks up via getattr(toolbar, method_name)().
    toolbar.escape_fit_button = _on_click
    toolbar.toolitems = list(toolbar.toolitems) + [
        ("Fit", "Attach an interactive lmfit fitting panel to the active axes", "line-chart", "escape_fit_button")
    ]


def attach_fit_button(fig):
    """Attach a "Fit" button to ``fig``'s toolbar, opening an interactive
    lmfit panel (:func:`escape.fit_gui.AxesFitter`) on whichever of its axes
    was last clicked (the first axes, if none has been clicked yet) --
    Qt and ipympl (``%matplotlib widget``) backends only.

    A no-op, not an error, anywhere this doesn't apply: other backends
    (inline, plain ``Agg``, ...) have no interactive toolbar to attach to,
    and any failure while attaching (an unexpected toolbar shape, a
    ``rcParams['toolbar'] == 'None'`` figure, ...) is swallowed with a
    printed note rather than raised -- this is a convenience layered onto
    figure creation and should never be the reason a plot call fails.
    Idempotent: attaching twice to the same figure is a no-op the second time.
    """
    if getattr(fig, "_escape_fit_attached", False):
        return
    try:
        backend = _detect_plot_backend()
        if backend is None:
            return
        _track_active_axes(fig)
        if backend == "qt":
            _attach_fit_button_qt(fig)
        else:
            _attach_fit_button_ipympl(fig)
        fig._escape_fit_attached = True
    except Exception as e:
        print(f"[escape] couldn't attach the Fit button: {e}")


def nfigure(num=_AUTO_NAME, *, detached=False, title=None, fit_button=True, **kwargs):
    """Like ``plt.figure``, but always starts from a clean figure of the
    given name -- any existing figure with that name is closed first,
    instead of being reused/added to (matplotlib's default when ``num``
    matches an existing figure).

    If ``num`` isn't given, it's derived automatically from the current
    notebook cell (see :func:`_auto_cell_name`): re-running a cell replaces
    that cell's own figure, while different cells get different figures and
    so don't overwrite each other -- without having to invent a unique name
    by hand. Pass ``num`` explicitly to opt out of this and manage the name
    yourself (e.g. to reuse one figure across cells).

    Parameters
    ----------
    num : str or int, optional
        Figure name/number, forwarded to ``plt.figure``. Auto-derived from
        the current cell if omitted.
    detached : bool
        If ``True``, show the figure in a JupyterLab Sidecar panel (a tab
        beside/below the notebook) instead of inline. Requires the
        ``sidecar`` package and its JupyterLab extension. Re-running the
        same cell (same ``num``) replaces that figure's existing sidecar
        panel rather than opening another one.
    title : str, optional
        Sidecar panel title. Defaults to ``str(num)`` -- the figure's own
        name effectively doubles as its title.
    fit_button : bool
        Attach a "Fit" toolbar button (see :func:`attach_fit_button`) --
        Qt/ipympl backends only, a harmless no-op elsewhere. Defaults to
        ``True``: attaching it costs nothing (no ``lmfit`` import) unless
        actually clicked.
    **kwargs
        Forwarded to ``plt.figure``.
    """
    if num is _AUTO_NAME:
        num = _auto_cell_name() or "no name"
    if num in plt.get_figlabels():
        Warning('Figure of name "{num}" exists and is closed.')
    plt.close(num)
    _close_sidecar(num)
    fig = plt.figure(num=num, **kwargs)
    if fit_button:
        attach_fit_button(fig)
    if detached:
        _open_sidecar(num, title or str(num), lambda: plt.show(fig))
    return fig


def nsubplots(
    nrows=1, ncols=1, *, num=_AUTO_NAME, detached=False, title=None, fit_button=True, **kwargs
):
    """Like ``plt.subplots``, but always starts from a clean figure of the
    given name (see :func:`nfigure` for why/how ``num`` is auto-derived when
    omitted).

    Parameters
    ----------
    nrows, ncols : int
        Grid shape, forwarded to ``plt.subplots``.
    num : str or int, optional
        Figure name/number, forwarded to ``plt.subplots``. Auto-derived from
        the current cell if omitted.
    detached : bool
        Show the figure in a JupyterLab Sidecar panel instead of inline
        (see :func:`nfigure`).
    title : str, optional
        Sidecar panel title. Defaults to ``str(num)``.
    fit_button : bool
        Attach a "Fit" toolbar button (see :func:`attach_fit_button`) to the
        figure. Defaults to ``True`` (see :func:`nfigure`).
    **kwargs
        Forwarded to ``plt.subplots``.
    """
    if num is _AUTO_NAME:
        num = _auto_cell_name() or "no name"
    if num in plt.get_figlabels():
        Warning('Figure of name "{num}" exists and is closed.')
    plt.close(num)
    _close_sidecar(num)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, num=num, **kwargs)
    if fit_button:
        attach_fit_button(fig)
    if detached:
        _open_sidecar(num, title or str(num), lambda: plt.show(fig))
    return fig, ax


def nsubplot_mosaic(*args, num=_AUTO_NAME, detached=False, title=None, fit_button=True, **kwargs):
    """Like ``plt.subplot_mosaic``, but always starts from a clean figure of
    the given name (see :func:`nfigure` for why/how ``num`` is auto-derived
    when omitted).

    Parameters
    ----------
    *args
        Forwarded to ``plt.subplot_mosaic`` (the mosaic layout, typically).
    num : str or int, optional
        Figure name/number, forwarded to ``plt.subplot_mosaic``.
        Auto-derived from the current cell if omitted.
    detached : bool
        Show the figure in a JupyterLab Sidecar panel instead of inline
        (see :func:`nfigure`).
    title : str, optional
        Sidecar panel title. Defaults to ``str(num)``.
    fit_button : bool
        Attach a "Fit" toolbar button (see :func:`attach_fit_button`) to the
        figure. Defaults to ``True`` (see :func:`nfigure`).
    **kwargs
        Forwarded to ``plt.subplot_mosaic``.
    """
    if num is _AUTO_NAME:
        num = _auto_cell_name() or "no name"
    if num in plt.get_figlabels():
        Warning('Figure of name "{num}" exists and is closed.')
    plt.close(num)
    _close_sidecar(num)
    fig, axd = plt.subplot_mosaic(*args, num=num, **kwargs)
    if fit_button:
        attach_fit_button(fig)
    if detached:
        _open_sidecar(num, title or str(num), lambda: plt.show(fig))
    return fig, axd


class StepViewerP(widgets.VBox):
    """Like :class:`StepViewer`, but displaying into a caller-supplied widget
    instead of creating its own plain ``imshow``.

    Each step's mean image is computed in the background (see
    :class:`StepViewer`); on slider changes, the resulting image is pushed
    into ``wid`` via ``wid.update_data(image)`` rather than drawn onto an
    axes owned by this viewer. This lets the display be something richer
    than a bare image -- e.g. a :class:`MultipleRoiSelector`, so ROI
    selection stays live as the viewed step changes.

    Parameters
    ----------
    array : escape.Array
        Array with scan steps to view; each step's mean is shown at its
        slider position.
    wid : ipywidgets.Widget
        Display widget with an ``update_data(image)`` method, shown above
        the step slider.
    data_selection : slice
        Event selection (within each step) averaged for that step's image.
    update_rate : float
        Seconds between polls of the background computation for newly
        finished steps.
    detached : bool
        If ``True``, show this widget in a JupyterLab Sidecar panel instead
        of inline. Re-running with the same resolved ``title`` replaces
        that widget's existing sidecar panel rather than opening another
        one. Requires the ``sidecar`` package.
    title : str, optional
        Sidecar panel title. Defaults to ``array.name`` if set, else
        ``"StepViewerP"``.
    """

    def __init__(
        self,
        array,
        wid,
        data_selection=slice(None, 100),
        update_rate=1,
        detached=False,
        title=None,
    ):
        super().__init__()
        self.array = array
        self.data_plot = len(self.array.scan) * [np.nan * np.ones(array.shape[1:])]
        self.attr_data_plot = len(self.array.scan) * [None]

        self.output = wid

        # self.output = widgets.Output()
        # with self.output:
        #     plt.close(figname)
        #     tfig = plt.figure(figname, constrained_layout=True)
        #     self.ax = tfig.add_subplot()
        #     plt.show(tfig)
        # Creating a random step order, this is about to become subject to user input preference on computation.
        self.step_order = list(range(len(self.array.scan)))
        np.random.RandomState(0).shuffle(self.step_order)

        # !starting calculation of average DATA in custom order!

        self.data_queue = [
            self.array.scan[i][data_selection].nanmean(axis=0).persist()
            for i in self.step_order
        ]

        self.data_plot_done = set()
        steps_done = sorted(list(self.data_plot_done))
        if not steps_done:
            steps_done = [self.step_order[0]]
        self.selector = widgets.SelectionSlider(
            options=steps_done,
            value=self.step_order[0],
            description="Step number",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
        )

        self.step_text = widgets.Output()

        # self.update(self.step_order[0])

        self.children = [
            self.output,
            widgets.HBox([self.selector, self.step_text]),
        ]

        array_name = getattr(array, "name", None)
        resolved_title = title or array_name or "StepViewerP"
        _close_sidecar(resolved_title)
        if detached:
            _open_sidecar(resolved_title, resolved_title, lambda: display(self))
            _suppress_inline_redisplay(self)

        self.selector.observe(lambda d: self.update(d["new"]), names="value")

        # @debug.capture(clear_output=False)
        self.uded = 0
        self.result = None

        def update_threadfunc():
            while len(self.queue_done()) < len(self.array.scan):
                self.update_data()
                self.update_selector()
                self.uded += 1
                sleep(update_rate)
            self.update_data()
            self.update_selector()
            self.uded += 1
            self.tito = [len(self.queue_done()), len(self.array.scan)]

        self.update_thread = Thread(target=update_threadfunc)
        self.update_thread.start()

    def update(self, ix):
        self.output.update_data(self.data_plot[ix])
        with self.step_text:
            self.step_text.clear_output()
            print(str(self.array.scan.par_steps.T[ix]))

    def update_data(self):
        for n in self.queue_done():
            if n in self.data_plot_done:
                continue
            else:
                self.data_plot[self.step_order[n]] = self.data_queue[n].compute()
                self.data_plot_done.add(n)

    def update_selector(self):
        value = self.selector.value
        steps_done = sorted([self.step_order[n] for n in list(self.data_plot_done)])
        if not steps_done:
            steps_done = [self.step_order[0]]
        self.selector.set_trait("options", tuple(steps_done))
        if value in steps_done:
            self.selector.set_trait("value", value)
        else:
            self.selector.set_trait("value", steps_done[0])

    def queue_done(self):
        dn = []
        for tmp in self.data_queue:
            element = list(tmp.dask.values())[0]
            if hasattr(
                element, "done"
            ):  # in this case the persist above is in background, i.e. the scheduler is distributed.
                dn.append(element.done())
            else:
                dn.append(True)  # assuming computation has happened!

        return np.asarray(dn).nonzero()[0]


_NORM_KINDS = ("log", "symlog", "diverging", "two_slope", "power")


def _build_norm(kind, kwargs, vmin, vmax):
    """Build a `matplotlib.colors.Normalize` for one of `StackViewer`'s
    ``norm=`` kinds, given a (possibly missing/degenerate) data range.

    Parameters
    ----------
    kind : None, str, or matplotlib.colors.Normalize
        ``None`` -- no explicit norm (caller falls back to imshow/pcolormesh's
        own default). A `Normalize` instance is returned as-is (escape hatch
        for anything not covered below, e.g. `BoundaryNorm`). Otherwise one
        of ``"log"``, ``"symlog"``, ``"diverging"`` (`CenteredNorm`),
        ``"two_slope"`` (`TwoSlopeNorm`, independent +/- ranges around a
        center) or ``"power"`` (`PowerNorm`, gamma scaling).
    kwargs : dict or None
        Extra keyword arguments forwarded to the norm's constructor (e.g.
        ``{"gamma": 0.5}`` for "power", ``{"vcenter": 1.0}`` for "diverging"
        or "two_slope", ``{"linthresh": 0.1}`` for "symlog").
    vmin, vmax : float or None
        Data range to seed the norm with when `kwargs` doesn't already
        specify one. May be ``None``/degenerate (e.g. before any real frame
        has loaded) -- safe fallback bounds are used in that case; the first
        real frame's autoscale corrects them immediately after.
    """
    if kind is None:
        return None
    if isinstance(kind, mcolors.Normalize):
        return kind
    if kind not in _NORM_KINDS:
        raise ValueError(
            f"Unknown norm {kind!r}; expected one of {_NORM_KINDS}, a Normalize instance, or None."
        )

    kwargs = dict(kwargs or {})
    if (
        vmin is None
        or vmax is None
        or not (np.isfinite(vmin) and np.isfinite(vmax))
        or vmin == vmax
    ):
        vmin, vmax = (
            0.0,
            1.0,
        )  # placeholder bounds; corrected by autoscale on the first real frame

    if kind == "log":
        kwargs.setdefault("vmin", max(vmin, 1e-3))
        kwargs.setdefault("vmax", max(vmax, kwargs["vmin"] * 10))
        return mcolors.LogNorm(**kwargs)
    if kind == "symlog":
        kwargs.setdefault("linthresh", 1.0)
        kwargs.setdefault("linscale", 1.0)
        kwargs.setdefault("base", 10)
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)
        return mcolors.SymLogNorm(**kwargs)
    if kind == "diverging":
        kwargs.setdefault("vcenter", 0.0)
        return mcolors.CenteredNorm(**kwargs)
    if kind == "two_slope":
        vcenter = kwargs.setdefault("vcenter", 0.0)
        kwargs.setdefault("vmin", min(vmin, vcenter - 1e-6))
        kwargs.setdefault("vmax", max(vmax, vcenter + 1e-6))
        return mcolors.TwoSlopeNorm(**kwargs)
    if kind == "power":
        kwargs.setdefault("gamma", 0.5)
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)
        return mcolors.PowerNorm(**kwargs)


class RoiRegion:
    """One rectangular region of interest on a shared image axes.

    Owns a ``matplotlib.widgets.RectangleSelector`` attached directly to
    ``ax`` (so the rectangle is drawn on the image itself, not a copy of it)
    and a small preview axes showing just the pixels inside the rectangle.
    Redraws its preview -- and calls ``on_change(self)``, if given -- both
    when the selector fires its own callback (drag released) and on every
    intermediate step of an in-progress drag, since matplotlib already
    updates ``.extents`` live as the mouse moves.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes already showing the image to select a region on.
    preview_ax : matplotlib.axes.Axes
        Axes to draw this region's cropped preview into.
    main_mappable : matplotlib.cm.ScalarMappable
        The image/mesh artist shown on ``ax`` (an ``AxesImage`` from
        ``imshow``, or a ``QuadMesh`` from ``pcolormesh``) -- used instead of
        ``ax.get_images()[0]`` so this also works when the main plot is a
        mesh (e.g. ``StackViewer(xdata=..., ydata=...)``), which
        ``get_images()`` doesn't see.
    name : str
        Label for this region (tab title, trend-plot legend entry).
    color : str, optional
        Colour for this region's rectangle outline, used consistently for
        its rectangle, preview border and trend-plot line.
    cmap : str, optional
        Colormap for this region's preview (should match the main image's).
    make_norm : callable(vmin, vmax) -> Normalize, optional
        Builds a fresh norm instance for the preview's *initial* display,
        matching the main image's norm type/kwargs (e.g. log, diverging).
        An independent instance, not shared with the main image -- see
        :meth:`set_clim` for how the two stay in sync afterwards.
    xdata, ydata : array-like, optional
        Physical (non-pixel-index) coordinates for the main image's columns
        /rows, if any -- so this region's rectangle (drawn in those same
        coordinates) is sliced out of the underlying array correctly.
        Assumed monotonic. ``None`` means plain pixel indices.
    on_change : callable(RoiRegion), optional
        Called after every redraw (selection made, dragged, or refreshed
        because the underlying image data changed).
    """

    def __init__(
        self,
        ax,
        preview_ax,
        main_mappable,
        name="roi",
        color=None,
        cmap=None,
        make_norm=None,
        xdata=None,
        ydata=None,
        on_change=None,
    ):
        self.ax = ax
        self.preview_ax = preview_ax
        self.preview_fig = preview_ax.figure
        self.main_mappable = main_mappable
        self.name = name
        self.color = color
        self.cmap = cmap
        self.make_norm = make_norm
        self.xdata = None if xdata is None else np.asarray(xdata)
        self.ydata = None if ydata is None else np.asarray(ydata)
        self.on_change = on_change

        props = (
            dict(edgecolor=color, facecolor=color or "red", alpha=0.15)
            if color
            else None
        )
        self.selector = RectangleSelector(
            ax,
            self._on_select,
            useblit=False,
            button=[1, 3],  # exclude the middle button
            minspanx=5,
            minspany=5,
            spancoords="data",
            interactive=True,
            props=props,
        )
        self.selector.set_active(False)
        self._last_drag_extents = None
        ax.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)

    @property
    def extents(self):
        """(xmin, xmax, ymin, ymax) in the shared axes' data coordinates --
        pixel indices, or physical units if xdata/ydata were given."""
        return self.selector.extents

    @extents.setter
    def extents(self, value):
        self.selector.extents = value

    @property
    def active(self):
        return self.selector.active

    def set_active(self, active):
        self.selector.set_active(active)

    def set_visible(self, visible):
        self.selector.set_visible(visible)

    def _slice(self):
        xmin, xmax, ymin, ymax = self.extents
        if self.xdata is None:
            ix0, ix1 = int(round(xmin)), int(round(xmax))
        else:
            ix0, ix1 = np.searchsorted(self.xdata, [xmin, xmax])
        if self.ydata is None:
            iy0, iy1 = int(round(ymin)), int(round(ymax))
        else:
            iy0, iy1 = np.searchsorted(self.ydata, [ymin, ymax])
        return slice(iy0, iy1), slice(ix0, ix1)

    def data(self, image=None):
        """Sub-array covered by this region, of ``image`` (default: whatever
        is currently shown on the shared main axes)."""
        if image is None:
            image = self.main_mappable.get_array()
        return np.asarray(image)[self._slice()]

    def mean(self, image=None):
        return float(np.nanmean(self.data(image)))

    def set_clim(self, vmin, vmax):
        images = self.preview_ax.get_images()
        if images:
            images[0].set_clim(vmin, vmax)
            self.preview_fig.canvas.draw_idle()

    def refresh(self, image=None):
        """Redraw the preview from ``image`` (or the shared main image) and
        notify ``on_change``."""
        data = self.data(image)
        images = self.preview_ax.get_images()
        if images:
            images[0].set_data(data)
            images[0].autoscale()
        else:
            vmin, vmax = self.main_mappable.get_clim()
            norm = self.make_norm(vmin, vmax) if self.make_norm else None
            kwargs = dict(cmap=self.cmap)
            if norm is not None:
                kwargs["norm"] = norm
            else:
                kwargs["vmin"], kwargs["vmax"] = vmin, vmax
            self.preview_ax.imshow(data, **kwargs)
        self.preview_fig.canvas.draw_idle()
        if self.on_change:
            self.on_change(self)

    def _on_select(self, eclick, erelease):
        self.refresh()

    def _on_motion(self, event):
        # RectangleSelector keeps `_eventpress` truthy while a drag is in
        # progress and already updates `.extents` live as the mouse moves;
        # re-run refresh() on every such update instead of only on release,
        # so the preview/trend track the rectangle continuously.
        if not self.active or getattr(self.selector, "_eventpress", None) is None:
            return
        if self.extents == self._last_drag_extents:
            return
        self._last_drag_extents = self.extents
        self.refresh()


class RoiPanel(widgets.VBox):
    """UI for managing a set of `RoiRegion` objects on a shared image axes.

    One "Add ROI" button plus a tab per region (each showing that region's
    cropped preview and an editable name), and a colormap-range slider kept
    in sync with the regions' previews. Selecting a tab makes only that
    region's rectangle active/visible on the main image, so overlapping
    regions don't fight each other for mouse events.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes already showing the image to select regions on.
    main_mappable : matplotlib.cm.ScalarMappable
        The image/mesh artist shown on ``ax`` (see `RoiRegion`).
    cmap : str, optional
        Colormap for region previews (should match the main image's).
    make_norm : callable(vmin, vmax) -> Normalize, optional
        Builds a norm for each new region's preview matching the main
        image's (see `RoiRegion`).
    xdata, ydata : array-like, optional
        Physical coordinates for the main image's columns/rows, if any (see
        `RoiRegion`).
    on_regions_changed : callable(), optional
        Called whenever a region is added or its selection/data changes.
    on_clim_changed : callable(vmin, vmax), optional
        Called when the colormap-range slider is moved by the user (i.e.
        not as a result of :meth:`sync_clim`), so a caller can mirror the
        change elsewhere (e.g. the main image this panel doesn't own).
    preview_figsize : (float, float)
        Figure size for each region's preview plot.
    """

    _COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    def __init__(
        self,
        ax,
        main_mappable,
        cmap=None,
        make_norm=None,
        xdata=None,
        ydata=None,
        on_regions_changed=None,
        on_clim_changed=None,
        preview_figsize=(3.2, 3.2),
    ):
        super().__init__()
        self.ax = ax
        self.main_mappable = main_mappable
        self.cmap = cmap
        self.make_norm = make_norm
        self.xdata = None if xdata is None else np.asarray(xdata)
        self.ydata = None if ydata is None else np.asarray(ydata)
        self.on_regions_changed = on_regions_changed
        self.on_clim_changed = on_clim_changed
        self.preview_figsize = preview_figsize
        self.regions = []
        self._suppress_clim_callback = False

        self.tabs = widgets.Tab()
        self.tabs.observe(self._on_tab_selected, names="selected_index")

        add_button = widgets.Button(description="Add ROI", button_style="primary")
        add_button.on_click(lambda _: self.add_region())

        vmin, vmax = self.main_mappable.get_clim()
        self.clim_slider = widgets.FloatRangeSlider(
            value=[vmin, vmax],
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 200 or 1.0,
            description="Colormap range:",
            continuous_update=True,
        )
        self.clim_slider.observe(self._on_clim_slider_changed, names="value")

        # Controls + tabs stack in one column ("selector box"); once StackViewer
        # appends a trend plot (see StackViewer._ensure_roi_panel) it joins this
        # box as a flex-wrapping sibling, so it flows beside/under this column
        # instead of splitting the slider away from the tabs it belongs above.
        self._selector_box = widgets.VBox(
            [widgets.HBox([self.clim_slider, add_button]), self.tabs]
        )
        self.layout = widgets.Layout(
            display="flex", flex_flow="row wrap", align_items="flex-start"
        )
        self.children = [self._selector_box]

    def add_region(self, extents=None, name=None):
        """Add a new region. Defaults to a box covering the middle ~40% of the
        image (in each dimension) if ``extents`` isn't given, so the preview
        tab always shows something immediately instead of staying blank until
        the rectangle is first dragged."""
        idx = len(self.regions)
        name = name or f"roi{idx}"
        color = self._COLORS[idx % len(self._COLORS)]

        preview_output = widgets.Output()
        with preview_output:
            preview_fig, preview_ax = plt.subplots(
                figsize=self.preview_figsize, constrained_layout=True
            )
            plt.show(preview_fig)

        title = widgets.Text(value=name, description="Name:")
        title.observe(
            lambda change, i=idx: self._on_title_changed(i, change["new"]),
            names="value",
        )
        self.tabs.children = self.tabs.children + (
            widgets.VBox([title, preview_output]),
        )
        self.tabs.set_title(idx, name)

        region = RoiRegion(
            self.ax,
            preview_ax,
            self.main_mappable,
            name=name,
            color=color,
            cmap=self.cmap,
            make_norm=self.make_norm,
            xdata=self.xdata,
            ydata=self.ydata,
            on_change=self._on_region_changed,
        )
        region.extents = extents if extents is not None else self._default_extents()
        self.regions.append(region)

        self._activate_only(idx)
        self.tabs.selected_index = idx
        region.refresh()
        return region

    def _default_extents(self):
        """A centred box covering ~40% of the image in each dimension, in
        whatever coordinates the main axes actually uses (pixel indices, or
        physical xdata/ydata if given)."""
        if self.xdata is not None:
            x0, x1 = float(self.xdata.min()), float(self.xdata.max())
        else:
            x0, x1 = 0, self.main_mappable.get_array().shape[1]
        if self.ydata is not None:
            y0, y1 = float(self.ydata.min()), float(self.ydata.max())
        else:
            y0, y1 = 0, self.main_mappable.get_array().shape[0]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) * 0.2, (y1 - y0) * 0.2
        return (cx - hw, cx + hw, cy - hh, cy + hh)

    def sync_clim(self, vmin, vmax):
        """Externally push a new colormap range: slider position and every
        region's preview. Does not invoke ``on_clim_changed``."""
        self._suppress_clim_callback = True
        try:
            if vmin < self.clim_slider.min:
                self.clim_slider.min = vmin
            if vmax > self.clim_slider.max:
                self.clim_slider.max = vmax
            self.clim_slider.value = (vmin, vmax)
            self.set_clim(vmin, vmax)
        finally:
            self._suppress_clim_callback = False

    def set_clim(self, vmin, vmax):
        for region in self.regions:
            region.set_clim(vmin, vmax)

    def refresh_all(self, image=None):
        for region in self.regions:
            region.refresh(image)

    def close(self):
        for region in self.regions:
            plt.close(region.preview_fig)

    def _activate_only(self, idx):
        for i, region in enumerate(self.regions):
            region.set_active(i == idx)
            region.set_visible(i == idx)

    def _on_tab_selected(self, change):
        idx = change["new"]
        if isinstance(idx, int) and 0 <= idx < len(self.regions):
            self._activate_only(idx)

    def _on_title_changed(self, idx, new_name):
        self.regions[idx].name = new_name
        self.tabs.set_title(idx, new_name)
        if self.on_regions_changed:
            self.on_regions_changed()

    def _on_region_changed(self, region):
        if self.on_regions_changed:
            self.on_regions_changed()

    def _on_clim_slider_changed(self, change):
        if self._suppress_clim_callback:
            return
        vmin, vmax = change["new"]
        self.set_clim(vmin, vmax)
        if self.on_clim_changed:
            self.on_clim_changed(vmin, vmax)


class StackViewer(widgets.VBox):
    """Slider viewer for a 3-D image stack, with lazy dask loading.

    Pass either an ``escape.Array`` with scan steps (one image per step,
    each the mean over that step's events) or any 3-D array — dask,
    numpy, or a scan-less ``escape.Array`` (one image per frame, no
    averaging). Nothing is computed until a slider position is visited.
    For dask-backed step data, moving the slider first shows a cheap
    preview (mean of ``preview_frames`` events) and, if the slider stays
    put for ``refine_delay`` seconds, replaces it with the full-step
    mean computed in a background thread. Moving the slider again
    cancels any refine still in flight, so nothing stale gets drawn.

    Set ``rois=True`` to add a `RoiPanel` next to the image, with rectangles
    drawn directly on it (no separate copy of the image); each region's mean
    is tracked per visited index and plotted live in an adjacent trend plot
    (recomputed from cached frames whenever a region is added/moved, so
    revisiting steps isn't needed).

    By default the image is drawn with plain ``imshow`` (fast) on a pixel-index
    grid with linear color scaling, auto-rescaled to each frame's min/max --
    exactly the original behaviour. Pass ``xdata``/``ydata`` and/or ``norm``
    to opt into more, at some extra per-frame cost:

    * ``xdata``, ``ydata`` -- physical (possibly non-uniform) coordinates for
      the image's columns/rows, e.g. real detector positions instead of pixel
      indices. Switches internally to ``pcolormesh`` (via
      :func:`escape.utilities.plot2D`), since ``imshow`` only supports a
      uniform pixel grid.
    * ``norm`` -- one of ``"log"``, ``"symlog"``, ``"diverging"`` (centered,
      symmetric around ``vcenter``), ``"two_slope"`` (centered, but with
      independent +/- ranges), ``"power"`` (gamma scaling), or your own
      ``matplotlib.colors.Normalize`` instance. See ``norm_kwargs`` for
      per-kind options (``vcenter``, ``gamma``, ``linthresh``, ...).
    * ``autoscale`` -- ``"minmax"`` (default) or ``"percentile"`` (robust to
      hot pixels/cosmic rays; see ``autoscale_percentile``) to control how
      each new frame's color range is chosen; ``None``/``False`` freezes the
      range instead of rescaling every frame (respected always for a custom
      ``norm`` instance, which is never auto-rescaled).

    Parameters
    ----------
    array : escape.Array or array-like
        3-D image stack, ``(n_events, ny, nx)``.
    preview_frames : int
        Events averaged for the fast preview of a step (step mode only).
    refine_delay : float
        Seconds of slider inactivity before the full-step mean is computed.
    rois : bool
        Add ROI selection and per-ROI trend plots.
    cmap : str
    xdata, ydata : array-like, optional
        Physical coordinates for the image's columns/rows respectively
        (length must match the image's x/y size). Assumed monotonic.
    norm : None, str, or matplotlib.colors.Normalize
        See above. ``None`` (default) is plain linear scaling.
    norm_kwargs : dict, optional
        Extra keyword arguments for the chosen ``norm`` (e.g.
        ``{"vcenter": 0.0}``, ``{"gamma": 0.5}``, ``{"linthresh": 0.1}``).
    autoscale : {"minmax", "percentile", None}
        Per-frame color-range strategy; see above.
    autoscale_percentile : (float, float)
        Percentile bounds used when ``autoscale="percentile"``.
    figname : str
        Base name for the created matplotlib figures.
    detached : bool
        If ``True``, show this widget in a JupyterLab Sidecar panel instead
        of inline. Re-running with the same ``figname`` replaces that
        widget's existing sidecar panel rather than opening another one.
        Requires the ``sidecar`` package.
    title : str, optional
        Sidecar panel title. Defaults to ``array.name`` if set, else
        ``figname``.
    """

    def __init__(
        self,
        array,
        preview_frames=50,
        refine_delay=0.5,
        rois=False,
        cmap="viridis",
        xdata=None,
        ydata=None,
        norm=None,
        norm_kwargs=None,
        autoscale="minmax",
        autoscale_percentile=(1, 99),
        figname="StackViewer",
        detached=False,
        title=None,
    ):
        super().__init__()
        self.figname = figname
        self.step_mode = hasattr(array, "scan") and len(array.scan) > 1
        if self.step_mode:
            self._scan = array.scan
            self._par_steps = self._scan.par_steps
            self.n = len(self._scan)
        else:
            if isinstance(array, da.Array):
                self._data3d = array
            elif hasattr(array, "scan"):
                self._data3d = array.data
            else:
                self._data3d = np.asarray(array)
            if self._data3d.ndim != 3:
                raise ValueError(
                    f"Expected a 3-D image stack (n_frames, ny, nx); got shape {self._data3d.shape}"
                )
            self.n = self._data3d.shape[0]

        self.preview_frames = preview_frames
        self.refine_delay = refine_delay
        self._rois_requested = rois
        self.roi_panel = None
        self._image_cache = {}
        self._roi_means = {}
        self._draw_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._timer = None

        self._cmap = cmap
        self._norm_kind = norm
        self._norm_kwargs = norm_kwargs
        self.autoscale = autoscale
        self.autoscale_percentile = autoscale_percentile
        self._xdata = None if xdata is None else np.asarray(xdata)
        self._ydata = None if ydata is None else np.asarray(ydata)
        self._use_mesh = self._xdata is not None or self._ydata is not None

        ny, nx = self._peek_shape()
        if self._xdata is not None and len(self._xdata) != nx:
            raise ValueError(
                f"xdata has length {len(self._xdata)}, expected {nx} (image width)"
            )
        if self._ydata is not None and len(self._ydata) != ny:
            raise ValueError(
                f"ydata has length {len(self._ydata)}, expected {ny} (image height)"
            )

        self.output = widgets.Output()
        with self.output:
            plt.close(figname)
            self.fig = plt.figure(figname, constrained_layout=True)
            self.ax = self.fig.add_subplot()
            placeholder = np.zeros((ny, nx))
            built_norm = _build_norm(norm, norm_kwargs, None, None)
            im_kwargs = dict(cmap=cmap)
            if built_norm is not None:
                im_kwargs["norm"] = built_norm
            if self._use_mesh:
                from .utilities import (
                    plot2D,
                )  # lazy: utilities imports this module itself

                x = self._xdata if self._xdata is not None else "auto"
                y = self._ydata if self._ydata is not None else "auto"
                self.im = plot2D(x, y, placeholder, axis=self.ax, **im_kwargs)
            else:
                self.im = self.ax.imshow(placeholder, **im_kwargs)
            self.cbar = self.fig.colorbar(self.im, ax=self.ax)
            plt.show(self.fig)

        self.slider = widgets.IntSlider(
            min=0,
            max=self.n - 1,
            value=0,
            step=1,
            description="index",
            continuous_update=True,
        )
        self.status = widgets.Label(value="")
        self.slider.observe(lambda change: self._load(change["new"]), names="value")

        # Image + its slider/status form one unit ("card"); once an ROI panel is
        # added (see _ensure_roi_panel) it becomes a sibling of this card in a
        # flex-wrapping layout, so the two flow side by side or stack depending on
        # available width, instead of always being forced into one rigid column.
        self._image_box = widgets.VBox(
            [self.output, widgets.HBox([self.slider, self.status])]
        )
        self.layout = widgets.Layout(
            display="flex", flex_flow="row wrap", align_items="flex-start"
        )
        self.children = [self._image_box]

        # Colorbar drag/pan/zoom (which matplotlib supports out of the box) does
        # NOT go through self.im.set_clim() -- it mutates self.im.norm.vmin/vmax
        # directly (see Colorbar.drag_pan in matplotlib's colorbar.py), which only
        # fires self.im's own 'changed' callback on matplotlib versions new enough
        # to cascade norm -> mappable callbacks (the Colorizer refactor). Hooking
        # the norm's *own* callback registry directly is what actually catches a
        # colorbar drag on any matplotlib version; the mappable-level hook is kept
        # too so set_clim()-driven changes (e.g. from the ROI slider) are covered
        # even where that cascade is absent.
        self._syncing_clim = False
        self.im.callbacks.connect("changed", self._on_main_clim_changed)
        self.im.norm.callbacks.connect("changed", self._on_main_clim_changed)

        array_name = getattr(array, "name", None)
        resolved_title = title or array_name or figname
        _close_sidecar(figname)
        if detached:
            _open_sidecar(figname, resolved_title, lambda: display(self))
            _suppress_inline_redisplay(self)

        self._load(0)

    def _peek_shape(self):
        return self._block(0).shape[-2:]

    def _block(self, idx):
        if self.step_mode:
            return self._scan[idx].data
        return self._data3d[idx]

    def _label(self, idx):
        if self.step_mode:
            return str(dict(self._par_steps.iloc[idx]))
        return f"frame {idx}"

    def _load(self, idx):
        cancel_event = threading.Event()
        self._cancel_event.set()
        self._cancel_event = cancel_event
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if idx in self._image_cache:
            self._show(idx, self._image_cache[idx], final=True)
            return

        block = self._block(idx)
        is_dask = isinstance(block, da.Array)

        if self.step_mode:
            n_frames = block.shape[0]
            preview_n = min(self.preview_frames, n_frames)
            if is_dask:
                threading.Thread(
                    target=self._load_step_bg,
                    args=(idx, block, preview_n, n_frames, cancel_event),
                ).start()
            else:
                data = np.mean(block, axis=0) if n_frames > 1 else np.asarray(block[0])
                self._show(idx, data, final=True)
        else:
            if is_dask:
                threading.Thread(
                    target=self._load_frame_bg, args=(idx, block, cancel_event)
                ).start()
            else:
                self._show(idx, np.asarray(block), final=True)

    def _load_step_bg(self, idx, block, preview_n, n_frames, cancel_event):
        preview = np.asarray(block[:preview_n].mean(axis=0).compute())
        if cancel_event.is_set():
            return
        self._show(idx, preview, final=(preview_n == n_frames))
        if preview_n < n_frames:
            self._timer = threading.Timer(
                self.refine_delay, self._refine_step, args=(idx, block, cancel_event)
            )
            self._timer.start()

    def _refine_step(self, idx, block, cancel_event):
        if cancel_event.is_set():
            return
        full = block.mean(axis=0).persist()
        result = full.compute() if hasattr(full, "compute") else np.asarray(full)
        if cancel_event.is_set():
            return
        self._show(idx, np.asarray(result), final=True)

    def _load_frame_bg(self, idx, block, cancel_event):
        result = np.asarray(block.compute())
        if cancel_event.is_set():
            return
        self._show(idx, result, final=True)

    def _update_image_array(self, data):
        if self._use_mesh:
            self.im.set_array(np.asarray(data))
        else:
            self.im.set_data(data)

    def _autoscale_norm(self, data):
        """Apply this frame's color range per ``self.autoscale``. A no-op if
        autoscale is disabled, or the norm is a user-supplied instance
        (respected exactly, never auto-rescaled)."""
        if not self.autoscale or isinstance(self._norm_kind, mcolors.Normalize):
            return
        if self.autoscale == "percentile":
            vmin, vmax = np.nanpercentile(data, self.autoscale_percentile)
        else:
            vmin, vmax = np.nanmin(data), np.nanmax(data)
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
            return
        norm = self.im.norm
        if isinstance(norm, mcolors.CenteredNorm):
            # vmin/vmax aren't independently settable in a way that keeps
            # CenteredNorm's symmetry-around-vcenter guarantee -- halfrange is.
            norm.halfrange = max(abs(vmin - norm.vcenter), abs(vmax - norm.vcenter))
        else:
            self.im.set_clim(float(vmin), float(vmax))

    def _show(self, idx, data, final):
        self._ensure_roi_panel()
        with self._draw_lock:
            if idx != self.slider.value:
                return
            self._update_image_array(data)
            self._autoscale_norm(data)
            self.fig.canvas.draw_idle()
            self.status.value = self._label(idx) + ("" if final else " (preview…)")
            if self.roi_panel is not None:
                self.roi_panel.refresh_all(data)
        if final:
            self._image_cache[idx] = data
            if self.roi_panel is not None:
                self._record_roi_means(idx, data)

    def _ensure_roi_panel(self):
        if not self._rois_requested or self.roi_panel is not None:
            return
        make_norm = (
            (
                lambda vmin, vmax: _build_norm(
                    self._norm_kind, self._norm_kwargs, vmin, vmax
                )
            )
            if self._norm_kind is not None
            else None
        )
        self.roi_panel = RoiPanel(
            self.ax,
            self.im,
            cmap=self._cmap,
            make_norm=make_norm,
            xdata=self._xdata,
            ydata=self._ydata,
            on_regions_changed=self._on_roi_change,
            on_clim_changed=self._on_roi_panel_clim_changed,
        )
        self.trend_output = widgets.Output()
        with self.trend_output:
            plt.close(f"{self.figname}_trend")
            self.trend_fig, self.trend_ax = plt.subplots(
                num=f"{self.figname}_trend", constrained_layout=True, figsize=(5, 3)
            )
            plt.show(self.trend_fig)

        # Trend plot joins the ROI panel's own (also flex-wrapping) box, so it
        # flows next to/under the ROI tabs rather than always sitting in its own
        # full-width row below everything.
        self.roi_panel.children = list(self.roi_panel.children) + [self.trend_output]
        self.children = [self._image_box, self.roi_panel]

        # Seed the panel's slider/previews with the main image's current clim.
        self._on_main_clim_changed()

    def _on_main_clim_changed(self, mappable=None):
        """Propagate the main image's clim to the ROI panel's slider/previews.

        Connected to both self.im's and self.im.norm's own 'changed' callback
        registries (see __init__), whose signal handlers are invoked with
        different arguments (the mappable, the norm, or nothing at all,
        depending on matplotlib version) -- so the argument is ignored and
        self.im.get_clim() is re-read fresh instead.
        """
        if self.roi_panel is None or self._syncing_clim:
            return
        self._syncing_clim = True
        try:
            self.roi_panel.sync_clim(*self.im.get_clim())
        finally:
            self._syncing_clim = False

    def _on_roi_panel_clim_changed(self, vmin, vmax):
        """Propagate the ROI panel's clim slider back to the main image/colorbar."""
        if self._syncing_clim:
            return
        self._syncing_clim = True
        try:
            self.im.set_clim(vmin, vmax)
            self.fig.canvas.draw_idle()
        finally:
            self._syncing_clim = False

    def _record_roi_means(self, idx, data):
        for region in self.roi_panel.regions:
            self._roi_means.setdefault(region.name, {})[idx] = region.mean(data)
        self._redraw_trend()

    def _on_roi_change(self):
        self._roi_means = {}
        for idx, data in sorted(self._image_cache.items()):
            for region in self.roi_panel.regions:
                self._roi_means.setdefault(region.name, {})[idx] = region.mean(data)
        self._redraw_trend()

    def _redraw_trend(self):
        self.trend_ax.cla()
        regions_by_name = {r.name: r for r in self.roi_panel.regions}
        for name, d in self._roi_means.items():
            xs = sorted(d)
            color = regions_by_name[name].color if name in regions_by_name else None
            self.trend_ax.plot(
                xs, [d[x] for x in xs], marker="o", label=name, color=color
            )
        if self._roi_means:
            self.trend_ax.legend()
        self.trend_ax.set_xlabel("step index" if self.step_mode else "frame index")
        self.trend_ax.set_ylabel("ROI mean")
        self.trend_fig.canvas.draw_idle()

    def close_viewer(self):
        """Cancel pending background work and close the created figures."""
        self._cancel_event.set()
        if self._timer is not None:
            self._timer.cancel()
        plt.close(self.fig)
        if self.roi_panel is not None:
            self.roi_panel.close()
            plt.close(self.trend_fig)


def errortube(x, y, yerr=None, xerr=None, fmt=None, axis=None, falpha=0.3, **kwargs):
    """Plot a line with a shaded error band (a lighter-weight
    ``fill_between``-style alternative to matplotlib's ``errorbar``).

    Parameters
    ----------
    x, y : array-like
        Line coordinates.
    yerr : array-like, optional
        Either a 1-D array of symmetric errors, or a 2-D ``(2, N)`` array of
        ``(lower, upper)`` errors for an asymmetric band. Omit for a plain
        line with no shaded band.
    xerr : array-like, optional
        Accepted for interface symmetry with ``errorbar`` but not currently
        used -- no horizontal error shading is drawn.
    fmt : str, optional
        Matplotlib format string for the line (e.g. ``"o-"``).
    axis : matplotlib.axes.Axes, optional
        Axes to plot into. Defaults to the current axes.
    falpha : float
        Alpha (transparency) of the shaded error band.
    **kwargs
        Forwarded to ``axis.plot`` for the line itself.

    Returns
    -------
    lh : matplotlib.lines.Line2D
        The plotted line.
    fh : matplotlib.patches.Polygon or None
        The shaded error band, or ``None`` if ``yerr`` wasn't given.
    """
    if not axis:
        axis = plt.gca()

    args = [x, y]
    if fmt is not None:
        args.append(fmt)
    lh = axis.plot(*args, **kwargs)[0]

    if yerr is not None:
        yerr = np.atleast_1d(yerr)
        if yerr.ndim == 1:
            fh = axis.fill(
                np.hstack([np.asarray(x), np.asarray(x)[::-1]]),
                np.hstack([np.asarray(y) - yerr, np.asarray(y)[::-1] + yerr[::-1]]),
                alpha=falpha,
                color=lh.get_color(),
                zorder=lh.get_zorder() - 0.1,
            )
        if yerr.ndim == 2:
            fh = axis.fill(
                np.hstack([np.asarray(x), np.asarray(x)[::-1]]),
                np.hstack(
                    [np.asarray(y) - yerr[0, :], np.asarray(y)[::-1] + yerr[1, ::-1]]
                ),
                alpha=falpha,
                color=lh.get_color(),
                zorder=lh.get_zorder() - 0.1,
            )
    else:
        fh = None

    return lh, fh


# ----------------------------------------------------------------------------
# Dual x-axis plotting for paired (possibly non-monotonic) x arrays
# ----------------------------------------------------------------------------
# You have three same-length arrays: x1, x2, y. Both x1[i] and x2[i] describe
# the same point y[i] (e.g. depth and age of a sediment sample, or index and
# wall-clock time of a sensor reading). You want one plot of y, with x1
# driving the main axis normally, and a second axis on top that shows where
# each *x2* value falls -- even though x2 is not a monotonic function of x1
# (it can wiggle, reverse, repeat).
#
# Matplotlib's built-in tools for a second x-axis (``Axes.twiny`` combined
# with ``Axes.secondary_xaxis(functions=(fwd, inv))``) both assume the
# relationship between the two x-scales is an invertible (monotonic)
# function: they place "nice" tick values by transforming *numbers*, not by
# looking up paired samples. That assumption fails for genuinely
# non-monotonic pairings, so this takes a different, index-based approach:
#
# 1. Pick a handful of "nice" target values within the range of x2.
# 2. Walk the (x1, x2, y) sample path index-by-index and find every place the
#    piecewise-linear path crosses each target x2 value (there can be zero,
#    one, or several crossings per value, since x2 need not be monotonic).
#    Each crossing gives an exact x1-position via linear interpolation
#    between the two neighbouring samples -- this needs no global
#    monotonicity, only that consecutive samples are locally
#    linear-interpolable.
# 3. Place secondary-axis ticks at those x1-positions, labelled with the
#    target x2 values, with optional markers on the primary curve and drop
#    lines tying each tick to its point, and staggered label heights when
#    neighbouring ticks are too close to read.
#
# The "index a paired non-monotonic array and interpolate crossings" trick
# mirrors what geoscientists do by hand for age-depth (or depth-time) plots
# in stratigraphy/paleoclimate figures -- ties, reversals and all.


@dataclass
class SecondaryCrossing:
    """One resolved secondary-axis tick: where the (x1, x2, y) path crosses
    a target x2 ``value``, at primary-axis position ``x1_pos`` (with the
    corresponding curve height ``y_pos``, used for markers/drop-lines)."""

    x1_pos: float
    value: float
    y_pos: float


def _find_crossings(x1, x2, y, targets, tol_frac=1e-9):
    """Find every place the piecewise-linear (x1, x2, y) path crosses each
    target value of x2. Works regardless of whether x2 is monotonic.

    Returns a list of SecondaryCrossing, one per crossing found (a single
    target value may produce zero, one, or several crossings).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x2)
    span = np.ptp(x1) or 1.0
    tol = tol_frac * span

    crossings: list[SecondaryCrossing] = []

    for v in targets:
        # seen_x1 is scoped per target value: it only suppresses redundant
        # re-detection of the *same* crossing (e.g. a flat segment sitting
        # exactly on the target), not distinct targets that legitimately
        # land at the same x1 position (e.g. a repeated/sawtooth scan,
        # where several different step numbers share one x1 value).
        seen_x1: list[float] = []

        def _record(x1_pos, value, y_pos):
            for prev in seen_x1:
                if abs(prev - x1_pos) <= tol:
                    return
            seen_x1.append(x1_pos)
            crossings.append(SecondaryCrossing(x1_pos, value, y_pos))

        for i in range(n - 1):
            a, b = x2[i] - v, x2[i + 1] - v
            if a == 0:
                _record(x1[i], v, y[i])
            if a == 0 and b == 0:
                continue  # flat segment exactly on target, avoid double count
            if (a < 0 < b) or (a > 0 > b):
                frac = a / (a - b)
                x1_pos = x1[i] + frac * (x1[i + 1] - x1[i])
                y_pos = y[i] + frac * (y[i + 1] - y[i])
                _record(x1_pos, v, y_pos)
        # catch an exact hit on the final sample
        if x2[-1] == v:
            _record(x1[-1], v, y[-1])

    crossings.sort(key=lambda c: c.x1_pos)
    return crossings


def _nudge_duplicate_positions(crossings, xlim):
    """Nudge crossings sharing an (near-)identical x1 position apart.

    matplotlib silently coalesces ticks placed at identical x-positions --
    which happens whenever a repeated/sawtooth scan revisits the same x1
    value at several different x2 targets (e.g. step numbers) -- collapsing
    every label in the cluster onto the last one's text. The nudge is far
    below plotting resolution, so markers/drop-lines stay visually unchanged.
    """
    span = xlim[1] - xlim[0]
    eps = 1e-6 * (abs(span) or 1.0)
    for i in range(1, len(crossings)):
        if crossings[i].x1_pos - crossings[i - 1].x1_pos < eps:
            crossings[i].x1_pos = crossings[i - 1].x1_pos + eps


def _stagger_tick_labels(ax, ax2, crossings, gap_frac, secondary_side, n_levels=3):
    """Vertically stagger secondary tick labels that fall closer together
    than ``gap_frac`` of the axis width, so dense/colliding clusters stay
    legible instead of overprinting each other."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    gap = gap_frac * abs(span)
    positions = [c.x1_pos for c in crossings]
    labels = ax2.get_xticklabels()
    level = 0
    # Stack outward, away from the plot (up for a top axis, down for a bottom
    # one) -- not inward, which would push stacked labels down into the data.
    direction = 1 if secondary_side == "top" else -1
    for i in range(1, len(positions)):
        level = (
            (level + 1) % n_levels if abs(positions[i] - positions[i - 1]) < gap else 0
        )
        if level:
            offset = mtransforms.offset_copy(
                labels[i].get_transform(),
                fig=ax2.figure,
                y=direction * 15 * level,
                units="points",
            )
            labels[i].set_transform(offset)


def add_crossing_secondary_axis(
    ax,
    x1,
    x2,
    y=None,
    primary_side="bottom",
    n_ticks=6,
    tick_values=None,
    x2_label=None,
    fmt="{:.3g}",
    show_markers=True,
    show_drop_lines=True,
    stagger_labels=True,
    stagger_gap_frac=0.06,
    accent_color="#B0413E",
):
    """Add a non-monotonic-aware secondary x-axis (twiny) to an existing plot.

    Unlike ``dual_x_axis_plot``, this does not draw the primary curve -- it
    assumes ``ax`` already shows y plotted against x1, and only adds the
    secondary axis on top (or bottom) of it, with ticks placed wherever the
    paired (x1, x2) samples cross a handful of "nice" x2 values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that already plots y against x1.
    x1, x2 : array-like, same length
        Paired samples: x1 is the coordinate system already used by ``ax``;
        x2 is the secondary array to label (may be non-monotonic).
    y : array-like, optional
        Same length as x1/x2. Only needed for ``show_markers`` /
        ``show_drop_lines`` (used to place them on the curve); if omitted,
        those are disabled automatically.
    primary_side : {"bottom", "top"}
        Which side ``ax``'s primary axis is already on; the secondary axis
        is placed on the opposite side.
    n_ticks : int
        Approximate number of "nice" secondary tick values to aim for.
    tick_values : array-like, optional
        Explicit x2 values to place ticks at, bypassing the automatic
        "nice value" search.
    x2_label : str, optional
        Axis label for the secondary axis.
    fmt : str
        Format string used for secondary tick labels, applied as
        ``fmt.format(value)``.
    show_markers, show_drop_lines, stagger_labels, stagger_gap_frac, accent_color
        See ``dual_x_axis_plot``.

    Returns
    -------
    ax2 : the secondary Axes (twiny, x2 ticks)
    crossings : list[SecondaryCrossing]
        The resolved (x1_pos, value, y_pos) triples used for the secondary
        ticks, in case you want to annotate or reuse them.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if y is None:
        y = np.zeros_like(x1)
        show_markers = False
        show_drop_lines = False
    else:
        y = np.asarray(y, dtype=float)
    if not (len(x1) == len(x2) == len(y)):
        raise ValueError("x1, x2, y must have the same length")

    secondary_side = "top" if primary_side == "bottom" else "bottom"

    if tick_values is None:
        locator = mticker.MaxNLocator(nbins=n_ticks)
        candidates = locator.tick_values(np.nanmin(x2), np.nanmax(x2))
        lo, hi = np.nanmin(x2), np.nanmax(x2)
        tick_values = [v for v in candidates if lo <= v <= hi]
    crossings = _find_crossings(x1, x2, y, tick_values)
    _nudge_duplicate_positions(crossings, ax.get_xlim())

    # --- secondary (twin) axis, forced onto the same x1 coordinate range ---
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    if secondary_side == "bottom":
        ax2.xaxis.tick_bottom()
        ax2.xaxis.set_label_position("bottom")
        ax2.spines["bottom"].set_position(("outward", 40))
        ax2.spines["bottom"].set_visible(True)

    ax2.set_xticks([c.x1_pos for c in crossings])
    ax2.set_xticklabels([fmt.format(c.value) for c in crossings])
    ax2.tick_params(axis="x", colors=accent_color)
    ax2.spines[secondary_side].set_color(accent_color)
    if x2_label:
        ax2.set_xlabel(x2_label, color=accent_color)

    if show_markers:
        ax.plot(
            [c.x1_pos for c in crossings],
            [c.y_pos for c in crossings],
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=accent_color,
            markeredgewidth=1.3,
            linestyle="none",
            zorder=5,
        )

    if show_drop_lines:
        ylim = ax.get_ylim()
        target_y = ylim[1] if secondary_side == "top" else ylim[0]
        for c in crossings:
            ax.plot(
                [c.x1_pos, c.x1_pos],
                [c.y_pos, target_y],
                linestyle=":",
                linewidth=0.9,
                color=accent_color,
                alpha=0.55,
                zorder=1,
            )
        ax.set_ylim(ylim)  # drop lines must not rescale the axis

    if stagger_labels and len(crossings) > 1:
        _stagger_tick_labels(ax, ax2, crossings, stagger_gap_frac, secondary_side)

    return ax2, crossings


def dual_x_axis_plot(
    x1,
    x2,
    y,
    ax=None,
    primary_side="bottom",
    n_ticks=6,
    tick_values=None,
    x1_label=None,
    x2_label=None,
    fmt="{:.3g}",
    show_markers=True,
    show_drop_lines=True,
    stagger_labels=True,
    stagger_gap_frac=0.06,
    accent_color="#B0413E",
    plot_kwargs=None,
):
    """Plot y against x1 with a second, non-monotonic-aware x-axis for x2.

    Parameters
    ----------
    x1, x2, y : array-like, same length
        Paired samples: point i is (x1[i], x2[i], y[i]). x1 is the primary,
        "trusted" x-array (ideally monotonic; it drives the main axis and
        gets ordinary automatic ticks). x2 is the secondary x-array and may
        be non-monotonic, noisy, or reversing -- its ticks are derived by
        interpolated crossing-detection instead of a functional transform.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. A new figure/axes is created if omitted.
    primary_side : {"bottom", "top"}
        Which side carries the primary (x1) axis; the secondary (x2) axis
        is placed on the opposite side.
    n_ticks : int
        Approximate number of "nice" secondary tick values to aim for
        (actual count of ticks drawn can differ, since a value may cross
        zero, one, or several times).
    tick_values : array-like, optional
        Explicit x2 values to place ticks at, bypassing the automatic
        "nice value" search.
    x1_label, x2_label : str, optional
        Axis labels for the primary and secondary axes.
    fmt : str
        Format string used for secondary tick labels, applied as
        ``fmt.format(value)``.
    show_markers : bool
        Draw a small marker on the curve at each point a secondary tick
        corresponds to (helps disambiguate repeated/non-monotonic values).
    show_drop_lines : bool
        Draw a thin dotted line from each secondary tick down/up to its
        marker, tying the (possibly unevenly spaced) tick to its data point.
    stagger_labels : bool
        Alternate the vertical offset of secondary tick labels that fall
        closer together than ``stagger_gap_frac`` of the axis width, to
        keep dense/irregular tick clusters legible.
    stagger_gap_frac : float
        Minimum spacing (as a fraction of the x1 axis width) below which
        neighbouring secondary labels are staggered.
    accent_color : str
        Color used for secondary ticks, markers and drop lines, to visually
        group them as "belonging" to the secondary axis.
    plot_kwargs : dict, optional
        Extra keyword arguments passed to the primary ``ax.plot`` call.

    Returns
    -------
    ax : the primary Axes (x1, y)
    ax2 : the secondary Axes (twiny, x2 ticks)
    crossings : list[SecondaryCrossing]
        The resolved (x1_pos, value, y_pos) triples used for the secondary
        ticks, in case you want to annotate or reuse them.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y = np.asarray(y, dtype=float)
    if not (len(x1) == len(x2) == len(y)):
        raise ValueError("x1, x2, y must have the same length")

    if ax is None:
        _, ax = plt.subplots()

    plot_kwargs = dict(plot_kwargs or {})
    plot_kwargs.setdefault("color", "0.15")
    plot_kwargs.setdefault("linewidth", 1.5)
    ax.plot(x1, y, **plot_kwargs)

    if primary_side == "top":
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    if x1_label:
        ax.set_xlabel(x1_label)

    ax2, crossings = add_crossing_secondary_axis(
        ax,
        x1,
        x2,
        y,
        primary_side=primary_side,
        n_ticks=n_ticks,
        tick_values=tick_values,
        x2_label=x2_label,
        fmt=fmt,
        show_markers=show_markers,
        show_drop_lines=show_drop_lines,
        stagger_labels=stagger_labels,
        stagger_gap_frac=stagger_gap_frac,
        accent_color=accent_color,
    )
    return ax, ax2, crossings


# ----------------------------------------------------------------------------
# Step-index secondary axis: monotonic-segment aware
# ----------------------------------------------------------------------------
# ``add_crossing_secondary_axis`` is fully general (x2 can be anything), which
# means it always falls back to the "search for crossings of nice round
# numbers" approach. But the common case in this codebase is much more
# specific: x1 is a scan parameter, x2 is simply the step index (0..n-1),
# which is *always* monotonic by construction. All the interesting structure
# lives in x1:
#
# * If x1 is monotonic over the whole scan (one pass, e.g. a plain sorted
#   scan), the (x1 -> step) relationship is invertible, so we can hand it
#   straight to matplotlib's own ``Axes.secondary_xaxis(functions=(fwd,
#   inv))`` -- giving a real, densely and "nicely" ticked axis that reflects
#   x1's true (possibly nonlinear, e.g. log-spaced) shape, rather than the
#   handful of manually-searched crossing points above.
# * If x1 repeats itself (a sawtooth -- deliberately re-scanning the same
#   range several times to average out drift), it is no longer globally
#   invertible. We instead split it into its maximal monotonic runs
#   ("segments"), get a locally-invertible nonlinear tick set from each
#   segment independently, merge them, and stack any that land on the same
#   x1 position (reusing the machinery above).
#
# Either way, the points where x1 changes direction ("turning points" --  the
# segment boundaries) are worth calling out explicitly: ``label_turning_points``
# annotates each one, directly on the curve, with its step index, decluttered
# with ``adjustText`` if it's installed.


def _monotonic_segments(x1):
    """Split x1 into maximal monotonic (non-decreasing or non-increasing) runs.

    Returns a list of ``(start, end)`` data-index pairs (both inclusive),
    covering the whole array, split at each local extremum ("turning point").
    A flat (zero-diff) step is treated as a continuation of whatever
    direction preceded it, so plateaus don't create spurious segments.
    """
    x1 = np.asarray(x1, dtype=float)
    n = len(x1)
    if n < 3:
        return [(0, n - 1)]

    d = np.diff(x1)
    sign = np.where(d > 0, 1, np.where(d < 0, -1, 0)).astype(int)
    last = 0
    for i in range(len(sign)):
        if sign[i] == 0:
            sign[i] = last
        else:
            last = sign[i]

    turning_points = [i for i in range(1, n - 1) if sign[i] != sign[i - 1]]
    boundaries = [0] + turning_points + [n - 1]
    return [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]


def _make_forward_inverse(x1_seg, step_seg):
    """Build (forward, inverse) callables between a monotonic x1 segment and
    its step indices, for use with ``Axes.secondary_xaxis``."""
    if x1_seg[0] <= x1_seg[-1]:
        x1_asc, step_asc = x1_seg, step_seg
    else:
        x1_asc, step_asc = x1_seg[::-1], step_seg[::-1]

    def forward(x1_query):
        return np.interp(np.asarray(x1_query, dtype=float), x1_asc, step_asc)

    def inverse(step_query):
        return np.interp(np.asarray(step_query, dtype=float), step_seg, x1_seg)

    return forward, inverse


def _label_turning_points(ax, x1, y, turning_points, fmt, color):
    """Annotate each turning point directly on the curve with its step
    index, decluttered with ``adjustText`` if available.

    Returns the list of created Text objects (empty if there was nothing to
    label). Falls back to plain (possibly overlapping) labels with a
    warning if ``adjustText`` isn't installed.
    """
    if not turning_points:
        return []

    texts = [
        ax.annotate(
            fmt.format(i),
            (x1[i], y[i]),
            fontsize=8,
            fontweight="bold",
            color=color,
            zorder=6,
        )
        for i in turning_points
    ]

    try:
        from adjustText import adjust_text
    except ImportError:
        import warnings

        warnings.warn(
            "label_turning_points=True but the 'adjustText' package is not "
            "installed (pip install adjustText) -- turning-point labels are "
            "shown but may overlap.",
            stacklevel=2,
        )
        return texts

    adjust_text(
        texts,
        x=x1,
        y=y,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
    )
    return texts


def add_step_secondary_axis(
    ax,
    x1,
    y=None,
    x2_label="step",
    fmt="{:.0f}",
    n_ticks=6,
    show_markers=True,
    label_turning_points=True,
    stagger_gap_frac=0.06,
    accent_color="#B0413E",
):
    """Add a step-index secondary axis to a plot of y vs. scan parameter x1.

    Specialised sibling of :func:`add_crossing_secondary_axis` for the most
    common case in this codebase: the secondary quantity is simply the step
    index ``0..len(x1)-1``. Automatically detects whether x1 is a single
    monotonic pass or contains repeated ("sawtooth") cycles and adapts:

    * **Single pass** -- x1 is monotonic over the whole array. Uses a real
      ``Axes.secondary_xaxis`` with an interpolated forward/inverse mapping,
      giving a densely, "nicely" ticked axis that reflects x1's true
      (possibly nonlinear) shape.
    * **Repeated / sawtooth** -- x1 is no longer globally invertible, so this
      falls back to :func:`add_crossing_secondary_axis` (searching for where
      the path crosses a handful of "nice" step values -- which, since step
      is a plain index, works out to one exact crossing per target). Targets
      that land on the same x1 position because a repeat revisits it are
      stacked vertically rather than overprinting each other.

    In both cases, every point where x1 changes direction (a "turning
    point" -- e.g. where a sawtooth resets) is optionally annotated directly
    on the curve with its step index (see ``label_turning_points``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that already plots y against x1.
    x1 : array-like
        The scan parameter already used as ``ax``'s x-coordinate.
    y : array-like, optional
        Same length as x1. Needed for ``show_markers`` and
        ``label_turning_points``; both are skipped automatically if omitted.
    x2_label : str
        Label for the secondary axis.
    fmt : str
        Format string for both secondary tick labels and turning-point
        labels, applied as ``fmt.format(step_index)``.
    n_ticks : int
        Approximate number of "nice" secondary tick values to aim for
        (repeated case only -- ignored for a single monotonic pass, where
        matplotlib's own dense functional axis is used instead).
    show_markers : bool
        Draw a marker on the curve at each secondary tick position (repeated
        case only -- the single-pass case has no extra points to mark).
    label_turning_points : bool
        Annotate each direction-reversal point with its step index, using
        ``adjustText`` (if installed) to avoid overlaps.
    stagger_gap_frac : float
        Minimum spacing (as a fraction of the x1 axis width) below which
        neighbouring secondary tick labels are vertically staggered.
    accent_color : str
        Color used for the secondary axis, its ticks, and turning-point
        labels.

    Returns
    -------
    ax2 : the secondary Axes
    crossings : list[SecondaryCrossing]
        The tick positions used (empty for the single-pass case, which uses
        a functional axis instead of explicit tick positions).
    turning_points : list[int]
        Data indices where x1 changes direction.
    """
    x1 = np.asarray(x1, dtype=float)
    n = len(x1)
    step = np.arange(n, dtype=float)
    y = None if y is None else np.asarray(y, dtype=float)

    segments = _monotonic_segments(x1)
    turning_points = [b for (a, b) in segments[:-1]]

    if len(segments) <= 1:
        forward, inverse = _make_forward_inverse(x1, step)
        ax2 = ax.secondary_xaxis("top", functions=(forward, inverse))
        ax2.set_xlabel(x2_label, color=accent_color)
        ax2.tick_params(axis="x", colors=accent_color)
        ax2.spines["top"].set_color(accent_color)
        ax2.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, pos: fmt.format(v))
        )
        crossings = []
    else:
        # Pick a tick *stride* from one representative cycle's length, rather
        # than searching for "nice" numbers over the full 0..n-1 range: two
        # independent nice-number searches over different absolute ranges
        # (e.g. one per repeat) generally do *not* land on the same phase, so
        # ticks from different repeats would end up close but not stacked.
        # A single global stride derived from the cycle length repeats at the
        # same phase every cycle by construction, so repeats reliably land on
        # (and get stacked onto) the same x1 positions.
        real_segments = [(a, b) for a, b in segments if b > a]
        seg_len = (
            (real_segments[0][1] - real_segments[0][0] + 1) if real_segments else n
        )
        candidates = mticker.MaxNLocator(nbins=n_ticks).tick_values(0, seg_len - 1)
        raw_stride = (candidates[1] - candidates[0]) if len(candidates) > 1 else seg_len
        # Snap to a divisor of the cycle length (nearest, ties favouring the
        # sparser option) so the stride tiles exactly and every repeat lands
        # on the same phase -- an arbitrary "nice" float stride generally
        # wouldn't divide the cycle evenly and repeats would drift out of
        # alignment instead of stacking.
        divisors = [d for d in range(1, seg_len + 1) if seg_len % d == 0]
        stride = min(divisors, key=lambda d: (abs(d - raw_stride), -d))
        tick_values = np.arange(0, n, stride)

        ax2, crossings = add_crossing_secondary_axis(
            ax,
            x1,
            step,
            y,
            tick_values=tick_values,
            x2_label=x2_label,
            fmt=fmt,
            show_markers=show_markers,
            show_drop_lines=False,
            stagger_gap_frac=stagger_gap_frac,
            accent_color=accent_color,
        )

    if label_turning_points and y is not None:
        _label_turning_points(ax, x1, y, turning_points, fmt, accent_color)

    return ax2, crossings, turning_points
