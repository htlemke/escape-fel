import matplotlib.pyplot as plt
from time import sleep
import ipywidgets as widgets
import numpy as np
from threading import Thread


class GinputNB:
    def __init__(self, fig=None):
        self.x = []
        self.y = []
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0

    def get(self, Npts=-1.0, plotspec="rd-"):
        self.mcid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.kcid = self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        if plotspec:
            self.line = plt.plot(self.x, self.y, plotspec)[0]
        self.collecting = Npts
        return self

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
    return widgets.Layout(
        border="solid 1px black",
        margin="0px 10px 10px 0px",
        padding="5px 5px 5px 5px",
        flex_flow="row wrap",
    )


class SpanSelectNB:
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
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(self.get_image_roi_data())
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
                # self.ax_roi.set_aspect(abs(shape[0] / shape[1]))
            else:
                self.ax_roi.imshow(self.get_image_roi_data())
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
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(self.get_image_roi_data())
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
                # self.ax_roi.set_aspect(abs(shape[0] / shape[1]))
            else:
                self.ax_roi.imshow(self.get_image_roi_data())
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
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(self.get_image_roi_data())
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
                # self.ax_roi.set_aspect(abs(shape[0] / shape[1]))
            else:
                self.ax_roi.imshow(self.get_image_roi_data())
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
        #         if self.selector.active_handle:
        #             self.selector.set_active(True)
        #         else:
        #             self.selector.set_active(False)
        #         'eclick and erelease are the press and release events'
        #         x1, y1 = eclick.xdata, eclick.ydata
        #         x2, y2 = erelease.xdata, erelease.ydata
        #         print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        #         print(" The button you used were: %s %s" % (eclick.button, erelease.button))
        if self.ax_roi:
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(self.get_image_roi_data())
                shape = i.get_array().shape
                i.set_extent((-0.5, shape[1] + 0.5, shape[0] + 0.5, -0.5))
                # self.ax_roi.set_aspect(abs(shape[0] / shape[1]))
            else:
                self.ax_roi.imshow(self.get_image_roi_data())
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
        return self.selector.extents[0]

    @property
    def xmax(self):
        return self.selector.extents[1]

    @property
    def ymin(self):
        return self.selector.extents[2]

    @property
    def ymax(self):
        return self.selector.extents[3]


class MultipleRoiSelector(widgets.HBox):
    def __init__(self, data, rois={}, callbacks_changeanyroi=[], name="RoiSelector"):
        # super().__init__(layout=widgets.Layout(flex_flow="row wrap"))
        super().__init__()
        self.data = data
        self.name = name
        self.roi_selectors = []
        self._tabs_rois = widgets.Tab()

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
            step=self.data.ptp() / 500,
            continuous_update=False,
            disabled=False,
        )

        self._add_roi_button.on_click(self.add_roi)
        self._roi_titles = []

        self.axs_rois = []
        self.figs_rois = []
        self.create_data_plot()

        self.children = [
            widgets.VBox(
                [
                    widgets.HBox(
                        [self._clim_slider, self._add_roi_button],
                    ),
                    self._output_data,
                ],
                # layout=widgets.Layout(min_width="400px"),
            ),
            self._tabs_rois,
        ]
        self.layout = make_box_layout()
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

    @property
    def rois(self):
        o = {}
        for tn, ts in zip(self._roi_titles, self.roi_selectors):
            o[tn.value] = (ts.xmin, ts.xmax, ts.ymin, ts.ymax)
        return o

    def create_data_plot(self):
        self._output_data = widgets.Output()
        with self._output_data:
            plt.close(self.name)
            fig, ax = plt.subplots(
                num=self.name,
                # constrained_layout=True
                figsize=[5, 5],
            )
            self.fig_data = fig
            self.ax_data = ax
            ih = self.ax_data.imshow(self.data)
            plt.colorbar(mappable=ih)
            plt.tight_layout()
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

    def set_clim(self, *args, **kwargs):

        with self.debug:
            print(*args, **kwargs)
        i = self.ax_data.get_images()[0]
        i.set_clim(*args, **kwargs)
        plt.draw()
        for n, ax in enumerate(self.axs_rois):
            print(n)
            if ax.get_images():
                i = ax.get_images()[0]
                i.set_clim(*args, **kwargs)
        plt.draw()

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
    def __init__(
        self,
        array,
        data_selection=slice(None, 100),
        update_rate=1,
        figname="StepViewer",
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


def nfigure(num="no name", **kwargs):
    if num in plt.get_figlabels():
        Warning('Figure of name "{num}" exists and is closed.')
    plt.close(num)
    return plt.figure(num=num, **kwargs)


def nsubplots(nrows=1, ncols=1, *, num="no name", **kwargs):
    if num in plt.get_figlabels():
        Warning('Figure of name "{num}" exists and is closed.')
    plt.close(num)
    return plt.subplots(nrows=nrows, ncols=ncols, num=num, **kwargs)


def nsubplot_mosaic(*args, num="no name", **kwargs):
    if num in plt.get_figlabels():
        Warning('Figure of name "{num}" exists and is closed.')
    plt.close(num)
    return plt.subplot_mosaic(*args, num=num, **kwargs)


class StepViewerP(widgets.VBox):
    def __init__(
        self,
        array,
        wid,
        data_selection=slice(None, 100),
        update_rate=1,
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


def errortube(x, y, yerr=None, xerr=None, fmt=None, axis=None, falpha=.3, **kwargs):
    if not axis:
        axis=plt.gca()

    args = [x,y]
    if fmt is not None:
        args.append(fmt)
    lh = axis.plot(*args,**kwargs)[0]
    
    if yerr is not None:
        yerr = np.atleast_1d(yerr)
        if yerr.ndim ==     1:
            fh = axis.fill(np.hstack([np.asarray(x),np.asarray(x)[::-1]]),
                    np.hstack([np.asarray(y)-yerr,np.asarray(y)[::-1]+yerr[::-1]]),
                    alpha=falpha,
                    color = lh.get_color(),
                    zorder=lh.get_zorder()-0.1
            )
        if yerr.ndim == 2:
            fh = axis.fill(np.hstack([np.asarray(x),np.asarray(x)[::-1]]),
                    np.hstack([np.asarray(y)-yerr[0,:],np.asarray(y)[::-1]+yerr[1,::-1]]),
                    alpha=falpha,
                    color = lh.get_color(),
                    zorder=lh.get_zorder()-0.1
            )
    else:
        fh = None
    
    return lh,fh



    
    


    