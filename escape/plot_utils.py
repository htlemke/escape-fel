from matplotlib import pyplot as plt


def getSpanCoordinates(direction="horizontal", axh=None, fig=None, data=None):
    """Tool for selecting a span, functionality similar to ginput. Finish with right mouse button."""
    if not axh:
        axh = plt.gca()
    if not fig:
        fig = plt.gcf()

    class ROI:
        def __init__(self, fig, axh, direction):
            self.fig = fig
            self.axh = axh
            self.lims = []
            self.boxh = []
            self.finished = False
            self.direction = direction

        def coo(self, tmin, tmax):
            self.lims = [tmin, tmax]
            if self.boxh:
                self.boxh.remove()
            if self.direction is "horizontal":
                self.boxh = self.axh.axvspan(tmin, tmax, facecolor="r", alpha=0.5)
                delta = tmax - tmin
                axh.set_xlim([tmin - 0.2 * delta, tmax + 0.2 * delta])
                if data is not None:
                    dat = data[1]

            if self.direction is "vertical":
                self.boxh = self.axh.axhspan(tmin, tmax, facecolor="r", alpha=0.5)
                delta = tmax - tmin
                axh.set_ylim([tmin - 0.2 * delta, tmax + 0.2 * delta])
            fig.canvas.draw()

        def button_press_callback(self, event):
            if event.inaxes:
                if event.button == 3:
                    self.finished = True

    roi = ROI(fig, axh, direction)
    selector = plt.matplotlib.widgets.SpanSelector(axh, roi.coo, direction)
    fig.canvas.mpl_connect("button_press_event", roi.button_press_callback)
    print("Select Span region of interest, finish with right click.")
    while not roi.finished:
        plt.waitforbuttonpress()
    print("Span %s selected." % (roi.lims))
    roi.boxh.remove()
    fig.canvas.draw()
    del selector
    return roi.lims

