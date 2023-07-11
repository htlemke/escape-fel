import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from threading import Thread
import time


class HistPlot:
    def __init__(
        self,
        data,
        axes=None,
        label=None,
        scanVariable=0,
        autosetAxlabel=True,
        alpha=0.3,
    ):
        self.data = data
        if axes is None:
            axes = plt.gca()
        if label is None:
            label = self.data.name
        self.axes = axes
        self.label = label
        self.scanVariable = scanVariable
        self.autosetAxlabel = autosetAxlabel
        self.alpha = alpha
        self.autoscale = True
        self.isUpdating = False
        self.drawn = None
        self.updateInterval = 0.5
        self._minimumSleep = 0.01

    def _getplotData(self):
        x = np.asarray(self.data.scan[self.scanVariable])
        y = np.asarray(self.data.lens())
        yerr = np.sqrt(y)
        sorter = x.argsort()
        # TODO: maybe histogram 0 bin issue
        return x[sorter], y[sorter], yerr[sorter]

    def plot(self):
        self._lastPlotRequest = time.time()
        x, y, yerr = self._getplotData()
        props = next(self.axes._get_lines.prop_cycler)
        err = self.axes.fill_between(
            x,
            y + yerr,
            np.max(np.vstack([y - yerr, np.zeros_like(y)]), axis=0),
            color=props["color"],
            alpha=self.alpha,
            step="mid",
            lw=0,
        )
        line = self.axes.step(x, y, where="mid", label=self.label, **props)[0]
        self.drawn = dict(err=err, line=line)
        if self.autosetAxlabel:
            self.axes.set_xlabel(
                "%s / %s"
                % (
                    self.data.scan._parameters[self.scanVariable].name,
                    self.data.scan._parameters[self.scanVariable].unit,
                )
            )
            self.axes.set_ylabel("%s / %s" % (self.data.name, self.data.unit))
        plt.draw()
        self._lastPlottingTime = time.time() - self._lastPlotRequest

    def replot(self):
        self._lastPlotRequest = time.time()
        x, y, yerr = self._getplotData()
        v = self._generateYerrVertices(x, y, yerr)
        # self.drawn['err'].get_paths()[0].vertices = v
        color = self.drawn["err"].get_facecolor()
        self.drawn["err"].remove()
        self.drawn["err"] = self.axes.fill_between(
            x,
            y + yerr,
            np.max(np.vstack([y - yerr, np.zeros_like(y)]), axis=0),
            color=color,
            alpha=self.alpha,
            step="mid",
            lw=0,
        )
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        if self.autoscale:
            self.axes.relim()
            self.axes.autoscale_view(True, True, True)
        self._lastPlottingTime = time.time() - self._lastPlotRequest

    def _generateYerrVertices(self, x, y, yerr):
        try:
            y0 = y + yerr
            y1 = y - yerr
            xv = np.hstack([x[0], x, x[::-1], x[0]])
            yv = np.hstack([y0[0], y0, y1[::-1], y1[0]])
        except:
            xv = []
            yv = []

        return np.vstack([xv, yv]).T

    def _updateLoop(self, interval=0.5):
        # if self.drawn is None:
        # self.plot()
        while self.isUpdating:
            self.replot()
            time.sleep(
                max(self.updateInterval - self._lastPlottingTime, self._minimumSleep)
            )

    def updateContinuously(self, interval=0.5):
        self.updateInterval = interval
        self.isUpdating = True
        self._activeUpdatingThread = Thread(target=self._updateLoop)
        self._activeUpdatingThread.setDaemon(True)
        self._activeUpdatingThread.start()


class Plot:
    def __init__(
        self,
        data,
        axes=None,
        label=None,
        scanVariable=0,
        autosetAxlabel=True,
        errPercentiles=[69.3, 95.0],
        step=False,
        alpha=0.3,
    ):
        self.data = data
        if axes is None:
            axes = plt.gca()
        if label is None:
            label = self.data.name
        self.axes = axes
        self.label = label
        self.scanVariable = scanVariable
        self.autosetAxlabel = autosetAxlabel
        self.alpha = alpha
        self.autoscale = True
        self.isUpdating = False
        self.drawn = None
        self.updateInterval = 0.5
        self._minimumSleep = 0.01
        self.errPercentiles = errPercentiles
        self.step = step

    def _getplotData(self):
        try:
            x = np.asarray(self.data.scan[self.scanVariable])
            sorter = x.argsort()
            y = np.asarray(self.data.median())[sorter]
            yerr = [
                (np.asarray(self.data.centerPerc(tP)).T)[:, sorter]
                for tP in self.errPercentiles
            ]
            lens = np.asarray(self.data.lens())[sorter]
            yerr = [(tyerr - y) / np.sqrt(lens) + y for tyerr in yerr]
            return x[sorter], y, yerr
        except:
            return [], [], []

    def plot(self):
        self._lastPlotRequest = time.time()
        x, y, yerr = self._getplotData()
        if len(x) == 0:
            self._lastPlottingTime = time.time() - self._lastPlotRequest
            return
        props = next(self.axes._get_lines.prop_cycler)
        errs = []
        if self.step:
            for n, eP in enumerate(self.errPercentiles):
                errs.append(
                    self.axes.fill_between(
                        x,
                        yerr[n][0],
                        yerr[n][1],
                        color=props["color"],
                        alpha=self.alpha,
                        step="mid",
                        lw=0,
                    )
                )
            line = self.axes.step(x, y, where="mid", label=self.label, **props)[0]
        else:
            for n, eP in enumerate(self.errPercentiles):
                errs.append(
                    self.axes.fill_between(
                        x,
                        yerr[n][0],
                        yerr[n][1],
                        color=props["color"],
                        alpha=self.alpha,
                        step="mid",
                        lw=0,
                    )
                )
            line = self.axes.plot(x, y, ".-", label=self.label, **props)[0]
        self.drawn = dict(err=errs, line=line)
        if self.autosetAxlabel:
            self.axes.set_xlabel(
                "%s / %s"
                % (
                    self.data.scan._parameters[self.scanVariable].name,
                    self.data.scan._parameters[self.scanVariable].unit,
                )
            )
            self.axes.set_ylabel("%s / %s" % (self.data.name, self.data.unit))
        plt.draw_all()
        self._lastPlottingTime = time.time() - self._lastPlotRequest

    def replot(self):
        self._lastPlotRequest = time.time()
        x, y, yerr = self._getplotData()
        if self.drawn is None:
            self.plot()
            return
        errs = []
        if self.step:
            for tyerr, coll in zip(yerr, self.drawn["err"]):
                color = coll.get_facecolor()
                coll.remove()
                errs.append(
                    self.axes.fill_between(
                        x,
                        tyerr[0],
                        tyerr[1],
                        color=color,
                        alpha=self.alpha,
                        step="mid",
                        lw=0,
                    )
                )
        else:
            for tyerr, coll in zip(yerr, self.drawn["err"]):
                color = coll.get_facecolor()
                coll.remove()
                errs.append(
                    self.axes.fill_between(
                        x, tyerr[0], tyerr[1], color=color, alpha=self.alpha, lw=0
                    )
                )

        self.drawn["err"] = errs
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        if self.autoscale:
            self.axes.relim()
            self.axes.autoscale_view(True, True, True)
        plt.draw_all()
        self._lastPlottingTime = time.time() - self._lastPlotRequest

    def _updateLoop(self, interval=0.5):
        # if self.drawn is None:
        # self.plot()
        while self.isUpdating:
            self.replot()
            time.sleep(
                max(self.updateInterval - self._lastPlottingTime, self._minimumSleep)
            )

    def updateContinuously(self, interval=0.5):
        self.updateInterval = interval
        self.isUpdating = True
        self._activeUpdatingThread = Thread(target=self._updateLoop)
        self._activeUpdatingThread.setDaemon(True)
        self._activeUpdatingThread.start()


class PlotCorrelation:
    def __init__(
        self,
        data_x,
        data_y,
        Nlast=200,
        axes=None,
        label=None,
        scanVariable=0,
        autosetAxlabel=True,
    ):
        self.data_x = data_x
        self.data_y = data_y
        self.Nlast = Nlast
        if axes is None:
            axes = plt.gca()
        if label is None:
            label = self.data_y.name
        self.axes = axes
        self.label = label
        self.autosetAxlabel = autosetAxlabel
        self.autoscale = True
        self.isUpdating = False
        self.drawn = None
        self.updateInterval = 0.5
        self._minimumSleep = 0.01

    def _getplotData(self, Nlast=None):
        if not Nlast is None:
            self.Nlast = Nlast
        flatten = lambda l: [item for sublist in l for item in sublist]
        # try:
        xi = np.asarray(flatten(self.data_x.eventIds))
        yi = np.asarray(flatten(self.data_y.eventIds))
        xd = np.asarray(flatten(self.data_x.data))
        yd = np.asarray(flatten(self.data_y.data))

        xsel = np.in1d(xi, yi)
        ysel = np.in1d(yi, xi)

        xi = xi[np.where(xsel)]
        xd = xd[np.where(xsel)]
        yi = yi[np.where(ysel)]
        yd = yd[np.where(ysel)]

        mx = np.max(xi)
        xsel = xi > (mx - self.Nlast)
        ysel = yi > (mx - self.Nlast)

        xind = np.argsort(xi[xsel])
        xd = xd[xsel][xind]
        yind = np.argsort(yi[ysel])
        yd = yd[ysel][yind]
        return xd, yd
        # except:
        # return [],[]

    def plot(self):
        self._lastPlotRequest = time.time()
        x, y = self._getplotData()
        if len(x) == 0:
            self._lastPlottingTime = time.time() - self._lastPlotRequest
            return
        props = next(self.axes._get_lines.prop_cycler)
        line = self.axes.plot(x, y, ".", label=self.label, **props)[0]
        self.drawn = dict(line=line)
        if self.autosetAxlabel:
            self.axes.set_xlabel("%s / %s" % (self.data_x.name, self.data_x.unit))
            self.axes.set_ylabel("%s / %s" % (self.data_y.name, self.data_y.unit))
        plt.draw_all()
        self._lastPlottingTime = time.time() - self._lastPlotRequest

    def replot(self):
        self._lastPlotRequest = time.time()
        x, y = self._getplotData()
        if self.drawn is None:
            self.plot()
            return
        self.drawn["line"].set_xdata(x)
        self.drawn["line"].set_ydata(y)
        if self.autoscale:
            self.axes.relim()
            self.axes.autoscale_view(True, True, True)
        plt.draw_all()
        self._lastPlottingTime = time.time() - self._lastPlotRequest

    def _updateLoop(self, interval=0.5):
        # if self.drawn is None:
        # self.plot()
        while self.isUpdating:
            self.replot()
            time.sleep(
                max(self.updateInterval - self._lastPlottingTime, self._minimumSleep)
            )

    def updateContinuously(self, interval=0.5):
        self.updateInterval = interval
        self.isUpdating = True
        self._activeUpdatingThread = Thread(target=self._updateLoop)
        self._activeUpdatingThread.setDaemon(True)
        self._activeUpdatingThread.start()
