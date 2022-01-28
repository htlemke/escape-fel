import matplotlib.pyplot as plt
from time import sleep


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
