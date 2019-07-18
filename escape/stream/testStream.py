from bsread import Source
from bsread.sender import Sender
import numpy as np
from numpy.random import poisson, randn
from functools import partial
from scipy.interpolate import PchipInterpolator


def relnoise(x, fac=1000):
    return 1 / np.sqrt(fac) / np.sqrt(x)


class TestData:
    def __init__(self, tstart=-1, tend=10):
        self.tstart = tstart
        self.tend = tend
        self.pump_drops = 5
        self.pump_frac = 0.05
        self.pump_noise = 0.1
        self.sig_noise = 0.02
        self.pulseId = -1
        self.driftTimescale = 500  # in "pulses"
        self.driftNodes = None
        self.driftData = None

    def updateDriftNodes(self):
        if self.driftNodes is None:
            delta = np.cumsum(poisson(self.driftTimescale, 4))
            delta -= delta[1]
            delta += np.int(self.pulseId)
            self.driftNodes = delta
            self.driftData = randn(4)
            self.driftInterpolator = PchipInterpolator(self.driftNodes, self.driftData)
        else:
            while self.pulseId > self.driftNodes[2]:
                self.driftNodes = np.hstack(
                    [
                        self.driftNodes[1:],
                        poisson(self.driftTimescale) + self.driftNodes[-1],
                    ]
                )
                self.driftData = np.hstack([self.driftData[1:], randn(1)])
                self.driftInterpolator = PchipInterpolator(
                    self.driftNodes, self.driftData
                )

    def generateData(self, pulse_id):
        self.pulseId = float(pulse_id)
        self.updateDriftNodes()

        drift = self.driftInterpolator(pulse_id)
        t = -(self.tstart - self.tend) * np.random.random_sample() + self.tstart
        tr = pulse_id * 0.01

        i0 = np.random.gamma(2.3, 1)
        sig = 1 - np.cos(2 * np.pi / 0.7 * t) * np.exp(-t / 2)
        pump_on = not np.random.poisson(1.0 / self.pump_drops)
        i_pump = self.pump_frac * (float(pump_on) + self.pump_noise * np.random.randn())
        if t < 0:
            i_pump = 0.0

        i_drift = 1 + 0.07 * drift
        i = i_drift * (i0 * (1 + i_pump * sig))
        i += relnoise(i) * np.random.randn()
        if np.isnan(i):
            i = 0.0
        self.i0 = i0
        self.i = i
        self.pump_on = float(pump_on)
        self.t = t
        self.i_pump = i_pump
        self.drift = drift
        return {"i0":i0, "i":i, "t":t, "i_pump":i_pump, "pump_on":pump_on, "pulse_id":pulse_id, "drift":drift}

    def getPar(self, pulseId, parameter=None):
        if not pulseId == self.pulseId:
            self.generateData(pulseId)
        return self.__dict__[parameter]


pars = ["i0", "i", "t", "i_pump", "pump_on", "pulseId", "drift"]


def createStream():
    s = TestData()
    generator = Sender()
    for par in pars:
        generator.add_channel(par, partial(s.getPar, parameter=par))
    generator.generate_stream()


class StreamReader:
    def __init__(self, host="localhost", port=9999):
        self.source = Source("localhost", 9999)
        self.s = self.source.connect()

    def readStream(self, Nevents):
        data = []
        for n in range(Nevents):
            m = self.s.receive()
            data.append([m.data.data[par].value for par in pars])

        return np.asarray(data)
