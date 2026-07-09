"""
Test stream generator using bsread Sender.

Fix over original testStream.py:
  - The old code used add_channel(name, callback) + generate_stream(), which omits dtype
    metadata from the data header.  Receivers that strictly parse the header then reject
    or misinterpret the values.
  - The new createStream() pre-registers channels via add_channel_from_value() (which
    infers dtype/shape from a real Python value) and then sends explicit data dicts in a
    plain loop, keeping the header stable across all messages.

Run in a separate multiprocessing.Process (not Thread) to avoid GIL contention with zmq:
    from multiprocessing import Process
    p = Process(target=createStream)
    p.start()
"""

import time
import numpy as np
from numpy.random import poisson, randn
from scipy.interpolate import PchipInterpolator


def _relnoise(x, fac=1000):
    return 1.0 / np.sqrt(fac) / np.sqrt(np.abs(x) + 1e-10)


class TestData:
    """Generates realistic synthetic pump-probe data with shot noise and slow drift."""

    def __init__(self, tstart=-1, tend=10):
        self.tstart = tstart
        self.tend = tend
        self.pump_drops = 5        # inverse probability of pump being on
        self.pump_frac = 0.05
        self.pump_noise = 0.1
        self.pulseId = -1
        self.driftTimescale = 500  # pulse-period units
        self._drift_nodes = None
        self._drift_data = None
        self._drift_itp = None

        # Current-event cached values (set by generateData)
        self.i0 = 0.0
        self.i = 0.0
        self.pump_on = 0.0
        self.t = 0.0
        self.i_pump = 0.0
        self.drift = 0.0

    def _update_drift(self):
        pid = float(self.pulseId)
        if self._drift_nodes is None:
            delta = np.cumsum(poisson(self.driftTimescale, 4)).astype(float)
            delta -= delta[1]
            delta += pid
            self._drift_nodes = delta
            self._drift_data = randn(4)
            self._drift_itp = PchipInterpolator(self._drift_nodes, self._drift_data)
        else:
            while pid > self._drift_nodes[2]:
                self._drift_nodes = np.hstack(
                    [self._drift_nodes[1:],
                     poisson(self.driftTimescale) + self._drift_nodes[-1]]
                )
                self._drift_data = np.hstack([self._drift_data[1:], randn(1)])
                self._drift_itp = PchipInterpolator(self._drift_nodes, self._drift_data)

    def generateData(self, pulse_id):
        self.pulseId = float(pulse_id)
        self._update_drift()

        drift = float(self._drift_itp(pulse_id))
        t = float(
            -(self.tstart - self.tend) * np.random.random_sample() + self.tstart
        )

        i0 = float(np.random.gamma(2.3, 1))
        sig = 1.0 - np.cos(2 * np.pi / 0.7 * t) * np.exp(-t / 2)
        pump_on = bool(not np.random.poisson(1.0 / self.pump_drops))
        i_pump = float(
            self.pump_frac * (float(pump_on) + self.pump_noise * np.random.randn())
        )
        if t < 0:
            i_pump = 0.0

        i_drift = 1.0 + 0.07 * drift
        i = float(i_drift * (i0 * (1.0 + i_pump * sig)))
        i += _relnoise(i) * np.random.randn()
        if np.isnan(i):
            i = 0.0

        self.i0 = i0
        self.i = i
        self.pump_on = float(pump_on)
        self.t = t
        self.i_pump = i_pump
        self.drift = drift

        return {
            "i0": i0,
            "i": i,
            "t": t,
            "i_pump": i_pump,
            "pump_on": self.pump_on,
            "pulse_id": float(pulse_id),
            "drift": drift,
        }

    def getPar(self, pulse_id, parameter=None):
        if pulse_id != self.pulseId:
            self.generateData(pulse_id)
        return getattr(self, parameter)


# Names of channels the test stream broadcasts.
CHANNEL_NAMES = ["i0", "i", "t", "i_pump", "pump_on", "pulse_id", "drift"]


def createStream(port=9999, interval=0.01):
    """Broadcast synthetic pump-probe data on localhost via bsread.

    Pre-registers all channels so that dtype metadata is stable across all messages,
    then sends explicit data dicts in a loop.  This is compatible with bsread >=4.x.

    Parameters
    ----------
    port : int
        ZMQ port to bind to (default 9999).
    interval : float
        Seconds between pulses (default 0.01 → ~100 Hz).
    """
    from bsread.sender import Sender

    s = TestData()
    # Generate one event to learn dtypes/shapes before opening the socket.
    first_data = s.generateData(0)

    sender = Sender(port=port)
    for name, value in first_data.items():
        sender.add_channel_from_value(name, value)
    sender.open()

    # Pre-compute channel order so we can send positional args without check_data
    # overhead (avoids recreating the data header on every pulse).
    channel_order = list(sender.channels.keys())

    pulse_id = 1
    try:
        while True:
            data = s.generateData(pulse_id)
            values = [data[name] for name in channel_order]
            sender.send(*values, pulse_id=pulse_id, check_data=False)
            pulse_id += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


class StreamReader:
    """Simple blocking reader for the local test stream (diagnostic use)."""

    def __init__(self, host="localhost", port=9999):
        from bsread import Source
        self.source = Source(host=host, port=port, all_channels=True)
        self.source.connect()

    def readStream(self, n_events):
        data = []
        for _ in range(n_events):
            m = self.source.receive()
            row = {name: m.data.data[name].value for name in m.data.data}
            data.append(row)
        return data

    def close(self):
        self.source.disconnect()
