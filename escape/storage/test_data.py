import numpy as np
from dask import array as da
from numpy.random import poisson, randn
from scipy.interpolate import PchipInterpolator

from .storage import Array


def _relnoise(x, fac=1000):
    return 1.0 / np.sqrt(fac) / np.sqrt(np.abs(x) + 1e-10)


class TestData:
    """Generates realistic synthetic pump-probe data with shot noise and slow drift."""

    def __init__(self, tstart=-1, tstepsize=0.2, tjitter=0.2, step_length=2000):
        self.tstart = tstart
        self.tstepsize = tstepsize
        self.tjitter = tjitter
        self.step_length = step_length
        self.pump_drops = 5  # inverse probability of pump being on
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
        self.timetool = 0.0
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
                    [
                        self._drift_nodes[1:],
                        poisson(self.driftTimescale) + self._drift_nodes[-1],
                    ]
                )
                self._drift_data = np.hstack([self._drift_data[1:], randn(1)])
                self._drift_itp = PchipInterpolator(self._drift_nodes, self._drift_data)

    def generateData(self, pulse_id):
        self.pulseId = float(pulse_id)
        self._update_drift()

        drift = float(self._drift_itp(pulse_id))
        step_index = int(pulse_id // self.step_length)
        t_nominal = self.tstart + self.tstepsize * step_index
        timetool = float(self.tjitter * np.random.randn())
        t = t_nominal + timetool

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
        self.timetool = timetool
        self.i_pump = i_pump
        self.drift = drift

        return {
            "i0": i0,
            "i": i,
            "t": t,
            "timetool": timetool,
            "i_pump": i_pump,
            "pump_on": self.pump_on,
            "pulse_id": float(pulse_id),
            "drift": drift,
        }

    def getPar(self, pulse_id, parameter=None):
        if pulse_id != self.pulseId:
            self.generateData(pulse_id)
        return getattr(self, parameter)


def get_test_data(N_pulses=1e4, as_array=True, as_da=True, step_length=200):
    """Generate a synthetic pump-probe dataset for testing/examples.

    Simulates a delay scan (steps of ``t_nominal`` spaced by ``tstepsize``,
    starting at ``tstart``) with per-shot timetool jitter, shot noise, a
    slow intensity drift, and a pump on/off pattern -- see
    :class:`TestData` for the generative model.

    Parameters
    ----------
    N_pulses : int, optional
        Total number of simulated shots. Default 10000.
    as_array : bool, optional
        If True (default), wrap each channel as an :class:`escape.Array`
        sharing the delay scan's step structure. If False, return plain
        numpy arrays.
    as_da : bool, optional
        Only used when ``as_array`` is True: if True (default), back each
        Array with a dask array; if False, use a plain in-memory numpy
        array.
    step_length : int, optional
        Number of shots per scan step. Default 200.

    Returns
    -------
    dict
        Keys are channel names, values are :class:`escape.Array` (or numpy
        array, if ``as_array=False``), each with ``N_pulses`` events:

        - ``"i0"`` : incoming intensity monitor.
        - ``"i"`` : signal intensity (pump-probe response riding on ``i0``).
        - ``"t"`` : fully corrected per-shot delay, i.e. ``t_nominal +
          timetool`` -- the quantity you'd bin a real pump-probe scan on.
        - ``"timetool"`` : per-shot timetool jitter only (mean zero), *not*
          including the nominal per-step delay -- pair this with
          ``time_vec="auto"`` in :meth:`~escape.storage.storage_tools.ArrayTools.timetool_binning`,
          which adds the nominal delay back in per scan step.
        - ``"i_pump"`` : per-shot effective pump strength.
        - ``"pump_on"`` : 1.0/0.0 pump on/off flag.
        - ``"drift"`` : slow intensity drift shared by nearby shots.
    """
    N_pulses = int(N_pulses)
    td = TestData(step_length=step_length)
    d = {
        key: np.asarray(tl)
        for key, tl in zip(
            td.generateData(0).keys(),
            zip(*[list(td.generateData(n).values()) for n in range(N_pulses)]),
        )
    }

    n_full_steps, remainder = divmod(N_pulses, step_length)
    step_lengths = [step_length] * n_full_steps + ([remainder] if remainder else [])
    t_nominal = [td.tstart + td.tstepsize * n for n in range(len(step_lengths))]
    scan_parameter = {"t": {"values": t_nominal}}

    if as_array:
        pulse_id = d.pop("pulse_id")
        if as_da:
            d = {
                key: Array(
                    data=da.from_array(arr),
                    index=pulse_id,
                    step_lengths=step_lengths,
                    parameter=scan_parameter,
                )
                for key, arr in d.items()
            }
        else:
            d = {
                key: Array(
                    data=arr,
                    index=pulse_id,
                    step_lengths=step_lengths,
                    parameter=scan_parameter,
                )
                for key, arr in d.items()
            }
    return d
