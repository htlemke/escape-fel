import numpy as np
from dask import array as da
from scipy.interpolate import PchipInterpolator

from ..adaptive_scan.model import ImpulsiveResponse, Oscillation
from .storage import Array


def _relnoise(x, fac=1000):
    return 1.0 / np.sqrt(fac) / np.sqrt(np.abs(x) + 1e-10)


class TestData:
    """Generates realistic synthetic pump-probe data with shot noise and slow drift.

    Parameters
    ----------
    tstart, tstepsize : float
        Nominal delay of the first scan step and the step spacing.
    tjitter : float
        Standard deviation of the per-shot timing jitter (timetool).
    step_length : int
        Shots per scan step.
    response : ImpulsiveResponse or callable, optional
        Relative pump-induced change as a function of the true delay ``t``
        (only applied after the response's time zero ``t0``, default 0).  Defaults to
        :meth:`ImpulsiveResponse.default`, i.e.
        ``1 - cos(2 pi t / 0.7) * exp(-t / 2)``.  Build multi-mode signals
        from :class:`~escape.adaptive_scan.model.Oscillation` components.
    delays : array-like, optional
        Nominal delay of every scan step, overriding ``tstart``/``tstepsize``
        (arbitrary, e.g. adaptively chosen, delays).
    seed : int or numpy.random.Generator, optional
        Random seed for reproducible data.
    """

    def __init__(
        self,
        tstart=-1,
        tstepsize=0.2,
        tjitter=0.2,
        step_length=2000,
        response=None,
        delays=None,
        seed=None,
    ):
        self.tstart = tstart
        self.tstepsize = tstepsize
        self.tjitter = tjitter
        self.step_length = step_length
        self.response = ImpulsiveResponse.default() if response is None else response
        self.delays = None if delays is None else np.asarray(delays, float)
        self.rng = np.random.default_rng(seed)
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
            delta = np.cumsum(self.rng.poisson(self.driftTimescale, 4)).astype(float)
            delta -= delta[1]
            delta += pid
            self._drift_nodes = delta
            self._drift_data = self.rng.standard_normal(4)
            self._drift_itp = PchipInterpolator(self._drift_nodes, self._drift_data)
        else:
            while pid > self._drift_nodes[2]:
                self._drift_nodes = np.hstack(
                    [
                        self._drift_nodes[1:],
                        self.rng.poisson(self.driftTimescale) + self._drift_nodes[-1],
                    ]
                )
                self._drift_data = np.hstack(
                    [self._drift_data[1:], self.rng.standard_normal(1)]
                )
                self._drift_itp = PchipInterpolator(self._drift_nodes, self._drift_data)

    def generateData(self, pulse_id):
        self.pulseId = float(pulse_id)
        self._update_drift()

        drift = float(self._drift_itp(pulse_id))
        t_nominal = self.nominal_delay(int(pulse_id // self.step_length))
        timetool = float(self.tjitter * self.rng.standard_normal())
        t = t_nominal + timetool

        i0 = float(self.rng.gamma(2.3, 1))
        sig = float(self.response(t))
        pump_on = bool(not self.rng.poisson(1.0 / self.pump_drops))
        i_pump = float(
            self.pump_frac
            * (float(pump_on) + self.pump_noise * self.rng.standard_normal())
        )
        if t < getattr(self.response, "t0", 0.0):
            i_pump = 0.0

        i_drift = 1.0 + 0.07 * drift
        i = float(i_drift * (i0 * (1.0 + i_pump * sig)))
        i += _relnoise(i) * self.rng.standard_normal()
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

    def nominal_delay(self, step_index):
        """Nominal delay of scan step ``step_index``."""
        if self.delays is not None:
            return float(self.delays[step_index])
        return self.tstart + self.tstepsize * step_index

    def getPar(self, pulse_id, parameter=None):
        if pulse_id != self.pulseId:
            self.generateData(pulse_id)
        return getattr(self, parameter)


def get_test_data(
    N_pulses=1e4,
    as_array=True,
    as_da=True,
    step_length=200,
    response=None,
    delays=None,
    seed=None,
    **kwargs,
):
    """Generate a synthetic pump-probe dataset for testing/examples.

    Simulates a delay scan (steps of ``t_nominal`` spaced by ``tstepsize``,
    starting at ``tstart``) with per-shot timetool jitter, shot noise, a
    slow intensity drift, and a pump on/off pattern -- see
    :class:`TestData` for the generative model.  The pump response is an
    impulsive step with damped oscillation by default; pass ``response`` to
    simulate other (e.g. several overlapping, growing-in) oscillations.

    Parameters
    ----------
    N_pulses : int, optional
        Total number of simulated shots. Default 10000.  Ignored when
        ``delays`` is given (then ``len(delays) * step_length``).
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
    response : ImpulsiveResponse or callable, optional
        Pump-induced relative change vs. true delay; see :class:`TestData`.
    delays : array-like, optional
        Nominal delay of each scan step (any order/spacing, repeats allowed).
    seed : int, optional
        Random seed; the same seed reproduces the same dataset.
    **kwargs
        Further :class:`TestData` parameters (``tstart``, ``tstepsize``,
        ``tjitter``).

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
    td = TestData(
        step_length=step_length, response=response, delays=delays, seed=seed, **kwargs
    )
    if delays is not None:
        N_pulses = len(td.delays) * step_length
    N_pulses = int(N_pulses)
    d = {
        key: np.asarray(tl)
        for key, tl in zip(
            td.generateData(0).keys(),
            zip(*[list(td.generateData(n).values()) for n in range(N_pulses)]),
        )
    }

    n_full_steps, remainder = divmod(N_pulses, step_length)
    step_lengths = [step_length] * n_full_steps + ([remainder] if remainder else [])
    t_nominal = [td.nominal_delay(n) for n in range(len(step_lengths))]
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
