"""Synthetic data generators for escape documentation examples and testing.

These functions create realistic FEL-like event data without requiring external
data sources, making them suitable for documentation notebooks and unit tests.
All generators accept a ``seed`` parameter for reproducible output.
"""

import numpy as np
from .storage import Array


def make_array(
    n_events: int = 2000,
    data_fn=None,
    name: str = "signal",
    sparse_ids: bool = False,
    seed: int = None,
) -> Array:
    """Create a simple 1-D escape Array with synthetic scalar data.

    Parameters
    ----------
    n_events : int
        Number of events (pulses).
    data_fn : callable, optional
        ``f(index) -> values``.  If *None*, standard-normal noise is used.
    name : str
        Name tag stored in the returned Array.
    sparse_ids : bool
        If *True*, ~5 % of pulse IDs are randomly dropped to mimic real data
        where not every instrument records every pulse.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    escape.Array
        1-D Array with shape ``(n_events,)`` (or fewer if ``sparse_ids=True``).

    Examples
    --------
    >>> from escape.storage.example_data import make_array
    >>> import numpy as np
    >>> sig = make_array(1000, lambda ix: np.sin(ix / 200.0), seed=0)
    >>> sig.shape
    (1000,)
    """
    rng = np.random.default_rng(seed)
    ids = np.arange(n_events, dtype=np.int64)
    if sparse_ids:
        mask = rng.random(n_events) > 0.05
        ids = ids[mask]
    if data_fn is None:
        data = rng.standard_normal(len(ids)).astype(np.float32)
    else:
        data = np.asarray(data_fn(ids), dtype=np.float32)
    return Array(data=data, index=ids, step_lengths=[len(ids)], name=name)


def make_scan(
    n_steps: int = 10,
    n_events_per_step: int = 500,
    scan_par_name: str = "delay",
    scan_par_values=None,
    signal_fn=None,
    noise: float = 0.1,
    name: str = "signal",
    seed: int = None,
) -> Array:
    """Create a multi-step scan escape Array with per-step parameter metadata.

    Generates realistic data where a 1-D scalar signal depends on a scan
    parameter (e.g. pump-probe delay) plus shot-to-shot noise.

    Parameters
    ----------
    n_steps : int
        Number of scan steps.
    n_events_per_step : int
        Events recorded per step.
    scan_par_name : str
        Name of the scanned parameter (e.g. ``"delay_ps"``).
    scan_par_values : array-like, optional
        Values of the scan parameter per step.  If *None*, equally spaced
        values in ``[0, 1]`` are used.
    signal_fn : callable, optional
        ``f(par_value) -> float`` giving the mean signal at each step.
        If *None* a simple cosine response is used.
    noise : float
        Standard deviation of additive Gaussian shot-to-shot noise.
    name : str
        Name tag for the returned Array.
    seed : int, optional
        Random seed.

    Returns
    -------
    escape.Array
        1-D Array with ``n_steps * n_events_per_step`` events and scan metadata.

    Examples
    --------
    >>> import numpy as np
    >>> from escape.storage.example_data import make_scan
    >>> delays = np.linspace(-0.5e-12, 2e-12, 20)
    >>> sig = make_scan(
    ...     n_steps=20,
    ...     n_events_per_step=300,
    ...     scan_par_name="delay_s",
    ...     scan_par_values=delays,
    ...     signal_fn=lambda t: 1.0 - float(t > 0) * np.exp(-t / 0.5e-12),
    ...     noise=0.05,
    ...     name="bragg_intensity",
    ...     seed=0,
    ... )
    >>> len(sig.scan)
    20
    """
    rng = np.random.default_rng(seed)
    if scan_par_values is None:
        scan_par_values = np.linspace(0, 1, n_steps)
    scan_par_values = np.asarray(scan_par_values, dtype=float)

    data_parts, index_parts = [], []
    step_lengths = []
    pulse_id = 0

    for step_val in scan_par_values:
        ids = np.arange(pulse_id, pulse_id + n_events_per_step, dtype=np.int64)
        if signal_fn is None:
            mean_val = 1.0 - 0.3 * np.cos(2 * np.pi * step_val)
        else:
            mean_val = float(signal_fn(step_val))
        vals = mean_val + noise * rng.standard_normal(n_events_per_step)
        data_parts.append(vals.astype(np.float32))
        index_parts.append(ids)
        step_lengths.append(n_events_per_step)
        pulse_id += n_events_per_step

    data = np.concatenate(data_parts)
    index = np.concatenate(index_parts)
    parameter = {scan_par_name: {"values": list(scan_par_values)}}
    return Array(
        data=data,
        index=index,
        step_lengths=step_lengths,
        parameter=parameter,
        name=name,
    )


def make_pump_probe_scan(
    n_steps: int = 15,
    n_events_per_step: int = 600,
    delays=None,
    response_fn=None,
    i0_noise: float = 0.05,
    noise: float = 0.08,
    pump_fraction: float = 0.5,
    seed: int = None,
):
    """Create synthetic pump-probe scan data with an intensity reference (I0).

    Models a typical FEL pump-probe experiment where each scan step corresponds
    to a nominal delay, and within each step roughly ``pump_fraction`` of shots
    are laser-pumped while the rest serve as unpumped references.

    Parameters
    ----------
    n_steps : int
        Number of delay steps.
    n_events_per_step : int
        Total events per step (split between pump-on and pump-off).
    delays : array-like, optional
        Delay values in seconds.  Defaults to ``n_steps`` log-spaced values
        between −0.2 ps and 5 ps.
    response_fn : callable, optional
        ``f(t_seconds) -> relative_change`` for the pump signal.  Defaults to
        an exponential rise with 500 fs time constant and 10 % amplitude.
    i0_noise : float
        Fractional (relative) noise on the I0 reference.
    noise : float
        Fractional shot-to-shot noise on the detector signal.
    pump_fraction : float
        Fraction of shots per step that are pump-ON.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple of escape.Array
        ``(signal, i0, pump_on, delay)``

        * *signal* – detector signal.
        * *i0* – incoming X-ray intensity.
        * *pump_on* – boolean flag (True = laser was fired).
        * *delay* – nominal delay value repeated for every event.

    Examples
    --------
    >>> from escape.storage.example_data import make_pump_probe_scan
    >>> sig, i0, pump_on, delay = make_pump_probe_scan(n_steps=10, seed=0)
    >>> # normalised per-step pump/probe ratio:
    >>> ratio = (sig[~pump_on] / i0[~pump_on]).scan.nanmean()
    """
    rng = np.random.default_rng(seed)
    if delays is None:
        delays = np.concatenate([
            np.array([-0.2e-12]),
            np.geomspace(0.05e-12, 5e-12, n_steps - 1),
        ])
    delays = np.asarray(delays, dtype=float)
    n_steps = len(delays)

    if response_fn is None:
        tau = 0.5e-12

        def response_fn(t):
            return 0.1 * np.where(t > 0, 1.0 - np.exp(-t / tau), 0.0)

    sig_parts, i0_parts, pump_parts = [], [], []
    index_parts = []
    step_lengths = []
    pulse_id = 0

    for t in delays:
        n = n_events_per_step
        ids = np.arange(pulse_id, pulse_id + n, dtype=np.int64)
        is_pump = rng.random(n) < pump_fraction

        i0_vals = 1.0 + i0_noise * rng.standard_normal(n)
        sig_base = i0_vals * (1.0 + noise * rng.standard_normal(n))
        delta = float(response_fn(t))
        sig_vals = sig_base.copy()
        sig_vals[is_pump] *= 1.0 + delta

        sig_parts.append(sig_vals.astype(np.float32))
        i0_parts.append(i0_vals.astype(np.float32))
        pump_parts.append(is_pump)
        index_parts.append(ids)
        step_lengths.append(n)
        pulse_id += n

    index_all = np.concatenate(index_parts)
    parameter = {"delay_s": {"values": list(delays)}}
    kwargs = dict(index=index_all, step_lengths=step_lengths, parameter=parameter)

    sig_arr = Array(data=np.concatenate(sig_parts), name="signal", **kwargs)
    i0_arr  = Array(data=np.concatenate(i0_parts),  name="i0",     **kwargs)
    pump_arr = Array(data=np.concatenate(pump_parts).astype(bool), name="pump_on", **kwargs)
    delay_arr = Array(
        data=np.repeat(delays, n_events_per_step).astype(np.float64),
        name="delay_s",
        **kwargs,
    )
    return sig_arr, i0_arr, pump_arr, delay_arr


def make_image_scan(
    n_steps: int = 5,
    n_events_per_step: int = 100,
    image_shape=(64, 64),
    peak_center=(32, 32),
    scan_par_name: str = "motor_mm",
    scan_par_values=None,
    seed: int = None,
) -> Array:
    """Create a scan Array with 2-D image data per event.

    Models a Bragg peak that shifts position as a scan motor moves.  Useful
    for demonstrating ROI selection and 2-D data processing.

    Parameters
    ----------
    n_steps : int
        Number of scan steps.
    n_events_per_step : int
        Images per step.
    image_shape : tuple of int
        Pixel dimensions ``(rows, cols)``.
    peak_center : tuple of int
        Default peak centre in pixels ``(row, col)`` for step 0.
        The peak shifts by 1.5 pixels per step along the row axis.
    scan_par_name : str
        Name of the scanned parameter.
    scan_par_values : array-like, optional
        Values per step.  Defaults to integers ``0, 1, …, n_steps-1``.
    seed : int, optional
        Random seed.

    Returns
    -------
    escape.Array
        Array with shape ``(n_steps * n_events_per_step, *image_shape)``.

    Examples
    --------
    >>> from escape.storage.example_data import make_image_scan
    >>> imgs = make_image_scan(n_steps=3, n_events_per_step=20, seed=0)
    >>> imgs.shape
    (60, 64, 64)
    >>> mean_step0 = imgs.scan[0].mean(axis=0)
    """
    rng = np.random.default_rng(seed)
    if scan_par_values is None:
        scan_par_values = np.arange(n_steps, dtype=float)
    scan_par_values = np.asarray(scan_par_values, dtype=float)

    rows = np.arange(image_shape[0])
    cols = np.arange(image_shape[1])
    c, r = np.meshgrid(cols, rows)

    all_images = []
    index_parts = []
    step_lengths = []
    pulse_id = 0

    sigma = 4.0
    for step_i, par_val in enumerate(scan_par_values):
        pr = peak_center[0] + step_i * 1.5
        pc = peak_center[1]
        peak = 50.0 * np.exp(-((r - pr) ** 2 + (c - pc) ** 2) / (2 * sigma ** 2))

        n = n_events_per_step
        noise_scale = 1.0 + 0.05 * rng.standard_normal((n, 1, 1))
        poisson_bg  = rng.poisson(0.5, (n, *image_shape)).astype(np.float32)
        imgs = (peak[None] * noise_scale + poisson_bg).astype(np.float32)

        ids = np.arange(pulse_id, pulse_id + n, dtype=np.int64)
        all_images.append(imgs)
        index_parts.append(ids)
        step_lengths.append(n)
        pulse_id += n

    data = np.concatenate(all_images, axis=0)
    index = np.concatenate(index_parts)
    parameter = {scan_par_name: {"values": list(scan_par_values)}}
    return Array(
        data=data,
        index=index,
        step_lengths=step_lengths,
        parameter=parameter,
        name="detector_image",
    )
