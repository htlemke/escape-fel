"""Synthetic data generators for escape documentation examples and testing.

These functions create realistic FEL-like event data without requiring external
data sources, making them suitable for documentation notebooks and unit tests.
All generators accept a ``seed`` parameter for reproducible output.

Available generators
--------------------
make_array          – plain 1-D Array (no scan)
make_scan           – 1-D scalar signal across a 1-D scan
make_pump_probe_scan – pump-probe scan returning (signal, i0, pump_on, delay)
make_image_scan     – 2-D image data per event across a scan (map-plot repr)
make_waveform_scan  – 1-D waveform per event across a scan (map-plot repr)
make_grid_scan      – scalar signal on a 2-D (or N-D) grid with an attached Grid
make_discrete_scan  – integer/discrete-valued scan (exercises repr edge-case handling)
make_detector_photon_stack – 2-D pixel-detector stack with charge-sharing photon
                             hits, for demonstrating droplet/cluster finding
"""

import numpy as np
import dask.array as da
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


def make_waveform_scan(
    n_steps: int = 12,
    n_events_per_step: int = 200,
    waveform_length: int = 128,
    scan_par_name: str = "delay_ps",
    scan_par_values=None,
    signal_fn=None,
    noise: float = 0.05,
    name: str = "tof_waveform",
    seed: int = None,
) -> Array:
    """Create a scan Array with a 1-D waveform per event.

    Models a time-of-flight or photodiode trace that changes shape across a
    scan.  This exercises the 2-D map-plot repr path (``ndim_nonzero == 2``).

    Parameters
    ----------
    n_steps : int
        Number of scan steps.
    n_events_per_step : int
        Waveforms recorded per step.
    waveform_length : int
        Number of samples in each waveform.
    scan_par_name : str
        Name of the scanned parameter.
    scan_par_values : array-like, optional
        Values per step.  Defaults to ``[0, 1, …, n_steps-1]``.
    signal_fn : callable, optional
        ``f(par_value, time_axis) -> waveform`` giving the mean waveform shape
        at each step.  Defaults to a Gaussian peak that shifts with scan par.
    noise : float
        Fractional shot-to-shot noise amplitude.
    name : str
        Name tag for the returned Array.
    seed : int, optional
        Random seed.

    Returns
    -------
    escape.Array
        Array with shape ``(n_steps * n_events_per_step, waveform_length)``.

    Examples
    --------
    >>> from escape.storage.example_data import make_waveform_scan
    >>> tof = make_waveform_scan(n_steps=8, waveform_length=64, seed=0)
    >>> tof.shape
    (1600, 64)
    >>> tof.scan.nanmean(plot=True)  # mean waveform vs scan parameter
    """
    rng = np.random.default_rng(seed)
    if scan_par_values is None:
        scan_par_values = np.arange(n_steps, dtype=float)
    scan_par_values = np.asarray(scan_par_values, dtype=float)

    t = np.linspace(0.0, 1.0, waveform_length)

    all_data, index_parts, step_lengths = [], [], []
    pulse_id = 0

    for par_val in scan_par_values:
        if signal_fn is None:
            span = float(scan_par_values.max() - scan_par_values.min())
            center = 0.2 + 0.5 * (par_val - scan_par_values.min()) / max(span, 1e-12)
            mean_wf = np.exp(-((t - center) ** 2) / (2 * 0.07 ** 2)).astype(np.float32)
        else:
            mean_wf = np.asarray(signal_fn(par_val, t), dtype=np.float32)

        n = n_events_per_step
        scale = 1.0 + noise * rng.standard_normal((n, 1))
        shot_noise = noise * rng.standard_normal((n, waveform_length)).astype(np.float32)
        waveforms = (mean_wf[None] * scale + shot_noise).astype(np.float32)

        ids = np.arange(pulse_id, pulse_id + n, dtype=np.int64)
        all_data.append(waveforms)
        index_parts.append(ids)
        step_lengths.append(n)
        pulse_id += n

    data = np.concatenate(all_data, axis=0)
    index = np.concatenate(index_parts)
    parameter = {scan_par_name: {"values": list(scan_par_values)}}
    return Array(
        data=data,
        index=index,
        step_lengths=step_lengths,
        parameter=parameter,
        name=name,
    )


def make_grid_scan(
    shape=(5, 8),
    n_events_per_step: int = 150,
    dim_names=("delay_ps", "motor_mm"),
    dim_ranges=((-0.5, 2.0), (0.0, 4.0)),
    signal_fn=None,
    noise: float = 0.08,
    name: str = "signal",
    seed: int = None,
) -> Array:
    """Create a scalar scan Array with a multi-dimensional grid structure.

    Steps are laid out on a full Cartesian grid so the attached
    :class:`~escape.storage.storage.Grid` object can reshape per-step
    aggregates into a 2-D (or N-D) image.  This exercises the grid heatmap
    repr path.

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions, e.g. ``(5, 8)`` for a 5-row × 8-column grid.
        Can be higher-dimensional (e.g. ``(3, 4, 5)``).
    n_events_per_step : int
        Events recorded at each grid point.
    dim_names : sequence of str
        Name of each grid axis (length must equal ``len(shape)``).
    dim_ranges : sequence of (float, float)
        ``(min, max)`` range of parameter values along each axis.
    signal_fn : callable, optional
        ``f(*par_values) -> float`` giving the mean signal at a grid point.
        Receives one positional argument per grid dimension.  Defaults to a
        Gaussian ridge along the first dimension.
    noise : float
        Standard deviation of additive Gaussian noise.
    name : str
        Name tag for the returned Array.
    seed : int, optional
        Random seed.

    Returns
    -------
    escape.Array
        1-D scalar Array with ``prod(shape) * n_events_per_step`` events and
        an attached Grid.

    Examples
    --------
    >>> from escape.storage.example_data import make_grid_scan
    >>> sig = make_grid_scan(shape=(6, 10), seed=0)
    >>> sig.grid.shape
    [6, 10]
    >>> sig.grid.nanmean(plot=True)          # 2-D heatmap
    >>> sig                                   # triggers grid repr plot
    """
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    positions = [
        np.linspace(lo, hi, s)
        for (lo, hi), s in zip(dim_ranges[:ndim], shape)
    ]

    grid_indices = list(np.ndindex(*shape))
    data_parts, index_parts, step_lengths = [], [], []
    scan_step_info = []
    par_values = {dn: [] for dn in dim_names[:ndim]}
    pulse_id = 0

    for grid_idx in grid_indices:
        coords = [float(positions[d][grid_idx[d]]) for d in range(ndim)]
        if signal_fn is None:
            # Gaussian ridge centred on middle of first axis, flat along others
            c0 = (dim_ranges[0][0] + dim_ranges[0][1]) / 2
            mean_val = float(np.exp(-((coords[0] - c0) ** 2) / (0.3 * (dim_ranges[0][1] - dim_ranges[0][0])) ** 2))
        else:
            mean_val = float(signal_fn(*coords))

        n = n_events_per_step
        ids = np.arange(pulse_id, pulse_id + n, dtype=np.int64)
        vals = (mean_val + noise * rng.standard_normal(n)).astype(np.float32)

        data_parts.append(vals)
        index_parts.append(ids)
        step_lengths.append(n)
        scan_step_info.append({"grid_index": list(grid_idx)})
        for d, dn in enumerate(dim_names[:ndim]):
            par_values[dn].append(coords[d])
        pulse_id += n

    data = np.concatenate(data_parts)
    index = np.concatenate(index_parts)
    parameter = {dn: {"values": par_values[dn]} for dn in dim_names[:ndim]}
    parameter["scan_step_info"] = {"values": scan_step_info}

    grid_specs = {
        "shape": list(shape),
        "positions": [np.asarray(p) for p in positions],
        "grid_dimension_names": list(dim_names[:ndim]),
    }
    return Array(
        data=data,
        index=index,
        step_lengths=step_lengths,
        parameter=parameter,
        grid_specs=grid_specs,
        name=name,
    )


def make_discrete_scan(
    n_steps: int = 10,
    n_events_per_step: int = 300,
    scan_par_name: str = "delay_ps",
    scan_par_values=None,
    values=(0, 1, 2, 3),
    name: str = "photon_count",
    seed: int = None,
) -> Array:
    """Create a scan with discrete integer values.

    Models a photon-counting detector or any channel with only a handful of
    distinct values.  The degenerate percentile range (all steps may share the
    same min/max) exercises the robustness fixes in the hist-plot repr path.

    Parameters
    ----------
    n_steps : int
        Number of scan steps.
    n_events_per_step : int
        Events per step.
    scan_par_name : str
        Name of the scanned parameter.
    scan_par_values : array-like, optional
        Defaults to equally spaced values in ``[0, 1]``.
    values : sequence of int
        Discrete output values that are randomly drawn per event.
    name : str
        Name tag.
    seed : int, optional
        Random seed.

    Returns
    -------
    escape.Array
        1-D integer Array useful for testing edge-case repr behaviour.

    Examples
    --------
    >>> from escape.storage.example_data import make_discrete_scan
    >>> cnt = make_discrete_scan(values=(0, 1), name="binary_flag", seed=0)
    >>> cnt.dtype
    dtype('int32')
    >>> cnt              # should render without errors despite degenerate range
    """
    rng = np.random.default_rng(seed)
    if scan_par_values is None:
        scan_par_values = np.linspace(0.0, 1.0, n_steps)
    scan_par_values = np.asarray(scan_par_values, dtype=float)
    values = np.asarray(values, dtype=np.int32)

    data_parts, index_parts, step_lengths = [], [], []
    pulse_id = 0
    for _ in scan_par_values:
        n = n_events_per_step
        ids = np.arange(pulse_id, pulse_id + n, dtype=np.int64)
        drawn = rng.choice(values, size=n).astype(np.int32)
        data_parts.append(drawn)
        index_parts.append(ids)
        step_lengths.append(n)
        pulse_id += n

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


def make_detector_photon_stack(
    n_events: int = 200,
    frame_shape=(128, 128),
    mean_photons: float = 6.0,
    photon_energy: float = 100.0,
    energy_jitter: float = 3.0,
    droplet_sigma: float = 0.25,
    read_noise: float = 1.5,
    chunk_size: int = 50,
    seed: int = None,
) -> Array:
    """Create a synthetic single-photon-counting detector stack.

    Models a pixel detector illuminated by a quasi-monochromatic source:
    each photon hit lands at a random sub-pixel position and deposits its
    energy as a small 2-D Gaussian charge cloud, integrated exactly over
    each pixel's area (via the Gaussian CDF / ``erf``, not just sampled at
    pixel centres) -- so how the energy splits between neighbours genuinely
    depends on where the hit lands relative to the pixel grid, the same way
    real charge-sharing does. Gaussian read noise is added on top.

    Useful for demonstrating
    :func:`escape.utilities_detectors.find_droplets`: a hit landing near a
    pixel corner splits its energy close to evenly between several
    neighbours, so the *single brightest pixel* alone systematically
    under-reports and scatters the photon energy depending on hit position,
    while the per-droplet *summed* intensity recovers the true (narrow)
    photon energy distribution regardless of where the hit landed.

    Parameters
    ----------
    n_events : int
        Number of detector frames (events/pulses).
    frame_shape : tuple of int
        Pixel dimensions ``(rows, cols)``.
    mean_photons : float
        Average number of photon hits per frame (Poisson-distributed).
    photon_energy : float
        Mean total deposited intensity per photon hit (arbitrary units).
    energy_jitter : float
        Standard deviation of the (small) shot-to-shot spread in deposited
        energy -- kept well below ``photon_energy`` so the demo isolates
        the pixel-splitting effect from intrinsic energy spread.
    droplet_sigma : float
        Gaussian width (pixels) of the charge cloud. Values well below 1
        pixel give a sharp, position-sensitive split (a few percent of hits
        land near a pixel corner and are split ~evenly four ways); larger
        values spread charge over more neighbours regardless of position.
    read_noise : float
        Standard deviation of Gaussian detector read noise added to every
        pixel.
    chunk_size : int
        Dask chunk length along the event axis.
    seed : int, optional
        Random seed.

    Returns
    -------
    escape.Array
        Dask-backed Array, shape ``(n_events, *frame_shape)``.

    Examples
    --------
    >>> from escape.storage.example_data import make_detector_photon_stack
    >>> frames = make_detector_photon_stack(n_events=50, seed=0)
    >>> frames.shape
    (50, 128, 128)
    """
    from scipy.special import erf

    def _gaussian_cdf(x, sigma):
        return 0.5 * (1.0 + erf(x / (sigma * np.sqrt(2.0))))

    rng = np.random.default_rng(seed)
    rows, cols = frame_shape
    data = rng.normal(0, read_noise, size=(n_events, rows, cols)).astype(np.float32)

    r_grid, c_grid = np.mgrid[0:rows, 0:cols]
    half_win = 4
    margin = half_win + 1

    for i in range(n_events):
        n_photons = rng.poisson(mean_photons)
        if n_photons == 0:
            continue
        pr = rng.uniform(margin, rows - margin, n_photons)
        pc = rng.uniform(margin, cols - margin, n_photons)
        amps = photon_energy + energy_jitter * rng.standard_normal(n_photons)
        for r0, c0, amp in zip(pr, pc, amps):
            r_lo, r_hi = int(round(r0)) - half_win, int(round(r0)) + half_win + 1
            c_lo, c_hi = int(round(c0)) - half_win, int(round(c0)) + half_win + 1
            rr = r_grid[r_lo:r_hi, c_lo:c_hi]
            cc = c_grid[r_lo:r_hi, c_lo:c_hi]
            # exact fraction of the 2-D Gaussian's mass landing in each
            # pixel's unit-square footprint, given the hit's sub-pixel
            # position (r0, c0) -- this is what makes the split depend on
            # where the photon lands relative to the pixel boundaries.
            frac_r = _gaussian_cdf(rr + 0.5 - r0, droplet_sigma) - _gaussian_cdf(
                rr - 0.5 - r0, droplet_sigma
            )
            frac_c = _gaussian_cdf(cc + 0.5 - c0, droplet_sigma) - _gaussian_cdf(
                cc - 0.5 - c0, droplet_sigma
            )
            data[i, r_lo:r_hi, c_lo:c_hi] += amp * frac_r * frac_c

    index = np.arange(n_events, dtype=np.int64)
    dask_data = da.from_array(data, chunks=(min(chunk_size, n_events), rows, cols))
    return Array(
        data=dask_data,
        index=index,
        step_lengths=[n_events],
        name="detector_stack",
    )
