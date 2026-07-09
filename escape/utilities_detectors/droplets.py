"""Detector-array processing utilities.

Currently: connected-component ("droplet") clustering for pixel detectors.
"""

from typing import NamedTuple

import numpy as np
from scipy import ndimage as _ndimage


# =============================================================================
# Droplet / single-photon-counting utilities
# =============================================================================
#
# Connected-component ("droplet") clustering for pixel detectors: the charge
# cloud from a single photon often splits across a few neighbouring pixels.
# Summing each cluster and re-assigning it to one pixel (or emitting it as a
# sparse event list) gives much cleaner intensity histograms and, for
# low-occupancy frames, a large data reduction.
#
# Built on scipy.ndimage -- no dependency beyond what escape already uses
# elsewhere (see escape.utilities' scipy.stats / scipy.interpolate imports).
# Runs lazily, one dask chunk of frames at a time, via
# Array.map_index_blocks -- see find_droplets() for how to point that at a
# SLURM-backed dask cluster without ever materializing the full stack.


def _droplet_label_chunk(mask, structure=None):
    """Label connected ``True`` regions in *mask* without merging across the
    leading (frame) axis.

    ``mask`` has shape ``(n_frames, *frame_shape)``. Returns ``(labeled,
    n_labels)`` exactly like ``scipy.ndimage.label``.
    """
    if structure is None:
        # Default connectivity structure has no way to know the leading
        # axis is "frames, not space" -- kill the two neighbour-frame
        # planes so droplets never merge across events. What's left in the
        # centre plane is exactly ndimage's own within-frame cross
        # (4-connectivity in 2-D), matching the classic droplet algorithm
        # where diagonal-only touching pixels are separate droplets.
        structure = _ndimage.generate_binary_structure(mask.ndim, 1)
        structure[0].fill(False)
        structure[2].fill(False)
    return _ndimage.label(mask, structure=structure)


def droplets_to_image_block(block, threshold, structure=None):
    """Collapse each connected above-threshold cluster onto its intensity-
    weighted centroid pixel, independently for every frame in *block*.

    Labeling and the droplet sums/centroids are each a single
    ``scipy.ndimage`` call over the whole chunk (no Python loop per frame),
    so this stays fast even for chunks of many 1000x1000 frames.

    Parameters
    ----------
    block : ndarray, shape (n_frames, *frame_shape)
        One dask chunk of raw detector frames.
    threshold : float
        Pixels at or below this value are background.
    structure : ndarray, optional
        Connectivity forwarded to ``scipy.ndimage.label``. Defaults to
        4-connectivity within a frame (see :func:`_droplet_label_chunk`).

    Returns
    -------
    ndarray, same shape/dtype as *block*
        Zero everywhere except each droplet's (rounded) centroid pixel,
        which holds the droplet's total summed intensity.

    Examples
    --------
    >>> import numpy as np
    >>> block = np.zeros((1, 6, 6))
    >>> block[0, 2:4, 2:4] = [[3.0, 1.0], [1.0, 0.0]]  # one 3-pixel droplet
    >>> out = droplets_to_image_block(block, threshold=0.5)
    >>> out.sum()
    5.0
    """
    mask = block > threshold
    labeled, n = _droplet_label_chunk(mask, structure=structure)
    out = np.zeros_like(block)
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    sums = np.asarray(_ndimage.sum(block, labeled, idx))
    centroids = np.asarray(_ndimage.center_of_mass(block, labeled, idx))
    coords = tuple(np.rint(centroids[:, d]).astype(int) for d in range(block.ndim))
    np.add.at(out, coords, sums)
    return out


def droplets_to_sparse_block(block, threshold, max_droplets, structure=None):
    """Same clustering as :func:`droplets_to_image_block`, but returns a
    fixed-capacity sparse event list per frame instead of a full image.

    Currently supports 2-D frames only, i.e. ``block.shape == (n_frames,
    n_rows, n_cols)``.

    Parameters
    ----------
    block : ndarray, shape (n_frames, n_rows, n_cols)
        One dask chunk of raw detector frames.
    threshold : float
        Pixels at or below this value are background.
    max_droplets : int
        Capacity per frame. Frames with more droplets keep only the
        strongest (highest-intensity) ``max_droplets`` of them; unused
        slots are filled with NaN. Choose generously -- silently dropping
        droplets biases low-occupancy statistics.
    structure : ndarray, optional
        Connectivity forwarded to ``scipy.ndimage.label`` (see
        :func:`droplets_to_image_block`).

    Returns
    -------
    ndarray, shape (n_frames, max_droplets, 3)
        Per droplet: ``(row, col, intensity)``.
    """
    if block.ndim != 3:
        raise ValueError(
            "droplets_to_sparse_block only supports 2-D frames "
            f"(n_frames, n_rows, n_cols); got block.shape={block.shape}"
        )
    n_frames = block.shape[0]
    out = np.full((n_frames, max_droplets, 3), np.nan, dtype=np.float64)
    mask = block > threshold
    labeled, n = _droplet_label_chunk(mask, structure=structure)
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    sums = np.asarray(_ndimage.sum(block, labeled, idx))
    centroids = np.asarray(_ndimage.center_of_mass(block, labeled, idx))
    frame_idx = np.rint(centroids[:, 0]).astype(int)

    counts = np.zeros(n_frames, dtype=int)
    for k in np.argsort(-sums):
        f = frame_idx[k]
        c = counts[f]
        if c >= max_droplets:
            continue
        out[f, c, :2] = centroids[k, 1:]
        out[f, c, 2] = sums[k]
        counts[f] += 1
    return out


def droplets_to_photon_count_block(block, threshold, photon_value, max_photons, structure=None):
    """Same clustering as :func:`droplets_to_image_block`, but instead of an
    image or an intensity list, bins each frame's droplets by *estimated
    photon multiplicity* (droplets that received more than one photon's
    worth of charge in the same event -- pile-up -- rather than a truly
    multi-photon-resolving measurement).

    Each droplet's summed intensity is divided by *photon_value* and rounded
    to the nearest integer photon count, then histogrammed per frame.

    Currently supports 2-D frames only, i.e. ``block.shape == (n_frames,
    n_rows, n_cols)``.

    Parameters
    ----------
    block : ndarray, shape (n_frames, n_rows, n_cols)
        One dask chunk of raw detector frames.
    threshold : float
        Pixels at or below this value are background.
    photon_value : float
        Detector intensity (ADU) corresponding to exactly one photon --
        typically the position of the single-photon peak in a histogram of
        droplet-summed intensities (see :func:`droplets_to_image_block`).
    max_photons : int
        Number of histogram bins. Bin ``i`` (0-indexed) counts droplets
        with an estimated ``i + 1`` photons, *except the last bin*, which
        is an overflow bin catching everything with ``>= max_photons``
        estimated photons -- so no droplet is silently dropped. Choose
        ``max_photons`` generously above where you expect real pile-up to
        end; a fat overflow bin will bias a later Poisson fit (see
        :func:`fit_photon_number_distribution`).
    structure : ndarray, optional
        Connectivity forwarded to ``scipy.ndimage.label`` (see
        :func:`droplets_to_image_block`).

    Returns
    -------
    ndarray[int64], shape (n_frames, max_photons)
        Per-frame histogram of estimated photon multiplicities: column
        ``i`` = count of droplets with ``i + 1`` photons (last column =
        overflow, ``>= max_photons``).
    """
    if block.ndim != 3:
        raise ValueError(
            "droplets_to_photon_count_block only supports 2-D frames "
            f"(n_frames, n_rows, n_cols); got block.shape={block.shape}"
        )
    n_frames = block.shape[0]
    out = np.zeros((n_frames, max_photons), dtype=np.int64)
    mask = block > threshold
    labeled, n = _droplet_label_chunk(mask, structure=structure)
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    sums = np.asarray(_ndimage.sum(block, labeled, idx))
    centroids = np.asarray(_ndimage.center_of_mass(block, labeled, idx))
    frame_idx = np.rint(centroids[:, 0]).astype(int)

    n_photons = np.clip(np.rint(sums / photon_value).astype(int), 1, None)
    bin_idx = np.clip(n_photons - 1, 0, max_photons - 1)  # last bin = overflow
    np.add.at(out, (frame_idx, bin_idx), 1)
    return out


class PoissonFitResult(NamedTuple):
    """Result of :func:`fit_photon_number_distribution`."""

    k: float
    """Fitted Poisson rate (mean photons *per opportunity*, zero-photon
    events included -- see the function docstring for why this is *not*
    simply the mean of the observed droplet multiplicities)."""
    expected_counts: np.ndarray
    """Counts this fit predicts per bin, for comparison against the input."""
    chi2: float
    """Pearson chi-square goodness-of-fit statistic."""
    dof: int
    """Degrees of freedom of the chi-square test."""
    p_value: float
    """p-value of the chi-square test (small => a real deviation from Poisson)."""
    reduced_chi2: float
    """``chi2 / dof`` -- the single-number fit-quality summary; ~1 is good."""


def fit_photon_number_distribution(counts, min_photons=1):
    """Fit a **zero-truncated** Poisson distribution to a histogram of
    droplet photon multiplicities, and report how good that fit is.

    This is a genuinely different kind of operation from the "image" /
    "sparse" / "photon_count" *modes* of :func:`find_droplets`: those all
    map one output per input event, lazily, one dask chunk at a time. A
    Poisson fit instead needs the *combined* histogram across many (in the
    fullness of time, all) events -- there's no meaningful "Poisson fit of
    chunk 3 alone". So this is a plain, eager function you call once on the
    already-summed histogram, not another ``mode=`` of ``find_droplets``.

    Why *zero-truncated*: a droplet only exists in the histogram at all if
    a cluster was found above threshold, i.e. if *at least one* photon
    landed there. A location that got zero photons in a given frame never
    produces a droplet, so it is structurally invisible to
    ``mode="photon_count"`` -- the histogram's bin 1 (single photon) is
    never diluted by the zero-photon case the way a plain Poisson
    distribution would have it. Fitting a plain (untruncated) Poisson to
    this data over-estimates the rate and -- worse -- reports a spuriously
    bad fit even for genuinely Poisson-distributed pile-up, since a real
    Poisson pmf and its zero-truncated counterpart only converge once the
    rate is large enough that ``P(0)`` is negligible. This function
    corrects for that, so ``k`` and ``reduced_chi2`` mean what you'd
    expect for the underlying physical photon rate.

    Typical usage::

        hist = find_droplets(
            frames, threshold=15.0, mode="photon_count",
            photon_value=105.0, max_photons=8,
        )
        totals = hist.compute().data.sum(axis=0)  # aggregate over all events
        fit = fit_photon_number_distribution(totals)
        print(f"photon rate: {fit.k:.2f}, fit quality: {fit.reduced_chi2:.2f}")

    Parameters
    ----------
    counts : array-like, 1-D
        Observed counts per photon-multiplicity bin, e.g. the
        ``mode="photon_count"`` histogram summed over events.
        ``counts[i]`` is the number of droplets with an estimated
        ``min_photons + i`` photons. The *last* entry is that mode's
        overflow bin (``>= max_photons`` -- see
        :func:`droplets_to_photon_count_block`); a non-negligible overflow
        bin will bias this fit, so prefer re-running
        ``mode="photon_count"`` with a larger ``max_photons`` over trusting
        the fit if that's the case.
    min_photons : int
        Photon multiplicity of the first bin (``counts[0]``). Matches
        ``mode="photon_count"``'s bins, so leave this at the default ``1``
        (the truncation math below assumes the histogram starts at 1 --
        i.e. that zero-photon droplets are structurally absent, not just
        unobserved).

    Returns
    -------
    PoissonFitResult
        Named tuple ``(k, expected_counts, chi2, dof, p_value,
        reduced_chi2)``.

        ``k`` is the fitted Poisson *rate* -- the mean number of photons
        per opportunity if zero-photon opportunities were counted too
        (they structurally can't be, since no droplet would form). It is
        found by solving ``k / (1 - exp(-k)) == mean(observed
        multiplicities)`` for ``k``, which is the maximum-likelihood
        estimator for a zero-truncated Poisson mean. For a high photon
        rate (``P(0)`` negligible) this converges to the plain sample mean;
        for a low rate it is noticeably *smaller* than the observed mean,
        because the observed population excludes all the (numerous)
        zero-photon opportunities that would otherwise pull the mean down.

        ``reduced_chi2`` is the fit-quality parameter: close to 1 means the
        pile-up statistics are consistent with (zero-truncated) Poisson,
        well above 1 means the droplets are over-dispersed relative to
        that (e.g. detector saturation, multiple overlapping sources, or a
        mis-calibrated ``photon_value``).

        The chi-square statistic and p-value are only reliable if every
        bin's *expected* count is reasonably large (rule of thumb: >= 5) --
        merge or drop sparse tail bins first if that's not the case.
    """
    from scipy.optimize import brentq
    from scipy.stats import chisquare, poisson

    if min_photons != 1:
        raise NotImplementedError(
            "fit_photon_number_distribution currently only supports "
            "min_photons=1 -- the zero-truncation below assumes droplets "
            "are structurally bounded below by 1 photon."
        )

    counts = np.asarray(counts, dtype=float)
    n_bins = len(counts)
    total = counts.sum()
    if total <= 0:
        raise ValueError("counts sum to zero -- nothing to fit.")

    values = np.arange(min_photons, min_photons + n_bins)
    observed_mean = float(np.sum(values * counts) / total)
    if observed_mean < min_photons:
        raise ValueError(
            f"observed mean ({observed_mean}) is below min_photons "
            f"({min_photons}) -- counts/min_photons are inconsistent."
        )

    def _ztp_mean(k):
        # mean of a Poisson(k) distribution conditioned on being >= 1
        # (min_photons == 1); -expm1(-k) is the numerically stable 1 - exp(-k)
        return k / -np.expm1(-k)

    # As k -> 0+, _ztp_mean(k) -> 1 (the min_photons == 1 floor); it's
    # monotonically increasing from there, so the observed mean brackets a
    # unique k except in the degenerate near-all-singles case.
    if observed_mean - min_photons < 1e-9:
        k = 0.0
    else:
        k = brentq(lambda kk: _ztp_mean(kk) - observed_mean, 1e-9, observed_mean * 50)

    pmf = poisson.pmf(values, k) if k > 0 else (values == min_photons).astype(float)
    pmf = pmf / pmf.sum()  # renormalise over this truncated bin range
    expected_counts = pmf * total

    dof = max(n_bins - 2, 1)  # -1 bin normalisation, -1 fitted parameter (k)
    chi2, p_value = chisquare(counts, expected_counts, ddof=1)

    return PoissonFitResult(
        k=k,
        expected_counts=expected_counts,
        chi2=float(chi2),
        dof=dof,
        p_value=float(p_value),
        reduced_chi2=float(chi2) / dof,
    )


def find_droplets(
    array,
    threshold,
    mode="image",
    structure=None,
    max_droplets=500,
    photon_value=None,
    max_photons=10,
    chunk_size=None,
):
    """Photon-counting droplet finder for large 3-D detector stacks.

    Thresholds each frame, connects neighbouring above-threshold pixels into
    "droplets" (the charge cloud of one photon split across pixels), sums
    the intensity within each droplet, and then does one of three things
    with that per-droplet intensity, selected via *mode*:

    - ``"image"`` -- re-assign it to the droplet's intensity-weighted
      centroid pixel.
    - ``"sparse"`` -- emit it as part of a sparse per-frame event list.
    - ``"photon_count"`` -- estimate how many photons piled up into that one
      droplet (``round(intensity / photon_value)``) and histogram that per
      frame. Pair with :func:`fit_photon_number_distribution` (a separate,
      eager function -- see its docstring for why) to fit the aggregated
      histogram to a Poisson distribution.

    Runs lazily via :meth:`Array.map_index_blocks`: only one dask chunk of
    frames is ever materialized per task, so this is memory-safe for stacks
    far larger than a single machine's RAM -- including on a SLURM-backed
    dask cluster (``dask_jobqueue``, e.g.
    :class:`escape.swissfel.cluster.SwissFelCluster`), where each task runs
    on a worker that only ever sees its own chunk.

    Parameters
    ----------
    array : escape.Array
        Dask-backed Array, shape ``(n_events, *frame_shape)`` (2-D frames
        required for ``mode="sparse"`` and ``mode="photon_count"``).
    threshold : float
        Pixels at or below this value are background.
    mode : {"image", "sparse", "photon_count"}
        ``"image"`` -- same-shape output; drop-in replacement for the raw
        frames (e.g. for ``.scan.nanmean()``), giving cleaner intensity
        histograms once droplets are pooled onto single pixels.
        ``"sparse"`` -- ``(max_droplets, 3)`` ``(row, col, intensity)`` per
        event, hugely smaller than the raw frame for low-occupancy data.
        ``"photon_count"`` -- ``(max_photons,)`` per event: a histogram of
        how many droplets in that frame carried an estimated 1, 2, 3, ...
        photons (requires ``photon_value``; see
        :func:`droplets_to_photon_count_block`).
    structure : ndarray, optional
        Connectivity forwarded to ``scipy.ndimage.label`` (default:
        4-connected within a frame).
    max_droplets : int
        Per-frame capacity in ``"sparse"`` mode (ignored otherwise).
    photon_value : float, optional
        Detector intensity (ADU) of a single photon. Required for
        ``mode="photon_count"`` (ignored otherwise) -- see
        :func:`droplets_to_photon_count_block`.
    max_photons : int
        Number of photon-multiplicity histogram bins in ``"photon_count"``
        mode (ignored otherwise); the last bin is an overflow bin. See
        :func:`droplets_to_photon_count_block`.
    chunk_size : int, optional
        Rechunk the event axis to this many frames per chunk before
        mapping -- the main memory/parallelism knob. Defaults to the
        array's existing chunking.

    Returns
    -------
    escape.Array
        Lazy dask-backed Array; call ``.compute()`` to evaluate locally or
        ``.persist()`` to evaluate on a running (e.g. SLURM) cluster
        without gathering the result back to the local process.

    Examples
    --------
    Local, chunk-at-a-time (safe even if the full stack doesn't fit in
    RAM)::

        >>> from escape.utilities_detectors import find_droplets
        >>> imgs_clean = find_droplets(detector_stack, threshold=15.0)
        >>> imgs_clean.compute()

    Distributed over a SLURM allocation via dask_jobqueue -- each task
    pulls only one chunk of frames onto its worker, so a 1000x1000 stack of
    10^5+ frames never needs to fit on a single node::

        >>> from escape.swissfel.cluster import SwissFelCluster
        >>> cluster = SwissFelCluster(
        ...     local=False, cores=4, memory="8 GB", workers=20, processes=1,
        ... )
        >>> events = find_droplets(
        ...     detector_stack, threshold=15.0, mode="sparse",
        ...     max_droplets=200, chunk_size=50,
        ... )
        >>> events = events.persist()   # scatters computation over workers
        >>> events.compute()            # gather the (small) sparse result

    Photon pile-up statistics, fitted to a Poisson distribution::

        >>> from escape.utilities_detectors import fit_photon_number_distribution
        >>> hist = find_droplets(
        ...     detector_stack, threshold=15.0, mode="photon_count",
        ...     photon_value=105.0, max_photons=8,
        ... )
        >>> totals = hist.compute().data.sum(axis=0)  # aggregate over events
        >>> fit = fit_photon_number_distribution(totals)
        >>> fit.k, fit.reduced_chi2
    """
    if chunk_size is not None:
        from ..storage.storage import Array

        data = array.data.rechunk((chunk_size,) + array.data.shape[1:])
        array = Array(
            data=data,
            index=array.index,
            step_lengths=array.scan.step_lengths,
            parameter=array.scan.parameter,
        )

    if mode == "image":
        return array.map_index_blocks(
            droplets_to_image_block, threshold, structure=structure
        )
    elif mode == "sparse":
        return array.map_index_blocks(
            droplets_to_sparse_block,
            threshold,
            max_droplets,
            structure=structure,
            new_element_size=(max_droplets, 3),
            dtype=np.float64,
        )
    elif mode == "photon_count":
        if photon_value is None:
            raise ValueError(
                "mode='photon_count' requires photon_value (the ADU value "
                "of a single photon)."
            )
        return array.map_index_blocks(
            droplets_to_photon_count_block,
            threshold,
            photon_value,
            max_photons,
            structure=structure,
            drop_axis=[2],
            new_element_size=(max_photons,),
            dtype=np.int64,
        )
    else:
        raise ValueError(
            f"mode must be 'image', 'sparse' or 'photon_count', got {mode!r}"
        )
