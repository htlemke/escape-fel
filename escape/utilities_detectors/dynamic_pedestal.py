"""
Determine per-pixel dynamic-gain pedestal offsets for gain-switching
detectors (e.g. PSI Jungfrau).

Background
----------
Jungfrau-type detectors apply, per pixel and per event, a gain-dependent
correction::

    I_corr = (I_raw - P[gain]) / G[gain]

where ``gain`` in {0, 1, 2} is decided dynamically, pixel by pixel and
event by event, once the raw signal exceeds a threshold. P (pedestal) is
measured in a *forced*, static-gain acquisition and can differ from the
pedestal a pixel actually settles to right after a *dynamic* switch
(settling / memory effects in the analog chain). This produces a
systematic (not photon-statistics) discontinuity in I_corr exactly at the
gain-switching threshold(s), reproducible from shot to shot.

This module recovers a per-pixel raw-ADC pedestal offset dP[gain] by
enforcing continuity of I_corr against a secondary, switch-independent
intensity proxy (an external monitor, or an internal ROI intensity) across
each gain transition: fit a local line to the lower-gain branch and to the
higher-gain branch on either side of the transition, and solve for the
offset that makes them meet.

Pixels are processed in chunks (``pixel_chunk_size``) to bound peak memory:
the fast per-pixel binning below needs O(n_events * n_pixels) temporaries,
which is fine for an ROI/tile but would blow up (tens of GB) if applied to
a full multi-megapixel detector in one call.

See ``utilities_detectors/2023-10 gainswitch issue 1st gain.pdf`` for the
analysis this implements.
"""

from __future__ import annotations

import numpy as np


def decode_jungfrau_raw(image):
    """Split raw Jungfrau 16-bit words (as obtained with the DAQ's
    ``keep_raw_data: True`` / ``JFDataHandler`` setting, before any
    gain/pedestal correction) into ADC value and gain index.

    Matches the bit layout used in ``escape.swissfel.detector.apply_gain_pede_np``:
    bits 0-13 are the 14-bit ADC value, bits 14-15 encode the gain
    (0 -> gain 0, 1 -> gain 1, 2 or 3 -> gain 2).

    Returns
    -------
    adc : ndarray, same shape as ``image``, uint/int
    gain_idx : ndarray, same shape as ``image``, int (values 0, 1, 2)
    """
    mask14 = int("0b" + 14 * "1", 2)
    mask2 = int("0b" + 2 * "1", 2)
    adc = np.bitwise_and(image, mask14)
    raw_gain = np.bitwise_and(np.right_shift(image, 14), mask2)
    gain_idx = np.where(raw_gain >= 2, 2, raw_gain)
    return adc, gain_idx


def apply_calibration(raw, gain_idx, pedestal, gain, offset=None):
    """Vectorized ``(raw - pedestal[gain] - offset[gain]) / gain[gain]``.

    Parameters
    ----------
    raw, gain_idx : ndarray, shape (n_events, *pixel_shape)
        Raw ADC values and per-event gain index (0/1/2, ...).
    pedestal, gain : ndarray, shape (n_gains, *pixel_shape)
    offset : ndarray, shape (n_gains, *pixel_shape), optional
        Additive raw-ADC pedestal correction per gain, e.g. as returned by
        :func:`find_dynamic_pedestal_offsets`.
    """
    ped = np.choose(gain_idx, pedestal)
    g = np.choose(gain_idx, gain)
    if offset is not None:
        ped = ped + np.choose(gain_idx, np.nan_to_num(offset))
    return (raw.astype(np.float64) - ped) / g


def _sort_bins(monitor, bin_edges):
    """Pixel-independent part of the binning: sort events by monitor bin
    once, and locate the bin boundaries in the sorted order. Reused across
    gains and pixel chunks so it is only paid for once per call.

    Returns
    -------
    order : ndarray, shape (n_events,)
        Permutation that sorts events by monitor bin.
    boundaries : ndarray, shape (n_bins + 1,)
        boundaries[b]:boundaries[b+1] indexes (in ``order``) the events of
        bin b; correctly zero-length for empty bins.
    """
    n_bins = len(bin_edges) - 1
    raw_bin_idx = np.digitize(monitor, bin_edges) - 1
    in_range = (raw_bin_idx >= 0) & (raw_bin_idx < n_bins)
    bin_idx = np.where(in_range, raw_bin_idx, n_bins)  # n_bins = "out of range" bucket

    order = np.argsort(bin_idx, kind="stable")
    bin_idx_sorted = bin_idx[order]
    boundaries = np.searchsorted(bin_idx_sorted, np.arange(n_bins + 1))
    return order, boundaries


def _binned_sums(values, mask, order, boundaries):
    """Per-pixel sum/count of ``values`` in monitor bins, restricted to
    events where ``mask`` is True, given precomputed sort order and bin
    boundaries from :func:`_sort_bins`.

    values, mask : (n_events, *pixel_shape)   (mask boolean)

    A single cumsum + fancy-index gather over the *sorted* events gives all
    bin sums at once: O(n_events * n_pixels), no per-bin or per-event
    Python loop (unlike ``np.add.at``, which is correct but very slow, or a
    Python loop over bins, which multiplies the work by n_bins).

    Returns
    -------
    sums, counts : ndarray, shape (n_bins, *pixel_shape)
    """
    pixel_shape = values.shape[1:]

    keep_sorted = mask[order]
    vals_sorted = np.where(keep_sorted, values[order], 0.0)
    wgt_sorted = keep_sorted.astype(np.float64)

    zero_row = np.zeros((1,) + pixel_shape)
    cum_vals = np.concatenate([zero_row, np.cumsum(vals_sorted, axis=0)], axis=0)
    cum_wgt = np.concatenate([zero_row, np.cumsum(wgt_sorted, axis=0)], axis=0)

    sums = cum_vals[boundaries[1:]] - cum_vals[boundaries[:-1]]
    counts = cum_wgt[boundaries[1:]] - cum_wgt[boundaries[:-1]]
    return sums, counts


def _edge_bin(counts, n_bins, side):
    """Per-pixel index of the last (side='last') / first (side='first')
    populated bin along axis 0. Returns -1 / n_bins where none populated."""
    has = counts > 0
    idx_grid = np.arange(n_bins).reshape((n_bins,) + (1,) * (has.ndim - 1))
    if side == "last":
        return np.where(has, idx_grid, -1).max(axis=0)
    else:
        return np.where(has, idx_grid, n_bins).min(axis=0)


def _gather_window(arr, center_idx, k, forward):
    """Gather a window of k bins from ``arr`` (shape (n_bins, *pixel_shape))
    starting at / ending at ``center_idx`` (shape (*pixel_shape,)).

    forward=True  -> indices center_idx, center_idx+1, ..., center_idx+k-1
    forward=False -> indices center_idx, center_idx-1, ..., center_idx-k+1

    Returns gathered values (k, *pixel_shape) and a validity mask of the
    same shape (False where the window index fell outside [0, n_bins)).
    """
    n_bins = arr.shape[0]
    step = 1 if forward else -1
    offsets = (np.arange(k) * step).reshape((k,) + (1,) * center_idx.ndim)
    idx = center_idx[None, ...] + offsets
    valid = (idx >= 0) & (idx < n_bins)
    idx_clipped = np.clip(idx, 0, n_bins - 1)
    gathered = np.take_along_axis(arr, idx_clipped, axis=0)
    return gathered, valid


def _weighted_line_value_at(x, sums, counts, x_star):
    """Per-pixel weighted-least-squares line through the points
    (x_i, sums_i/counts_i) with weight counts_i, evaluated at x_star.

    x : (k, *pixel_shape)  bin centers of the window (already masked to 0
        weight where invalid, so their x value is irrelevant)
    sums, counts : (k, *pixel_shape)
    x_star : (*pixel_shape,)

    Returns predicted value at x_star, shape (*pixel_shape,); nan where
    underdetermined (< 2 populated bins in the window and not exactly on
    a single populated bin either).
    """
    w = counts
    y = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    Sw = w.sum(axis=0)
    Swx = (w * x).sum(axis=0)
    Swy = (w * y).sum(axis=0)
    Swxx = (w * x * x).sum(axis=0)
    Swxy = (w * x * y).sum(axis=0)

    denom = Sw * Swxx - Swx**2
    n_populated = (w > 0).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (Sw * Swxy - Swx * Swy) / denom
        intercept = (Swy - slope * Swx) / Sw
        mean_only = Swy / Sw  # fallback: flat (slope=0) weighted mean

        use_line = (n_populated >= 2) & (np.abs(denom) > 0)
        predicted = np.where(use_line, intercept + slope * x_star, mean_only)
        predicted = np.where(Sw > 0, predicted, np.nan)
    return predicted


def _gain_switch_offset_chunk(
    raw, gain_idx, pedestal, gain, g_low, g_high,
    order, boundaries, bin_centers, n_fit_bins, min_counts, margin_bins,
):
    """Core per-chunk computation shared by :func:`find_gain_switch_offset`.
    All arrays here already refer to a single, memory-bounded chunk of
    pixels (flattened to one axis)."""
    n_bins = len(bin_centers)

    I0 = apply_calibration(raw, gain_idx, pedestal, gain)  # no offset yet
    mask_low = gain_idx == g_low
    mask_high = gain_idx == g_high

    sums_low, counts_low = _binned_sums(I0, mask_low, order, boundaries)
    sums_high, counts_high = _binned_sums(I0, mask_high, order, boundaries)

    last_low = _edge_bin(counts_low, n_bins, "last")
    first_high = _edge_bin(counts_high, n_bins, "first")

    have_switch = (last_low >= 0) & (first_high < n_bins)
    last_low_c = np.where(have_switch, last_low, 0)
    first_high_c = np.where(have_switch, first_high, 0)

    # continuity point stays at the true (unshifted) switch location; the
    # margin only controls which populated bins feed the local fits below.
    x_star = 0.5 * (bin_centers[last_low_c] + bin_centers[first_high_c])

    fit_start_low = np.clip(last_low_c - margin_bins, 0, n_bins - 1)
    fit_start_high = np.clip(first_high_c + margin_bins, 0, n_bins - 1)

    s_win_low, v_low = _gather_window(sums_low, fit_start_low, n_fit_bins, forward=False)
    c_win_low, _ = _gather_window(counts_low, fit_start_low, n_fit_bins, forward=False)
    c_win_low = np.where(v_low, c_win_low, 0.0)
    x_win_low = np.broadcast_to(
        bin_centers[np.clip(fit_start_low[None, ...] - np.arange(n_fit_bins).reshape(
            (n_fit_bins,) + (1,) * fit_start_low.ndim), 0, n_bins - 1)],
        s_win_low.shape,
    )

    s_win_high, v_high = _gather_window(sums_high, fit_start_high, n_fit_bins, forward=True)
    c_win_high, _ = _gather_window(counts_high, fit_start_high, n_fit_bins, forward=True)
    c_win_high = np.where(v_high, c_win_high, 0.0)
    x_win_high = np.broadcast_to(
        bin_centers[np.clip(fit_start_high[None, ...] + np.arange(n_fit_bins).reshape(
            (n_fit_bins,) + (1,) * fit_start_high.ndim), 0, n_bins - 1)],
        s_win_high.shape,
    )

    pred_low = _weighted_line_value_at(x_win_low, s_win_low, c_win_low, x_star)
    pred_high = _weighted_line_value_at(x_win_high, s_win_high, c_win_high, x_star)

    offset = gain[g_high] * (pred_high - pred_low)

    n_low = counts_low.sum(axis=0)
    n_high = counts_high.sum(axis=0)
    enough_stats = have_switch & (n_low >= min_counts) & (n_high >= min_counts)
    offset = np.where(enough_stats, offset, np.nan)

    return offset, n_low, n_high


def find_gain_switch_offset(
    monitor,
    raw,
    gain_idx,
    pedestal,
    gain,
    g_low,
    g_high,
    bin_edges,
    n_fit_bins=4,
    min_counts=10,
    margin_bins=0,
    pixel_chunk_size=4096,
):
    """Recover the raw-ADC pedestal offset for gain index ``g_high`` that
    makes ``I_corr`` continuous with the ``g_low`` branch across their
    mutual transition, using ``monitor`` as the common independent
    variable (external intensity monitor, or an internal ROI sum -- any
    signal that varies smoothly across the switch and is itself unaffected
    by it).

    Parameters
    ----------
    monitor : ndarray, shape (n_events,)
    raw, gain_idx : ndarray, shape (n_events, *pixel_shape)
    pedestal, gain : ndarray, shape (n_gains, *pixel_shape)
        The existing (static/forced-gain) calibration.
    g_low, g_high : int
        Adjacent gain indices of the transition to calibrate, e.g. (0, 1)
        or (1, 2). g_high is assumed to be reached from g_low as the
        signal increases.
    bin_edges : ndarray, shape (n_bins + 1,)
        Monitor bin edges used to average out per-shot Poisson noise.
    n_fit_bins : int
        Number of populated bins on each side of the transition used for
        the local linear extrapolation (>=1; 1 reduces to nearest-bin
        matching).
    min_counts : int
        Minimum total event count on *each* side for a pixel to get a
        finite offset; below that the pixel is returned as nan (use
        :func:`aggregate_offsets` to fall back to a tile/asic-level value).
    margin_bins : int
        Number of populated bins immediately adjacent to the switch to
        *skip* before starting each fit window. The gain actually taken by
        a pixel is decided on its own noisy raw value, so bins right at
        the threshold are biased by this selection effect (only upward
        noise fluctuations cross into g_high there, and vice versa for
        g_low) -- not a pedestal effect. A small margin (e.g. 1-3) trades
        that bias against a larger linear-extrapolation gap; 0 disables it.
    pixel_chunk_size : int
        Number of pixels processed per chunk. The binning step needs
        O(n_events * n_pixels_in_chunk) temporary memory, so this bounds
        peak memory regardless of how large ``pixel_shape`` is (e.g. a
        full detector tile). Increase for speed if memory allows, decrease
        if you hit memory pressure on large ROIs / long runs.

    Returns
    -------
    offset : ndarray, shape (*pixel_shape,)
        Additive raw-ADC correction to ``pedestal[g_high]``. nan where
        the pixel never has enough statistics on both sides of the switch
        (i.e. it is not one of the "gain-switching pixels").
    n_low, n_high : ndarray, shape (*pixel_shape,)
        Total event counts feeding the low-side / high-side local fit, for
        diagnostics and masking.
    """
    pixel_shape = raw.shape[1:]
    npix = int(np.prod(pixel_shape)) if pixel_shape else 1
    n_events = raw.shape[0]

    order, boundaries = _sort_bins(monitor, bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    raw_flat = raw.reshape(n_events, npix)
    gain_idx_flat = gain_idx.reshape(n_events, npix)
    pedestal_flat = pedestal.reshape(pedestal.shape[0], npix)
    gain_flat = gain.reshape(gain.shape[0], npix)

    offset_flat = np.empty(npix)
    n_low_flat = np.empty(npix)
    n_high_flat = np.empty(npix)

    for start in range(0, npix, pixel_chunk_size):
        stop = min(start + pixel_chunk_size, npix)
        off, nl, nh = _gain_switch_offset_chunk(
            raw_flat[:, start:stop], gain_idx_flat[:, start:stop],
            pedestal_flat[:, start:stop], gain_flat[:, start:stop],
            g_low, g_high, order, boundaries, bin_centers,
            n_fit_bins, min_counts, margin_bins,
        )
        offset_flat[start:stop] = off
        n_low_flat[start:stop] = nl
        n_high_flat[start:stop] = nh

    return (
        offset_flat.reshape(pixel_shape),
        n_low_flat.reshape(pixel_shape),
        n_high_flat.reshape(pixel_shape),
    )


def find_dynamic_pedestal_offsets(
    monitor,
    raw,
    gain_idx,
    pedestal,
    gain,
    gain_pairs=((0, 1), (1, 2)),
    bin_edges=None,
    n_bins=80,
    n_fit_bins=4,
    min_counts=10,
    margin_bins=0,
    pixel_chunk_size=4096,
):
    """Convenience wrapper around :func:`find_gain_switch_offset` looping
    over successive gain transitions.

    Returns
    -------
    offsets : ndarray, shape (n_gains, *pixel_shape)
        Additive raw-ADC pedestal offsets, indexed like ``pedestal``.
        offsets[g_low] for the lowest gain of ``gain_pairs`` is left at 0
        (it is the reference branch); nan elsewhere statistics were
        insufficient.
    counts : dict
        {(g_low, g_high): (n_low, n_high)} diagnostic counts.

    Notes
    -----
    Transitions are resolved in the order given by ``gain_pairs`` and
    chained: e.g. for ``((0, 1), (1, 2))`` the gain-1 pedestal used as the
    "low" reference branch of the (1, 2) transition already includes the
    offset found for (0, 1), so errors don't leak from one transition into
    the next.
    """
    if bin_edges is None:
        bin_edges = np.linspace(np.nanmin(monitor), np.nanmax(monitor), n_bins + 1)

    n_gains = pedestal.shape[0]
    offsets = np.full((n_gains,) + pedestal.shape[1:], np.nan)
    offsets[gain_pairs[0][0]] = 0.0
    counts = {}
    pedestal_running = pedestal.copy()

    for g_low, g_high in gain_pairs:
        off, n_low, n_high = find_gain_switch_offset(
            monitor, raw, gain_idx, pedestal_running, gain, g_low, g_high,
            bin_edges, n_fit_bins=n_fit_bins, min_counts=min_counts,
            margin_bins=margin_bins, pixel_chunk_size=pixel_chunk_size,
        )
        offsets[g_high] = off
        counts[(g_low, g_high)] = (n_low, n_high)
        pedestal_running[g_high] = pedestal[g_high] + np.nan_to_num(off)

    return offsets, counts


def aggregate_offsets(offsets, block_shape):
    """Fill nan (insufficient-statistics) pixels of a per-gain offset map
    with the nanmedian offset of the surrounding block (e.g. one ASIC or
    tile), per the document's observation that the offset is close to
    constant per tile/asic. ``block_shape`` divides the last two axes of
    ``offsets`` evenly (e.g. (256, 256) for a Jungfrau ASIC).

    Parameters
    ----------
    offsets : ndarray, shape (..., ny, nx)
    block_shape : (by, bx)

    Returns
    -------
    filled : ndarray, same shape as offsets
    """
    by, bx = block_shape
    *lead, ny, nx = offsets.shape
    assert ny % by == 0 and nx % bx == 0, "block_shape must evenly divide ny, nx"
    reshaped = offsets.reshape(*lead, ny // by, by, nx // bx, bx)
    block_median = np.nanmedian(reshaped, axis=(-3, -1), keepdims=True)
    filled = np.where(np.isnan(reshaped), block_median, reshaped)
    return filled.reshape(offsets.shape)
