"""Rolling-reference binning prototype.

Alternative to a full-array reorder (cf. ``Array.digitize`` /
``array.data[ix]`` in storage.py, which fancy-indexes the *entire* lazy
array and can force a worker to materialize an entire bin's worth of
scattered chunks at once).

Here each dask chunk is processed independently and *locally* accumulates
its events into per-bin weighted sums, conditioned on a rolling window of
the last ``n_refs`` "reference" events seen so far. The per-chunk partial
sums are then combined into a single ``[n_bins_total, *event_shape]``
result via a delayed merge.

STATUS: prototype, not verified correct. See "Open issues" at the bottom
before relying on this for real data.
"""

import os
import tempfile
import threading
import time
import warnings
from collections import deque

import numpy as np
import dask.array as da
from dask import delayed


CMP_TYPES = ("none", "difference", "ratio")


def bin_running_ref(
    data, idop, iref, bins, weights, n_refs, n_ref_min,
    cmp_type="none", ref_weighted=True,
    compute_std=False, compute_quantile=False,
):
    """Accumulate one chunk's events into weighted per-bin sums.

    For every event flagged in ``idop``, once at least ``n_ref_min``
    reference events (flagged in ``iref``) have been seen *within this
    chunk*, the event is compared to the current rolling reference window
    (per ``cmp_type``) and added (weighted) to its target bin.

    Parameters
    ----------
    data : ndarray, shape (n_events, *event_shape)
        One chunk of the big stack, event axis first.
    idop : ndarray[bool], shape (n_events,)
        True for events that should be binned.
    iref : ndarray[bool], shape (n_events,)
        True for events usable as rolling reference.
    bins : ndarray[int], shape (n_events,)
        Target bin id for each event (only meaningful where ``idop`` is True).
    weights : ndarray or None, shape (n_events,)
        Per-event weight; defaults to all-ones.
    n_refs : int
        Rolling window length (max number of reference events retained).
    n_ref_min : int
        Minimum number of references required before an event is binned.
    cmp_type : {"none", "difference", "ratio"}, optional
        How a signal event is compared to its rolling reference window
        (``ref_mean``, the mean of the up-to-``n_refs`` most recent
        reference events seen so far in this chunk) before being binned.
        Matches the ``cmp_type`` convention of
        :meth:`escape.storage.storage_tools.ArrayTools.compare_to_reference`.

        - ``"none"`` (default): bin the raw signal value, unchanged.
          ``n_ref_min`` still gates *whether* an event is binned, but the
          collected reference frames are otherwise unused.
        - ``"difference"``: bin ``data[event] - ref_mean``.
        - ``"ratio"``: bin ``data[event] / ref_mean``.

    ref_weighted : bool, optional
        If True (default), ``ref_mean`` is the weighted mean of the
        rolling reference window (each reference frame weighted by its own
        ``weights`` entry). If False, it's a plain unweighted mean. Ignored
        when ``cmp_type == "none"``.
    compute_std : bool, optional
        If True, also accumulate ``sumsq_out`` (weighted sum of squared
        binned values), enough to recover the per-bin standard deviation
        at finalize time. Implemented as a plain additive accumulator
        (same merge behavior as ``data_out``/``n_in_bins``, no extra
        memory cost per chunk) -- note this is a "sum of squares" estimator
        and can lose precision if the binned values are large relative to
        their spread; center your comparison (e.g. ``cmp_type="difference"``)
        if that's a concern.
    compute_quantile : bool, optional
        If True, also collect the *raw* (bin_id -> (values, weights)) data
        needed for an exact (weighted) quantile/median/MAD later. Unlike
        every other accumulator here, this cannot be reduced online -- it
        concatenates rather than sums, so enabling it reintroduces
        per-bin memory proportional to that bin's event count. Only turn
        this on when you know a bin's worth of data fits in memory.

    Returns
    -------
    data_out : ndarray, shape (len(bin_ids), *event_shape)
        Weighted sums (of the, possibly reference-corrected, signal value)
        for only the bins that occur in this chunk.
    n_in_bins : ndarray, shape (len(bin_ids),)
        Summed weights per bin (denominator for a later weighted mean).
    bin_ids : ndarray[int], sorted
        Which global bin id each row of ``data_out``/``n_in_bins`` belongs to.
    sumsq_out : ndarray or None, shape (len(bin_ids), *event_shape)
        Weighted sum of squared values per bin, if ``compute_std``, else None.
    raw_out : dict or None
        ``{bin_id: (values, weights)}`` for every bin touched in this
        chunk, if ``compute_quantile``, else None.

    Notes
    -----
    ``rolling_refs`` is local to this call, i.e. it resets at every chunk
    boundary. See "Open issues" in the module docstring for why this
    matters for real (small) chunk sizes.
    """
    if cmp_type not in CMP_TYPES:
        raise ValueError(f"cmp_type must be one of {CMP_TYPES}, got {cmp_type!r}")
    if cmp_type != "none" and n_ref_min < 1:
        raise ValueError(
            "n_ref_min must be >= 1 when cmp_type != 'none' -- there must be "
            "at least one reference frame to compare a signal event against."
        )
    if len(idop) != len(bins):
        raise ValueError("idop and bins must have the same length")
    if weights is None:
        weights = np.ones(data.shape[0])

    bin_ids = np.sort(np.unique(bins[idop]))
    data_out = np.zeros((len(bin_ids), *data.shape[1:]))
    n_in_bins = np.zeros(len(bin_ids), dtype=float)
    sumsq_out = np.zeros((len(bin_ids), *data.shape[1:])) if compute_std else None
    raw_out = {} if compute_quantile else None

    rolling_refs = deque(maxlen=n_refs)
    rolling_ref_weights = deque(maxlen=n_refs)
    # Running sums kept in step with `rolling_refs` so `ref_mean` is O(1)
    # to update per reference event, instead of re-averaging the whole
    # window (up to `n_refs` full images) on every single signal event.
    ref_sum = np.zeros(data.shape[1:])
    ref_weighted_sum = np.zeros(data.shape[1:])
    ref_weight_total = 0.0

    pending = []
    for n, (is_dop, is_ref) in enumerate(zip(idop, iref)):
        if is_ref:
            if len(rolling_refs) == n_refs:
                oldest, oldest_w = rolling_refs[0], rolling_ref_weights[0]
                ref_sum -= oldest
                ref_weighted_sum -= oldest_w * oldest
                ref_weight_total -= oldest_w
            rolling_refs.append(data[n])
            rolling_ref_weights.append(weights[n])
            ref_sum += data[n]
            ref_weighted_sum += weights[n] * data[n]
            ref_weight_total += weights[n]
        if is_dop:
            pending.append(n)
            if len(rolling_refs) >= n_ref_min:
                if cmp_type != "none":
                    ref_mean = (
                        ref_weighted_sum / ref_weight_total
                        if ref_weighted
                        else ref_sum / len(rolling_refs)
                    )
                for tn in pending:
                    bin_idx = np.searchsorted(bin_ids, bins[tn])
                    value = data[tn]
                    if cmp_type == "difference":
                        value = value - ref_mean
                    elif cmp_type == "ratio":
                        value = value / ref_mean
                    data_out[bin_idx] += value * weights[tn]
                    n_in_bins[bin_idx] += weights[tn]
                    if compute_std:
                        sumsq_out[bin_idx] += weights[tn] * value**2
                    if compute_quantile:
                        bin_id = bin_ids[bin_idx]
                        vs, ws = raw_out.setdefault(bin_id, ([], []))
                        vs.append(value)
                        ws.append(weights[tn])
                pending = []
        # NOTE: any events left in `pending` when the chunk ends are
        # silently dropped -- see "Open issues" below.

    if compute_quantile:
        raw_out = {
            bin_id: (np.stack(vs), np.asarray(ws))
            for bin_id, (vs, ws) in raw_out.items()
        }

    return data_out, n_in_bins, bin_ids, sumsq_out, raw_out


def expected_bins_touched(chunk_size, n_bins_total, frac_signal):
    """Expected number of distinct bins a chunk of `chunk_size` events touches.

    Treats bin assignment as uniform random over `n_bins_total` bins (the
    demo's actual distribution); real bin assignments (e.g. from a scan
    parameter) may be more or less clustered per chunk depending on
    acquisition order, so treat this as an estimate, not a guarantee.
    """
    n_dop_expected = chunk_size * frac_signal
    return n_bins_total * (1 - (1 - 1 / n_bins_total) ** n_dop_expected)


def suggest_chunk_size(
    event_nbytes,
    n_bins_total,
    frac_signal,
    memory_budget_bytes,
    n_ref_min=1,
    frac_ref=None,
    safety_factor=0.5,
    min_chunk_size=1,
    max_chunk_size=200_000,
):
    """Pick the largest per-task chunk size that fits a memory budget.

    Models one task's peak memory as::

        chunk_size * event_nbytes                                  # raw input chunk
        + expected_bins_touched(chunk_size, ...) * event_nbytes     # compact output

    The first term grows linearly with chunk size; the second saturates at
    ``n_bins_total * event_nbytes`` (once a chunk is big enough to plausibly
    touch every bin). The sum is monotonic in chunk_size, so the largest
    chunk size under budget is found by bisection.

    This is the "speed vs. memory" knob you're asking about: bigger chunks
    mean fewer, larger tasks (less scheduling/merge overhead, and less
    relative boundary loss -- see the loss-rate table in this module's
    history/discussion), at the cost of more memory held per task.

    Also warns if the chosen chunk size is small enough that the rolling
    reference window (`n_ref_min` out of events occurring at rate
    `frac_ref`) will frequently fail to fill within one chunk -- empirically,
    per-chunk event loss becomes negligible once
    ``chunk_size >~ 10 * n_ref_min / frac_ref``, and is severe (whole
    chunks can be entirely dropped) below roughly half of that.
    """
    if frac_ref is None:
        frac_ref = 1 - frac_signal
    budget = memory_budget_bytes * safety_factor

    def peak_bytes(n):
        return n * event_nbytes + expected_bins_touched(n, n_bins_total, frac_signal) * event_nbytes

    if peak_bytes(min_chunk_size) > budget:
        raise ValueError(
            f"Even chunk_size={min_chunk_size} needs an estimated "
            f"{peak_bytes(min_chunk_size) / 1e9:.2f} GB per task, above the "
            f"{budget / 1e9:.2f} GB budget (memory_budget_bytes * safety_factor). "
            "Lowering memory_budget_bytes further won't help here -- you need "
            "fewer bins, a smaller event shape, or a lower-precision dtype."
        )

    lo, hi = min_chunk_size, max_chunk_size
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if peak_bytes(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1

    if frac_ref > 0:
        recommended_min = int(np.ceil(10 * n_ref_min / frac_ref))
        if lo < recommended_min:
            warnings.warn(
                f"suggest_chunk_size picked chunk_size={lo} to fit the memory "
                f"budget, but that is below the ~{recommended_min} "
                "(10 x n_ref_min / frac_ref) events typically needed for the "
                "rolling reference window to reliably fill within a chunk -- "
                "expect non-negligible (possibly whole-chunk) event loss at "
                "chunk boundaries. Consider raising memory_budget_bytes, "
                "lowering n_ref_min, or accepting the loss.",
                stacklevel=2,
            )
    return lo


def estimate_result_bytes(n_bins_total, event_shape, dtype=np.float64):
    """Size in bytes of the dense [n_bins_total, *event_shape] result array."""
    return int(n_bins_total * np.prod(event_shape, dtype=np.int64) * np.dtype(dtype).itemsize)


class ResultStore:
    """Accumulator for the final [n_bins_total, *event_shape] result.

    Backed by an in-memory numpy array when the dense result fits
    comfortably in the memory budget, or by an on-disk HDF5 dataset
    (read-modify-write, lock-guarded) when fine binning (many bins x large
    event shape) would otherwise force an allocation too big for one
    worker's RAM -- e.g. 10,000 bins x 4000x4000 float32 is 640 GB dense,
    which no single machine here holds, but HDF5 handles natively since
    each accumulate touches only the rows it needs.

    Threading note
    --------------
    The default lock is a plain `threading.Lock`, which only protects
    against races *within one process* -- fine for dask's threaded
    scheduler. For a multiprocessing or distributed scheduler, pass
    `lock=distributed.Lock(...)` (or `dask.utils.SerializableLock()`)
    instead, the same way `escape.storage.storage.store()` does via
    `get_lock()`; otherwise concurrent worker processes can corrupt the
    on-disk dataset with unserialized read-modify-write writes.

    ``finalize()`` never closes the backing HDF5 file (own tempfile or a
    caller-supplied ``parent_h5py`` group) -- it returns live dataset
    handles so the caller can wrap them in a lazy ``da.from_array`` right
    away. Keep the ``ResultStore`` instance (or the returned handles)
    referenced for as long as that lazy array is used.
    """

    def __init__(
        self,
        n_bins_total,
        event_shape,
        dtype=np.float64,
        backend="auto",
        memory_budget_bytes=4 * 1024**3,
        path=None,
        parent_h5py=None,
        lock=None,
        track_std=False,
    ):
        self.n_bins_total = n_bins_total
        self.event_shape = tuple(event_shape)
        self.dtype = np.dtype(dtype)
        self.track_std = track_std
        nbytes = estimate_result_bytes(n_bins_total, self.event_shape, self.dtype)
        nbytes_total = nbytes * (2 if track_std else 1)

        if backend == "auto":
            backend = "disk" if (parent_h5py is not None or nbytes_total > memory_budget_bytes) else "memory"
        self.backend = backend
        self.lock = lock or threading.Lock()

        if backend == "memory":
            self.path = None
            self._h5 = None
            self._data = np.zeros((n_bins_total, *self.event_shape), dtype=self.dtype)
            self._n = np.zeros(n_bins_total)
            self._sumsq = np.zeros((n_bins_total, *self.event_shape), dtype=self.dtype) if track_std else None
        elif backend == "disk":
            import h5py

            if parent_h5py is not None:
                self.path = None
                self._h5 = None  # caller owns this group's file lifecycle
                grp = parent_h5py
            else:
                if path is None:
                    fd, path = tempfile.mkstemp(suffix=".h5", prefix="escape_rollin_result_")
                    os.close(fd)
                self.path = path
                self._h5 = h5py.File(path, "a")
                grp = self._h5
            self._data = grp.create_dataset(
                "data_out",
                shape=(n_bins_total, *self.event_shape),
                dtype=self.dtype,
                chunks=(1, *self.event_shape),
            )
            self._n = grp.create_dataset("n_in_bins", shape=(n_bins_total,), dtype=float)
            self._sumsq = (
                grp.create_dataset(
                    "sumsq_out",
                    shape=(n_bins_total, *self.event_shape),
                    dtype=self.dtype,
                    chunks=(1, *self.event_shape),
                )
                if track_std
                else None
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}, expected 'auto'/'memory'/'disk'")

        print(
            f"ResultStore: backend={self.backend} (dense result would be "
            f"{nbytes_total / 1e9:.2f} GB, memory_budget_bytes={memory_budget_bytes / 1e9:.2f} GB)"
        )

    def accumulate(self, bin_ids, data, n, sumsq=None):
        """Add one (possibly already merged) partial's rows into the store."""
        bin_ids = np.asarray(bin_ids)
        with self.lock:
            if self.backend == "memory":
                self._data[bin_ids] += data
                self._n[bin_ids] += n
                if self.track_std:
                    self._sumsq[bin_ids] += sumsq
            else:
                # h5py fancy indexing requires strictly increasing indices;
                # bin_ids from np.unique()/np.sort() already are, but don't
                # assume callers always hand us that.
                order = np.argsort(bin_ids)
                ids_sorted = bin_ids[order]
                self._data[ids_sorted] = self._data[ids_sorted] + data[order]
                self._n[ids_sorted] = self._n[ids_sorted] + n[order]
                if self.track_std:
                    self._sumsq[ids_sorted] = self._sumsq[ids_sorted] + sumsq[order]

    def finalize(self):
        """Return (data, n, sumsq) handles: numpy arrays if in-memory, else
        live h5py Dataset objects (file left open -- see class docstring)."""
        if self.backend == "disk" and self._h5 is not None:
            self._h5.flush()
        return self._data, self._n, self._sumsq


def _write_to_store(partial, result_store):
    data, n, bin_ids, sumsq, _raw = partial
    result_store.accumulate(bin_ids, data, n, sumsq=sumsq)
    return True


def store_merge(partials, result_store, event_shape, arity=8):
    """Merge many per-chunk partials directly into `result_store`.

    Unlike `tree_merge`, this never forms one big dense array anywhere in
    the dask graph -- groups of `arity` chunk partials are combined
    in-memory (cheap, bounded size, same as one `tree_merge` level), then
    scattered into `result_store` under its lock. The (possibly disk-
    backed) `result_store` is the only place the full-sized result needs
    to exist, which is what makes this suitable for very fine binning
    (many bins) where even the final dense array wouldn't fit in RAM.

    Ignores any `raw_out` (quantile) data in the partials -- that path
    isn't store-backed, see `merge_raw_quantile_data` instead.
    """
    combined = [
        delayed(_merge_partial_group)(partials[i : i + arity], event_shape)
        for i in range(0, len(partials), arity)
    ]
    written = [delayed(_write_to_store)(c, result_store) for c in combined]
    return delayed(lambda *_: result_store.finalize())(*written)


def _merge_partial_group(partials, event_shape):
    """Combine a small list of compact (data, n, bin_ids, sumsq, raw) partials into one.

    Stays in the same compact (only-bins-actually-present) representation
    so intermediate tree levels don't need to allocate the full
    ``n_bins_total``-sized array. `sumsq` is combined the same additive
    way as `data`/`n` when present; `raw` (quantile data) is concatenated
    per bin rather than summed, since it can't be reduced online.
    """
    bin_ids = np.unique(np.concatenate([p[2] for p in partials]))
    data_out = np.zeros((len(bin_ids), *event_shape))
    n_out = np.zeros(len(bin_ids))
    track_std = partials[0][3] is not None
    sumsq_out = np.zeros((len(bin_ids), *event_shape)) if track_std else None
    track_raw = partials[0][4] is not None
    raw_out = {} if track_raw else None

    for data, n, ids, sumsq, raw in partials:
        idx = np.searchsorted(bin_ids, ids)
        data_out[idx] += data
        n_out[idx] += n
        if track_std:
            sumsq_out[idx] += sumsq
        if track_raw:
            for bin_id, (vs, ws) in raw.items():
                if bin_id in raw_out:
                    mv, mw = raw_out[bin_id]
                    raw_out[bin_id] = (np.concatenate([mv, vs]), np.concatenate([mw, ws]))
                else:
                    raw_out[bin_id] = (vs, ws)

    return data_out, n_out, bin_ids, sumsq_out, raw_out


def _finalize(partial, n_bins_total, event_shape):
    """Expand a compact partial into the dense [n_bins_total, *event_shape] result."""
    data, n, bin_ids, sumsq, raw = partial
    data_out = np.zeros((n_bins_total, *event_shape))
    n_out = np.zeros(n_bins_total)
    data_out[bin_ids] = data
    n_out[bin_ids] = n
    sumsq_out = None
    if sumsq is not None:
        sumsq_out = np.zeros((n_bins_total, *event_shape))
        sumsq_out[bin_ids] = sumsq
    return data_out, n_out, sumsq_out, raw


def tree_merge(partials, n_bins_total, event_shape, arity=8):
    """Merge many delayed per-chunk partials via a balanced tree.

    A single flat ``delayed(merge)(all_partials)`` call (as in the
    original prototype) gathers *every* chunk's result onto one worker at
    once -- for the ~1e5-1e6 chunks a real 4000x4000x1e6 stack produces
    (at typical per-image chunking) that reintroduces the same kind of
    memory blowup the rolling approach was meant to avoid. Merging in a
    tree bounds any single task to ``arity`` partials, at the cost of
    O(log(n_chunks)) sequential depth instead of O(1).

    Returns a Delayed resolving to ``(data_out, n_in_bins, sumsq_out, raw_out)``,
    the last two ``None`` unless ``compute_std``/``compute_quantile`` were
    enabled on the input partials.
    """
    level = list(partials)
    while len(level) > 1:
        groups = [level[i : i + arity] for i in range(0, len(level), arity)]
        level = [delayed(_merge_partial_group)(g, event_shape) for g in groups]
    return delayed(_finalize)(level[0], n_bins_total, event_shape)


def _merge_raw_group(raw_dicts):
    """Concatenate per-bin (values, weights) across several chunks' raw dicts."""
    merged = {}
    for raw in raw_dicts:
        for bin_id, (vs, ws) in raw.items():
            if bin_id in merged:
                mv, mw = merged[bin_id]
                merged[bin_id] = (np.concatenate([mv, vs]), np.concatenate([mw, ws]))
            else:
                merged[bin_id] = (vs, ws)
    return merged


def _get_raw(partial):
    return partial[4]


def merge_raw_quantile_data(partials, arity=8):
    """Tree-merge many chunks' ``raw_out`` data into one ``{bin_id: (values, weights)}``.

    Takes the full per-chunk partials as returned by `bin_running_ref`
    (called with ``compute_quantile=True``) -- same input shape as
    `tree_merge`/`store_merge`, no need to pre-extract the `raw_out` field.

    Concatenates rather than reduces, so this is only appropriate when you
    already know each bin's total event count is small enough to hold in
    memory -- see `bin_running_ref`'s `compute_quantile` docs. Returns a
    Delayed; compute it (or pass straight to `finalize_quantile` inside
    another delayed/compute call) to get the actual dict.
    """
    level = [delayed(_get_raw)(p) for p in partials]
    while len(level) > 1:
        groups = [level[i : i + arity] for i in range(0, len(level), arity)]
        level = [delayed(_merge_raw_group)(g) for g in groups]
    return level[0]


def finalize_quantile(raw_dict, n_bins_total, event_shape, quantiles=(0.1587, 0.5, 0.8413), weighted=False):
    """Compute per-bin (weighted) quantiles from `merge_raw_quantile_data`'s output.

    Parameters
    ----------
    raw_dict : dict
        ``{bin_id: (values, weights)}``, e.g. from computing the Delayed
        returned by `merge_raw_quantile_data`.
    quantiles : sequence of float in [0, 1]
        Defaults to a median and a +-1 sigma-equivalent interval, matching
        `escape.storage.storage.Scan.weighted_stat`'s convention.
    weighted : bool, optional
        If False (default), uses ``np.percentile(..., axis=0)`` -- fully
        vectorized over the event shape, fast. If True, uses
        `escape.utilities.weighted_quantile` looped over every pixel of
        the event shape -- correct, but O(prod(event_shape)) individual
        calls, impractical for large images. Prefer `compute_std` at that
        scale, or restrict this to a small ROI.

    Returns
    -------
    ndarray, shape (len(quantiles), n_bins_total, *event_shape)
        NaN where a bin had no data.
    """
    quantiles = np.asarray(quantiles)
    out = np.full((len(quantiles), n_bins_total, *event_shape), np.nan)
    for bin_id, (values, weights) in raw_dict.items():
        if weighted:
            from ..utilities import weighted_quantile

            for idx in np.ndindex(*values.shape[1:]):
                sel = (slice(None),) + idx
                out[(slice(None), bin_id) + idx] = weighted_quantile(
                    values[sel], quantiles, sample_weight=weights
                )
        else:
            out[:, bin_id] = np.percentile(values, quantiles * 100, axis=0)
    return out


def _demo(
    len_all=10000,
    event_shape=(1000, 1000),
    n_bins_total=200,
    n_refs=5,
    n_ref_min=1,
    frac_signal=0.8,
    chunk_size=50,
    memory_budget_bytes=2 * 1024**3,
    merge_arity=8,
    result_backend="auto",
    result_path=None,
    cmp_type="none",
    ref_weighted=True,
    compute_std=False,
):
    # | variable | meaning                                                     |
    # |----------|--------------------------------------------------------------|
    # | data     | data with first dimension being run over                     |
    # | weights  | per-event weight, same length as data.shape[0]               |
    # | idop     | event is used as a "signal"/binned event                     |
    # | iref     | event is usable as a rolling reference                       |
    # | bins     | target bin for each idop event (undefined for non-idop rows) |
    #
    # `chunk_size`: pass an int for a fixed processing chunk size (along the
    # event axis), or "auto" to derive it from `memory_budget_bytes` via
    # `suggest_chunk_size` -- see that function's docstring for the
    # speed-vs-memory trade-off this controls.
    #
    # `result_backend`: "auto" picks "disk" (HDF5-backed `ResultStore`) when
    # the dense [n_bins_total, *event_shape] result wouldn't fit in
    # `memory_budget_bytes`, else "memory". Force with "memory"/"disk".

    event_nbytes = int(np.prod(event_shape) * np.dtype(float).itemsize)

    if chunk_size == "auto":
        chunk_size = suggest_chunk_size(
            event_nbytes,
            n_bins_total,
            frac_signal,
            memory_budget_bytes,
            n_ref_min=n_ref_min,
        )
        print(f"suggest_chunk_size picked chunk_size={chunk_size}")

    data = da.random.random_sample([len_all, *event_shape]) + 1
    weights = da.random.random_sample(len_all)

    idop = da.zeros(len_all, dtype=bool)
    idop[np.random.choice(data.shape[0], size=int(frac_signal * len_all), replace=False)] = True
    iref = ~idop

    bins_idop = da.from_array(np.random.randint(0, n_bins_total, size=int(idop.sum().compute())))
    bins = -1 * np.ones_like(idop)
    bins[(idop,)] = bins_idop

    # `.rechunk(chunk_size)` alone would also tile the spatial (event) axes
    # since a single int applies the target chunk size to every dimension;
    # we only want chunking along axis 0, so re-collapse axes 1/2 back to
    # one chunk.
    data = data.rechunk(chunk_size)
    event_chunks = data.chunks[0]
    data = data.rechunk((event_chunks, *data.shape[1:]))
    bins = bins.rechunk(event_chunks)
    idop = idop.rechunk(event_chunks)
    iref = iref.rechunk(event_chunks)
    weights = weights.rechunk(event_chunks)

    partials = [
        delayed(bin_running_ref)(
            data_d, idop_d, iref_d, bins_d, weights_d, n_refs, n_ref_min,
            cmp_type=cmp_type, ref_weighted=ref_weighted, compute_std=compute_std,
        )
        for data_d, idop_d, iref_d, bins_d, weights_d in zip(
            data.to_delayed().ravel(),
            idop.to_delayed(),
            iref.to_delayed(),
            bins.to_delayed(),
            weights.to_delayed(),
        )
    ]

    t0 = time.time()
    result_store = ResultStore(
        n_bins_total,
        data.shape[1:],
        dtype=data.dtype,
        backend=result_backend,
        memory_budget_bytes=memory_budget_bytes,
        path=result_path,
        track_std=compute_std,
    )

    if result_store.backend == "memory":
        # dense result fits comfortably -- tree_merge's final delayed value
        # is a (data_out, n_in_bins, sumsq_out, raw_out) tuple of
        # differently-shaped pieces, so it can't be wrapped with
        # da.from_delayed (which expects a delayed producing one array of
        # the declared shape) -- compute the Delayed directly instead.
        merged = tree_merge(partials, n_bins_total, data.shape[1:], arity=merge_arity)
        data_out, _n_in_bins, _sumsq_out, _raw_out = merged.compute()
    else:
        # dense result doesn't fit in the memory budget -- accumulate
        # straight into the (HDF5-backed) store instead of ever forming
        # one big array in the graph.
        finalize_task = store_merge(partials, result_store, data.shape[1:], arity=merge_arity)
        data_out, _n_in_bins, _sumsq_out = finalize_task.compute()  # live h5py datasets

    print(f"compute took {time.time() - t0:.2f}s -> {data_out.shape}"
          + (f" (file: {result_store.path})" if result_store.path else ""))


if __name__ == "__main__":
    _demo()


# =============================================================================
# Open issues (not fixed here -- need your input on intended semantics)
# =============================================================================
#
# 1. Rolling reference resets at every chunk boundary.
#    `rolling_refs` is created fresh inside each `bin_running_ref` call, so
#    it never sees reference events from a previous chunk. Confirmed by
#    simulation: loss is concentrated at the *start* of each chunk (the
#    warm-up before n_ref_min refs have refilled the reset deque), not the
#    end -- once the window fills once, it never empties again for the
#    rest of that chunk. It shrinks fast with chunk size: at
#    frac_ref=0.2, n_ref_min=1, going chunk_size 20->50 takes average loss
#    from 1.5% to ~0%; below the threshold whole chunks can have zero
#    reference events and lose 100% of their signal events, not just a
#    few at an edge. Rule of thumb (now enforced as a warning in
#    `suggest_chunk_size`): loss is negligible once
#    chunk_size >~ 10 * n_ref_min / frac_ref.
#    This only manages the *symptom* (choose chunk_size large enough that
#    loss is tolerable) -- it doesn't restore true cross-chunk continuity.
#    If exact continuity is ever needed instead of a tolerable loss:
#      a) a small sequential state-carry pass between consecutive chunks
#         (only ~n_refs frames of state per hop, so cheap even with many
#         chunks), or
#      b) `dask.array.overlap.map_overlap(fn, depth=(n_refs, 0, 0), ...)`,
#         which is the built-in dask primitive for exactly this
#         "each block also needs a bit of its neighbor" pattern.
#    Still waiting on your read of the deque logic before touching this.
#
# 2. RESOLVED: the rolling reference frames used to only gate (via
#    `len(rolling_refs)`) rather than actually correct the signal value.
#    `bin_running_ref` now takes `cmp_type="none"|"difference"|"ratio"`
#    (matching ArrayTools.compare_to_reference's existing convention) and
#    `ref_weighted=True/False`, applying `data[event] - ref_mean` or
#    `data[event] / ref_mean` (ref_mean = weighted or unweighted mean of
#    the current rolling window) before binning. Default stays "none" for
#    backward compatibility with the original gate-only behavior.
#
# 3. `pending` only ever holds more than one index during the initial
#    warm-up (before n_ref_min refs have been seen) -- once the window is
#    filled, every idop event is flushed on the same iteration it's
#    appended. Is batching multiple pending events together intentional,
#    or is per-event flushing the actual desired behavior?
