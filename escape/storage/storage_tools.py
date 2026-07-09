from numbers import Number
import dask
import dask.array as da
from dask import delayed
from escape import utilities
from escape.utilities import MultipleRoiSelector, StepViewer, StepViewerP
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt


def _extract_component(result_tuple, idx):
    """Pull element `idx` out of a computed (data, n, sumsq[, raw]) tuple and
    materialize it to a plain ndarray -- works whether the element is
    already a numpy array (in-memory backend) or an h5py Dataset (disk
    backend), and is itself only evaluated once the caller actually
    computes the Delayed wrapping this call."""
    val = result_tuple[idx]
    return None if val is None else np.asarray(val[:])


class ArrayTools:
    def __init__(self, array):
        self._array = array

    def get_regions_of_interest_rectangular_2D(
        self,
        data_selection=slice(None, 100),
        rois={},
        show=True,
    ):
        def append_rois(s):
            s.result = {}
            for nam, roi in s.rois.items():
                roi = [int(np.round(tr)) for tr in roi]
                s.result[nam] = self._array[:, slice(*roi[2:]), slice(*roi[:2])]

        data = self._array[data_selection].mean(axis=0).compute()
        s = MultipleRoiSelector(data, rois=rois, callbacks_changeanyroi=[append_rois])
        if show:
            display(s)
        return s
    
    def compare_to_reference(
        self, 
        is_reference, 
        N_agg_ref=100, 
        weights=None, 
        cmp_type='ratio',
        axis_survey=None,
    ):  
        """Compare the array to a reference defined by a boolean mask is_reference.
        The comparison can be done by ratio or difference. The reference is aggregated
        over N_agg_ref pulses, and weights can be applied."""
        array = self._array
        if array.ndim >1:
            print("Warning: array has more than one dimension, no weights applied.")
            hdim = True
        else:
            hdim = False
        resort = is_reference.get_index_array(N_index_aggregation=N_agg_ref) # new sorting according to pulse id (/ real time) bins for reference taking
        array = resort.categorize(array)
        array = array.scan.tools.has_N_refsig(is_reference) # filter out steps that don't have any reference or signal 

        array_sig = array[~is_reference]
        array_ref = array[is_reference]
        
        if not hdim:
            if cmp_type =='ratio':
                array_cmp = array_sig.scan / array_ref.scan.weighted_stat(weights)[0]
            if cmp_type =='difference':
                array_cmp = array_sig.scan - array_ref.scan.weighted_stat(weights)[0]
        else:
            if cmp_type =='ratio':
                array_cmp = array_sig.scan / array_ref.scan.mean(axis=0)
            if cmp_type =='difference':
                array_cmp = array_sig.scan - array_ref.scan.mean(axis=0)

        if axis_survey and not hdim:
            array_ref.plot(axis=axis_survey,ms=.3,label="Reference, single pulse")
            # array_sig.plot(axis=axis.survey,ms=.3,label="Signal")

            array_ref.plot(axis=axis_survey,ms=.3, label='ref (off) single plse')
            array_ref.scan.plot(axis=axis_survey, label='Reference, aggregated')
            array_sig.scan.plot(axis=axis_survey, label='Signal, aggregated')

        return array_cmp

    def compare_to_reference_rolling_binned(
        self,
        is_signal=None,
        is_reference=None,
        bins=None,
        weights=None,
        cmp_type="ratio",
        ref_weighted=True,
        n_refs=10,
        n_ref_min=1,
        chunk_size="auto",
        memory_budget_bytes=2 * 1024**3,
        merge_arity=8,
        compute_std=True,
        compute_quantile=False,
        quantile_weighted=False,
        quantiles=(0.1587, 0.5, 0.8413),
        h5=None,
        compute=False,
    ):
        """Bin this Array by scan step, comparing each signal event to a
        rolling window of reference events, without ever reordering the
        full lazy array.

        The rolling-window sibling of :meth:`compare_to_reference`: instead
        of aggregating the reference over a fixed N pulses, it maintains a
        running window as it walks the data. Unlike
        :meth:`Array.digitize`/:meth:`Array.categorize` (which fancy-index
        the *entire* array into bin order -- expensive for a large lazily
        loaded stack, see the design notes in
        :mod:`escape.storage.rollin_dev`), this processes one dask chunk at
        a time: each chunk locally compares its signal events to the most
        recent reference events *it has seen so far* and accumulates
        directly into per-bin sums, which are then merged via a bounded
        tree (never gathering more than a few chunks' worth of data onto
        one worker at a time).

        ``is_signal``, ``is_reference``, ``weights`` and ``bins`` (when
        given as an Array) are matched to this Array's pulse IDs, not by
        position -- they don't need to already share this Array's index,
        length, or ordering.

        Nothing is computed by default (``compute=False``) -- every
        returned Array is dask-backed, so e.g. ``result[0].shape`` is free
        (bin count and event shape are known upfront, independent of the
        data values) and no chunk processing runs until you ask for it.

        **Computing more than one output**: `binned`, `n_signal` and `std`
        all depend on the *same* underlying per-chunk computation. Calling
        `.compute()` on each separately re-runs that whole computation
        once per call -- three times the cost for no reason. Either pass
        ``compute=True`` here (computes everything together, once), or do
        it yourself with ``escape.compute(binned, n_signal, std)`` /
        ``dask.compute(binned.data, n_signal.data, std.data, quantile)``.

        Parameters
        ----------
        is_signal : escape.Array, optional
            Boolean, per-event. True for events that should be binned.
            Defaults to ``~is_reference`` (every non-reference event is
            signal). Events with no matching pulse ID are treated as False.
        is_reference : escape.Array, optional
            Boolean, per-event. True for events usable as rolling
            reference. Defaults to all-False (no reference events) --
            only valid together with ``n_ref_min=0`` (which also forces
            ``cmp_type="none"``, since a comparison needs a reference).
        bins : escape.Array, optional
            An Array whose own scan-step grouping defines the bins (e.g.
            the output of :meth:`Array.digitize`) -- bin *n* is "the n-th
            scan step of `bins`". If not given, this Array's *own* scan
            step structure is used (bin *n* = this Array's n-th scan
            step), and no index matching is needed at all.
        weights : escape.Array, optional
            Per-event weight (e.g. an i0 channel). Defaults to uniform
            weights. Events with no matching pulse ID get weight 0 (i.e.
            they're excluded rather than distorting the average).
        cmp_type : {"none", "difference", "ratio"}, optional
            How a signal event is compared to its rolling reference window
            before being binned -- same convention as
            :meth:`compare_to_reference`. Default ``"ratio"`` (bin
            ``signal / ref_mean``).

            ``"ratio"`` is numerically unstable wherever the reference can
            be zero or near-zero (e.g. detector background far from a
            peak): tiny floating-point differences in how ``ref_mean``
            accumulates get massively amplified by the division, and
            results at those pixels/bins become meaningless. Prefer
            ``"difference"`` for data with a near-zero background, or make
            sure the reference is bounded away from zero first.
        ref_weighted : bool, optional
            If True (default), the rolling reference mean is weighted by
            ``weights``; if False, a plain unweighted mean.
        n_refs : int, optional
            Rolling window length (max number of reference events kept).
        n_ref_min : int, optional
            Minimum reference events required before a signal event is
            binned. See :mod:`escape.storage.rollin_dev`'s module docstring
            for the chunk-boundary loss this trades off against chunk size
            (also see ``chunk_size="auto"``, which accounts for it).
        chunk_size : int or "auto", optional
            Per-task processing chunk size along the event axis. "auto"
            (default) derives it from ``memory_budget_bytes`` via
            :func:`escape.storage.rollin_dev.suggest_chunk_size`.
        memory_budget_bytes : int, optional
            Target per-task memory budget used both for ``chunk_size="auto"``
            and for deciding whether the final result needs to be
            disk-backed (see ``h5``).
        merge_arity : int, optional
            Branching factor of the merge tree -- see
            :func:`escape.storage.rollin_dev.tree_merge`.
        compute_std : bool, optional
            Also return the per-bin standard deviation of the (comparison-
            applied) binned values. Cheap: accumulated in the same pass as
            the mean, no extra memory risk regardless of scale. Default True.
        compute_quantile : bool, optional
            Also return per-bin quantiles (``quantiles``, default a median
            +- 1 sigma-equivalent interval, matching
            :meth:`Scan.weighted_stat`'s convention). Unlike ``compute_std``
            this is **not** a streaming/memory-safe computation -- it
            materializes every contributing event of a bin at once. Only
            enable this once you know a bin's worth of data fits in memory.
            Default False.
        quantile_weighted : bool, optional
            If True, the quantile computation is weighted (via
            :func:`escape.utilities.weighted_quantile`, looped per pixel of
            the event shape -- slow for large images). If False (default),
            uses ``np.percentile`` unweighted, fully vectorized and fast.
        h5 : h5py.Group, optional
            If given, the binned result (and std, if requested) is written
            directly into this group as new datasets, and the returned
            Array wraps a lazy dask view of them rather than an in-memory
            array -- use this for binning too fine to hold densely in
            memory (e.g. thousands of bins x large images). If not given,
            the result is disk-backed automatically (to a temp file) only
            if it wouldn't fit in ``memory_budget_bytes``.
        compute : bool, optional
            If True, compute every requested output together (one shared
            dask pass -- see note above) before returning. If False
            (default), everything is returned lazy/dask-backed.

        Returns
        -------
        binned : escape.Array
            One row per bin, holding the (weighted) mean of the
            comparison-applied signal values.
        par_steps : pandas.DataFrame
            The scan-step metadata of whichever scan defined the bins
            (``bins.scan.par_steps`` if given, else ``self.scan.par_steps``).
        n_signal : escape.Array, shape (n_bins,)
            Summed weight (or plain count, if ``weights`` is None) of
            signal events that were actually binned per bin -- can be
            lower than the number of signal events assigned to that bin
            if some were dropped by the rolling-window warm-up.
        std : escape.Array or None
            Per-bin standard deviation, or None if ``compute_std=False``.
        quantile : dask.delayed.Delayed, numpy.ndarray, or None
            Resolves to shape ``(len(quantiles), n_bins, *event_shape)``.
            A Delayed if ``compute=False`` (call ``.compute()`` yourself),
            a plain ndarray if ``compute=True``, or None if
            ``compute_quantile=False``.
        """
        from ..utilities import match_array_to_index, bin_ids_from_source
        from . import rollin_dev as _rd
        from .storage import Array

        array = self._array
        n = len(array)
        index = array.index

        if is_reference is None:
            is_reference_arr = np.zeros(n, dtype=bool)
        else:
            is_reference_arr = match_array_to_index(index, is_reference, fill_value=False).astype(bool)

        if is_signal is None:
            is_signal_arr = ~is_reference_arr
        else:
            is_signal_arr = match_array_to_index(index, is_signal, fill_value=False).astype(bool)

        if n_ref_min > 0 and not is_reference_arr.any():
            # n_ref_min > 0 gates every bin on ever seeing that many reference
            # events -- with none at all, nothing is ever binned and every
            # bin silently comes out 0/0 = NaN. cmp_type != "none" already
            # implies n_ref_min >= 1 (enforced in bin_running_ref itself), so
            # this one check covers both cases.
            raise ValueError(
                "n_ref_min > 0 needs at least one reference event -- "
                "pass is_reference explicitly, or set n_ref_min=0 "
                "(only valid together with cmp_type='none')."
            )

        if weights is None:
            weights_arr = np.ones(n)
        else:
            weights_arr = match_array_to_index(index, weights, fill_value=0.0).astype(float)

        if bins is None:
            step_lengths = list(array.scan.step_lengths)
            bin_ids_arr = np.repeat(np.arange(len(step_lengths)), step_lengths)
            n_bins_total = len(step_lengths)
            bin_source_scan = array.scan
        else:
            bin_ids_arr = bin_ids_from_source(index, bins)
            n_bins_total = len(bins.scan.step_lengths)
            bin_source_scan = bins.scan
            is_signal_arr = is_signal_arr & (bin_ids_arr >= 0)

        par_steps = bin_source_scan.par_steps

        event_shape = tuple(array.shape[1:])
        event_nbytes = int(np.prod(event_shape, dtype=np.int64) * array.data.dtype.itemsize) if event_shape else array.data.dtype.itemsize
        frac_signal = float(is_signal_arr.mean()) if n else 0.0

        if chunk_size == "auto":
            chunk_size = _rd.suggest_chunk_size(
                event_nbytes, n_bins_total, frac_signal, memory_budget_bytes, n_ref_min=n_ref_min,
            )

        data = array.data
        if not isinstance(data, da.Array):
            data = da.from_array(data, chunks=(chunk_size, *event_shape))
        else:
            data = data.rechunk(chunk_size)
        event_chunks = data.chunks[0]
        data = data.rechunk((event_chunks, *data.shape[1:]))

        is_signal_da = da.from_array(is_signal_arr, chunks=event_chunks)
        is_reference_da = da.from_array(is_reference_arr, chunks=event_chunks)
        weights_da = da.from_array(weights_arr, chunks=event_chunks)
        bins_da = da.from_array(bin_ids_arr, chunks=event_chunks)

        partials = [
            delayed(_rd.bin_running_ref)(
                data_d, is_signal_d, is_reference_d, bins_d, weights_d, n_refs, n_ref_min,
                cmp_type=cmp_type, ref_weighted=ref_weighted,
                compute_std=compute_std, compute_quantile=compute_quantile,
            )
            for data_d, is_signal_d, is_reference_d, bins_d, weights_d in zip(
                data.to_delayed().ravel(),
                is_signal_da.to_delayed(),
                is_reference_da.to_delayed(),
                bins_da.to_delayed(),
                weights_da.to_delayed(),
            )
        ]

        # Decide memory vs. disk purely by size estimate -- constructing a
        # real ResultStore (which eagerly allocates its numpy buffers for
        # the "memory" backend) is only needed when we're actually going to
        # write into it via store_merge.
        nbytes_estimate = _rd.estimate_result_bytes(n_bins_total, event_shape, data.dtype) * (2 if compute_std else 1)
        use_disk = h5 is not None or nbytes_estimate > memory_budget_bytes

        if use_disk:
            result_store = _rd.ResultStore(
                n_bins_total, event_shape, dtype=data.dtype, backend="disk",
                memory_budget_bytes=memory_budget_bytes, parent_h5py=h5, track_std=compute_std,
            )
            merge_task = _rd.store_merge(partials, result_store, event_shape, arity=merge_arity)
        else:
            merge_task = _rd.tree_merge(partials, n_bins_total, event_shape, arity=merge_arity)

        # `merge_task` is a single Delayed; every da.from_delayed below shares
        # it, so dask only runs the underlying computation once *per shared
        # dask.compute() call* -- see the redundant-computation note above.
        data_out_da = da.from_delayed(
            delayed(_extract_component)(merge_task, 0), (n_bins_total, *event_shape), dtype=data.dtype
        )
        n_out_da = da.from_delayed(
            delayed(_extract_component)(merge_task, 1), (n_bins_total,), dtype=float
        )
        n_reshaped = n_out_da.reshape(-1, *([1] * len(event_shape)))
        mean = data_out_da / n_reshaped

        std_arr = None
        if compute_std:
            sumsq_out_da = da.from_delayed(
                delayed(_extract_component)(merge_task, 2), (n_bins_total, *event_shape), dtype=data.dtype
            )
            var = sumsq_out_da / n_reshaped - mean**2
            std_arr = da.sqrt(da.clip(var, 0, None))

        bin_index = np.arange(n_bins_total)
        bin_step_lengths = [1] * n_bins_total
        binned = Array(
            data=mean, index=bin_index, step_lengths=bin_step_lengths, parameter=bin_source_scan.parameter,
        )
        n_signal = Array(
            data=n_out_da, index=bin_index, step_lengths=bin_step_lengths, parameter=bin_source_scan.parameter,
        )
        std = (
            Array(data=std_arr, index=bin_index, step_lengths=bin_step_lengths, parameter=bin_source_scan.parameter)
            if compute_std
            else None
        )

        quantile = None
        if compute_quantile:
            raw_merged = _rd.merge_raw_quantile_data(partials, arity=merge_arity)
            quantile = delayed(_rd.finalize_quantile)(
                raw_merged, n_bins_total, event_shape, quantiles=quantiles, weighted=quantile_weighted,
            )

        if compute:
            to_compute = [binned.data, n_signal.data]
            if compute_std:
                to_compute.append(std.data)
            if compute_quantile:
                to_compute.append(quantile)
            computed = dask.compute(*to_compute)
            it = iter(computed)
            binned = Array(data=next(it), index=bin_index, step_lengths=bin_step_lengths, parameter=bin_source_scan.parameter)
            n_signal = Array(data=next(it), index=bin_index, step_lengths=bin_step_lengths, parameter=bin_source_scan.parameter)
            if compute_std:
                std = Array(data=next(it), index=bin_index, step_lengths=bin_step_lengths, parameter=bin_source_scan.parameter)
            if compute_quantile:
                quantile = next(it)

        return binned, par_steps, n_signal, std, quantile

    def timetool_binning(self,timetool, time_vec=None, time_bins=None):
        array = self._array

        if not time_vec: 
            t = timetool.scan.par_steps.iloc[:,0] # timevec scan

        t_tt = timetool.scan + t # taking the time of the step and adding the time tool delay for each shot - the real measured delay 
        
        tt_med = timetool.nanmedian() # try to get the average tt values as median, for binning. 

        if isinstance(time_bins,Number):
            time_bins = np.arange(
                utilities.roundto(np.nanmin(t)+tt_med,time_bins) - time_bins/2,
                utilities.roundto(np.nanmax(t)+tt_med,time_bins) + time_bins/2,
                time_bins)
        t_tt_binned = t_tt.digitize(time_bins)
        
        array_tt = (t_tt_binned).categorize(array)
        
        return array_tt

#>>>>>>>>>>>>>>>


def timetool_binning(array,timetool, time_vec=None, time_bins=None, tbinsize=20e-15):

    if not time_vec: 
        t = timetool.scan.par_steps.iloc[:,0] # timevec scan

    t_tt = timetool.scan + t # taking the time of the step and adding the time tool delay for each shot - the real measured delay 
    
    tt_med = timetool.nanmedian() # try to get the average tt values as median, for binning. 

    if isinstance(time_bins,Number):
        time_bins = np.arange(
            utilities.roundto(np.nanmin(t)+tt_med,time_bins),
            utilities.roundto(np.nanmax(t)+tt_med,time_bins)+time_bins,
            tbinsize)
    t_tt_binned = t_tt.digitize(time_bins)
    
    array_tt = (t_tt_binned).categorize(array)
    
    return array_tt


#<<<<<<<<<<<<<<<


def timetool_binning_dev(
    array,
    timetool,
    time_vec=None,
    time_bins=None,
    tbinsize=20e-15,
    long_delay_binsize=200e-15,
    long_delay_threshold_factor=5.0,
):
    """Bin *array* along a timetool-corrected delay axis, with special handling
    for isolated "long-delay" scan points.

    Typical pump-probe scenario
    ---------------------------
    The nominal delay axis (``time_vec``) is densely sampled around time zero
    with step spacing comparable to ``tbinsize``.  One or more isolated points
    lie far outside this range (e.g. +10 ps, +100 ps reference delays).
    Each shot also carries a per-shot timetool jitter correction so the true
    corrected delay for shot *k* in step *i* is::

        t_corrected[k] = time_vec[i] + timetool[k]

    ASCII layout of the resulting bin structure::

        counts
          |
          |  +--+--+--+--+--+--+--+--+--+          LD bin       LD bin
          |  |  |  |  |  |  |  |  |  |  |        +-------+    +-------+
          |  |  |  |  |  |  |  |  |  |  |        |       |    |       |
          +--+--+--+--+--+--+--+--+--+--+--------+-------+----+-------+-> time
             ^                          ^         ^            ^
         t_min+tt_med           t_max+tt_med  t_ld1+tt_med  t_ld2+tt_med
                                         |<-- gap -->|
                                            filtered

    Binning strategy
    ----------------
    - **Dense region**: fine bins of width ``tbinsize`` spanning
      ``[t_min + tt_med, t_max + tt_med]``.
    - **Long-delay points**: one isolated bin of width ``long_delay_binsize``
      centred on ``t_ld + tt_med`` per detected isolated step.
    - **Gap bins** between isolated and dense regions are removed after
      digitisation by selecting only scan steps whose corresponding bin width
      does not exceed ``1.5 * max(tbinsize, long_delay_binsize)``.

    Long-delay detection
    --------------------
    A step index *i* is classified as a long-delay point when *all* of its
    neighbours are separated from it by a gap larger than
    ``long_delay_threshold_factor * median(gaps)``.  Boundary points (first /
    last) only need to satisfy the single gap they have.

    Parameters
    ----------
    array : escape.Array
        Detector signal to bin.
    timetool : escape.Array
        Per-shot timetool delay correction sharing the same scan as *array*.
    time_vec : array-like, optional
        Nominal time delay for each scan step.  When ``None`` the first scan
        parameter column of *timetool* is used.
    time_bins : array-like or Number, optional
        Explicit bin edges (array) or range-snapping interval (Number, same
        semantics as the original ``timetool_binning``).  When supplied the
        long-delay detection and gap filtering are bypassed for array input;
        Number input still activates the long-delay logic.
    tbinsize : float
        Width of the fine bins in the dense region (s).  Default 20 fs.
    long_delay_binsize : float
        Width of the isolated bin for each long-delay point (s).  Default 200 fs.
    long_delay_threshold_factor : float
        Multiplier of the median gap used to identify wide (long-delay)
        separations.  Default 5.

    Returns
    -------
    escape.Array
        *array* re-binned along the timetool-corrected delay axis.
    """
    # ── 1. nominal time per step ─────────────────────────────────────────────
    if time_vec is None:
        t = timetool.scan.par_steps.iloc[:, 0].values
    else:
        t = np.asarray(time_vec, dtype=float)

    # ── 2. per-shot corrected time ───────────────────────────────────────────
    t_tt = timetool.scan + t

    tt_med = float(timetool.nanmedian())

    # ── 3. build bin edges ───────────────────────────────────────────────────
    explicit_bins = time_bins is not None and not isinstance(time_bins, Number)
    if explicit_bins:
        all_bins = np.asarray(time_bins, dtype=float)
        t_long = np.array([])
    else:
        # detect long-delay (isolated) points
        gaps = np.abs(np.diff(t))
        median_gap = np.median(gaps)
        wide_gap = gaps > long_delay_threshold_factor * median_gap

        left_wide = np.concatenate([[True], wide_gap])
        right_wide = np.concatenate([wide_gap, [True]])
        is_long_delay = left_wide & right_wide

        t_normal = t[~is_long_delay]
        t_long = t[is_long_delay]

        if len(t_normal) == 0:
            raise ValueError(
                "All time_vec points were classified as long-delay. "
                "Reduce long_delay_threshold_factor or supply explicit time_bins."
            )

        # fine bins for the dense region
        if isinstance(time_bins, Number):
            rng_lo = utilities.roundto(np.nanmin(t_normal) + tt_med, time_bins)
            rng_hi = utilities.roundto(np.nanmax(t_normal) + tt_med, time_bins)
            fine_bins = np.arange(rng_lo - tbinsize / 2, rng_hi + tbinsize, tbinsize)
        else:
            bin_lo = utilities.roundto(np.nanmin(t_normal) + tt_med, tbinsize) - tbinsize / 2
            bin_hi = utilities.roundto(np.nanmax(t_normal) + tt_med, tbinsize) + tbinsize / 2
            fine_bins = np.arange(bin_lo, bin_hi + tbinsize * 0.5, tbinsize)

        # isolated narrow bins for each long-delay point
        ld_edges = []
        for t_ld in t_long:
            center = t_ld + tt_med
            ld_edges.append(center - long_delay_binsize / 2)
            ld_edges.append(center + long_delay_binsize / 2)

        all_bins = (
            np.sort(np.concatenate([fine_bins, np.array(ld_edges)]))
            if ld_edges
            else fine_bins
        )

    # ── 4. digitise ──────────────────────────────────────────────────────────
    t_tt_binned = t_tt.digitize(all_bins)

    # ── 5. filter out gap steps ───────────────────────────────────────────────
    if len(t_long) > 0 and not explicit_bins:
        bin_widths = np.diff(all_bins)
        max_valid_width = max(tbinsize, long_delay_binsize) * 1.5

        step_centers = t_tt_binned.scan.par_steps.iloc[:, 0].values
        step_bin_idx = np.searchsorted(all_bins, step_centers) - 1
        step_bin_idx = np.clip(step_bin_idx, 0, len(bin_widths) - 1)
        valid_steps = np.where(np.diff(all_bins)[step_bin_idx] <= max_valid_width)[0].tolist()

        if valid_steps:
            t_tt_binned = t_tt_binned.scan[valid_steps]

    # ── 6. categorise target array ───────────────────────────────────────────
    return t_tt_binned.categorize(array)



        # return (N_sig <= len(self._array[is_sig])) and (N_ref <= len(self._array[is_ref]))


class ScanTools:
    def __init__(self, scan):
        self._scan = scan

    def view_step_averages(
        self,
        data_selection=slice(None, 100),
    ):
        data = self._scan._array
        s = StepViewer(data)
        display(s)
        return s
    

    def has_N_refsig(self,is_ref,is_sig=None,N_ref = 1, N_sig=1):
        if is_sig is not None:
            sel = is_ref | is_sig
            #TODO
        
        is_ref = self._scan._array.categorize(is_ref).compute()

        valid_steps = []
        for n,step in enumerate(is_ref.scan):
            if (N_ref <= sum(step.data)) and (N_sig <= sum(~step.data)):
                valid_steps.append(n)

        return self._scan[valid_steps]
    

        


    def corr_ana_plot(self, referece, scanpar_name=None, axis=None):
        if not scanpar_name:
            names = list(self._scan.parameter.keys())
            scanpar_name = names[0]
        x = np.asarray(self._scan.parameter[scanpar_name]["values"]).ravel()
        corres = self._scan.correlation_analysis_to(referece)

        if not axis:
            axis = plt.gca()

        std = [tc[0] for tc in corres]
        std_fx = [tc[1] for tc in corres]

        ordercolors = ["b", "r"]
        for to, toc in zip([0, 1], ordercolors):
            axis.plot(
                x,
                [tc[to] for tc in std],
                toc + "--" + ".",
                label=f"poly. order:{to+1}; zero free",
            )
        for to, toc in zip([0, 1], ordercolors):
            axis.plot(
                x,
                [tc[to] for tc in std_fx],
                toc + "-" + "o",
                label=f"poly. order:{to+1}; zero fixed",
            )
        axis.set_xlabel(scanpar_name)
        axis.set_ylabel(self._scan._array.name)
        axis.legend()
        plt.tight_layout()

    def get_regions_of_interest_rectangular_2D(
        self,
        data_selection=slice(None, 100),
        rois={},
        show=True,
    ):
        def append_rois(s):
            s.result = {}
            for nam, roi in s.rois.items():
                roi = [int(np.round(tr)) for tr in roi]
                s.result[nam] = self._array[:, slice(*roi[2:]), slice(*roi[:2])]

        if show:
            data = self._scan._array
            sm = MultipleRoiSelector(
                data[data_selection].mean(axis=0).compute(),
                rois=rois,
                callbacks_changeanyroi=[append_rois],
            )
            s = StepViewerP(data, sm, data_selection=data_selection)
            StepViewerP.rois = property(lambda self: self.output.rois)

            def append_rois():
                s.result = {}
                for nam, roi in sm.rois.items():
                    rroi = [int(np.round(tr)) for tr in roi]
                    s.result[nam] = self._scan._array[
                        :, slice(*rroi[2:]), slice(*rroi[:2])
                    ]
                    s.result[nam].name = nam

            # Drop the placeholder append_rois(s) callback registered at
            # construction time (it closed over the wrong `self`) in favor
            # of this closure -- but keep sm's own internal change-hook (see
            # MultipleRoiSelector.convert_rois_to_int/remember_last_roi_in_cell)
            # instead of clobbering it.
            new_callbacks = []
            if sm.convert_rois_to_int_mode or sm.remember_last_roi_in_cell:
                new_callbacks.append(sm._on_any_roi_changed)
            new_callbacks.append(append_rois)
            sm.callbacks_changeanyroi = new_callbacks
            for rs in sm.roi_selectors:
                rs.callbacks_changeroi = list(new_callbacks)
            append_rois()
            display(s)
        else:

            class Dummy:
                pass

            s = Dummy()
            s.rois = {}
            s.result = {}
            for nam, troi in rois.items():
                roi = [int(np.round(tr)) for tr in troi]
                s.rois[nam] = roi
                s.result[nam] = self._scan._array[:, slice(*roi[2:]), slice(*roi[:2])]

        return s

    # def get_rectangular_roi(self,roidef={'test':(0,1,0,1)}):
    #     rroi = [int(np.round(tr)) for tr in roi]
    #     ret = self._array
