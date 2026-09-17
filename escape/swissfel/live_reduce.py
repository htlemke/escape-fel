"""Incrementally reduce a SwissFEL run while it's still being acquired.

A run's scan-info JSON (``aux/scan_info_rel.json``) is rewritten as each
scan step completes, but different data sources for the same step finish
independently -- in a live run watched during development, bsread
(BSDATA.h5) lagged the detector/PV files by 10+ steps at times, and
``namespace_monitor.h5`` (the CA-monitor dump) only appears once, at the
very end. A file can also exist with a plausible size long before it's
actually complete: racing ``h5py.File(path, "r")`` against a file still
being written by the DAQ showed it fail cleanly (``bad object header
version number``, then ``addr overflow``) right up until the moment the
writer closed it, at which point it opens fine every time after --
``Path.exists()`` alone would have reported "ready" over a second before
that. See ``_h5_file_ready``/``count_ready_steps`` below, which this
module's polling loop uses instead of the existence-only check the
existing ``wait_for_data_files`` option relies on.

Run as a long-lived process on a server (see the ``__main__`` CLI at the
bottom), polling for newly-*ready* steps, reparsing just the growing
prefix with :func:`~escape.swissfel.parse.load_dataset_from_scan`, and
storing the small/scalar channels into a results file via
``DataSet.store_datasets_max_element_size()`` each time -- so the data is
already reduced and quick to load back
(:meth:`~escape.storage.DataSet.load_from_result_file`) during or after
acquisition, without re-parsing raw files from scratch each time
(``checknstore_parsing_result`` keeps repeated polls cheap) or waiting for
the whole run to finish.
"""
import time
from pathlib import Path

import h5py

from .parse import load_dataset_from_scan, interpret_raw_data_definition
from .cluster import readScanEcoJson_v01


def _h5_file_ready(path):
    """True if *path* can be opened cleanly as HDF5 right now.

    Not just ``Path.exists()`` -- see this module's docstring for why that
    alone isn't a reliable completeness signal for a file the DAQ may still
    be writing.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        with h5py.File(p, "r"):
            pass
        return True
    except Exception:
        return False


def count_ready_steps(scan_info_file, exclude_from_files=()):
    """How many leading scan steps have every referenced HDF5 file ready.

    Re-reads *scan_info_file* fresh (it's rewritten incrementally as a run
    progresses -- watched growing 7202 -> 7766 -> 10029 bytes etc. as steps
    completed, live) and returns ``(n_ready, n_total)``: the length of the
    longest *prefix* of ``scan_files`` where every referenced file opens
    cleanly, and the number of steps currently listed at all. A prefix, not
    just a count of individually-ready steps, because data sources finish
    each step independently and out of order with each other, but
    concatenating steps for a channel assumes a contiguous run from step 0.

    Parameters
    ----------
    scan_info_file : str or Path
        Path to the scan-info JSON file (``aux/scan_info_rel.json``).
    exclude_from_files : sequence of str, optional
        Same meaning as elsewhere -- substrings that exclude a referenced
        file from consideration entirely.

    Returns
    -------
    (int, int)
        ``(n_ready, n_total)``.
    """
    s, _ = readScanEcoJson_v01(
        scan_info_file, exclude_from_files=list(exclude_from_files)
    )
    n_ready = 0
    for files_step in s["scan_files"]:
        if not files_step or not all(_h5_file_ready(f) for f in files_step):
            break
        n_ready += 1
    return n_ready, len(s["scan_files"])


def live_reduce_scan(
    run_number=None,
    pgroup=None,
    exp_name=None,
    instrument="bernina",
    metadata_file=None,
    result_filename="auto",
    results_directory="./",
    result_type="zarr",
    checknstore_parsing_result="work_directory",
    max_element_size=5000,
    poll_interval=15,
    idle_polls_before_final_pass=4,
    max_polls=None,
    merge_data_sources=True,
    parse_version=3,
    verbose=1,
    **load_kwargs,
):
    """Watch a scan as it's being written and incrementally reduce it.

    See this module's docstring for the reasoning behind the approach.
    Blocks until acquisition looks finished (no newly-ready step for
    ``idle_polls_before_final_pass`` consecutive polls, or ``max_polls``
    reached), then does one final unrestricted pass with
    ``merge_data_sources=True`` to pick up status/``namespace_monitor.h5``
    data, which only finalizes at the very end of a run -- in the run this
    was developed against, tens of minutes after the last per-step data
    file, and namespace_monitor.h5 specifically has had reliability issues
    finishing at all for some runs, so that final pass's merge may end up
    contributing nothing; the per-step Array data collected along the way
    is unaffected either way.

    Parameters
    ----------
    run_number, pgroup, exp_name, instrument, metadata_file :
        Same as :func:`~escape.swissfel.parse.load_dataset_from_scan` --
        used once, up front, to locate the scan-info JSON via
        :func:`~escape.swissfel.parse.interpret_raw_data_definition`. Every
        poll re-reads that same file, which is what actually grows.
    result_filename, results_directory, result_type :
        Where to write the growing results file. ``"auto"`` (default)
        derives the name from the run, same as ``load_dataset_from_scan``,
        and resolves consistently across polls since the located
        scan-info path doesn't change.
    checknstore_parsing_result :
        Parse-level cache (same meaning as ``load_dataset_from_scan``) --
        keyed by file path and size, so it skips re-scanning steps already
        seen in an earlier poll, keeping each poll's cost proportional to
        the *new* steps, not the whole run so far. Passing ``False`` here
        defeats most of the point of polling repeatedly.
    max_element_size : int, optional
        Forwarded to ``store_datasets_max_element_size()`` each poll.
    poll_interval : float, optional
        Seconds between polls. Defaults to ``15``.
    idle_polls_before_final_pass : int, optional
        Consecutive polls with no newly-ready step before concluding
        acquisition has finished and doing the final pass. Defaults to
        ``4`` (so, by default, roughly ``4 * poll_interval`` seconds of
        true silence).
    max_polls : int or None, optional
        Hard cap on polls regardless of activity, as a safety net. ``None``
        (default) relies on ``idle_polls_before_final_pass`` instead.
    merge_data_sources : bool, optional
        Forwarded to the *final* pass's ``load_dataset_from_scan()`` call
        only (the incremental polls always pass ``False`` -- status/monitor
        data isn't valid mid-run anyway, see above). Defaults to ``True``.
    parse_version : {1, 2, 3}, optional
        Forwarded to every ``load_dataset_from_scan()`` call. Defaults to
        ``3``. Prefer ``2`` or ``3`` here (thread-based scanning) over
        ``1`` if this ever runs somewhere ``multiprocessing``'s "spawn"
        start method can't re-import ``__main__`` (e.g. piped stdin).
    **load_kwargs :
        Forwarded to every ``load_dataset_from_scan()`` call (e.g.
        ``alias_mappings``, ``load_dap_data``, ``exclude_from_files``,
        ``perm_result_file``).

    Returns
    -------
    escape.storage.DataSet
        The final, fully merged DataSet -- the same data that's been
        incrementally written to the results file throughout, so
        ``DataSet.load_from_result_file()`` against that same path picks
        up right where this left off, from any other process, at any time.
    """
    metadata_files = interpret_raw_data_definition(
        metadata_file=metadata_file,
        run_numbers=[run_number] if run_number is not None else None,
        pgroup=pgroup,
        exp_name=exp_name,
        instrument=instrument,
        verbose=bool(verbose),
    )
    if not metadata_files:
        raise FileNotFoundError(
            "Could not locate a scan-info JSON for this run -- if "
            "acquisition hasn't started yet, wait for its aux/ directory "
            "to appear before calling live_reduce_scan()."
        )
    scan_info_file = metadata_files[0]
    exclude_from_files = load_kwargs.get("exclude_from_files", [])

    n_stored = 0
    idle = 0
    poll_n = 0
    d = None

    while True:
        n_ready, n_total = count_ready_steps(scan_info_file, exclude_from_files)
        if verbose:
            print(
                f"poll {poll_n}: {n_ready}/{n_total} step(s) ready "
                f"(previously stored through step {n_stored})",
                flush=True,
            )

        if n_ready > n_stored:
            d = load_dataset_from_scan(
                metadata_file=scan_info_file,
                result_filename=result_filename,
                results_directory=results_directory,
                result_type=result_type,
                clear_result_file=(poll_n == 0),
                checknstore_parsing_result=checknstore_parsing_result,
                clear_parsing_result=False,
                step_selection=slice(0, n_ready),
                parse_version=parse_version,
                merge_data_sources=False,
                verbose=verbose,
                **load_kwargs,
            )
            try:
                d.store_datasets_max_element_size(max_element_size=max_element_size)
            except Exception as exc:
                print(f"poll {poll_n}: store failed: {exc!r}", flush=True)
            n_stored = n_ready
            idle = 0
        else:
            idle += 1

        poll_n += 1
        if max_polls and poll_n >= max_polls:
            if verbose:
                print(f"reached max_polls={max_polls}, stopping.", flush=True)
            break
        if idle >= idle_polls_before_final_pass:
            if verbose:
                print(
                    f"no newly-ready step for {idle} polls -- assuming "
                    f"acquisition finished.",
                    flush=True,
                )
            break
        time.sleep(poll_interval)

    if verbose:
        print("final pass: all steps + merge_data_sources...", flush=True)
    d = load_dataset_from_scan(
        metadata_file=scan_info_file,
        result_filename=result_filename,
        results_directory=results_directory,
        result_type=result_type,
        clear_result_file=False,
        checknstore_parsing_result=checknstore_parsing_result,
        clear_parsing_result=False,
        parse_version=parse_version,
        merge_data_sources=merge_data_sources,
        verbose=verbose,
        **load_kwargs,
    )
    d.store_datasets_max_element_size(max_element_size=max_element_size)
    if verbose:
        print("live_reduce_scan done.", flush=True)
    return d


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-number", type=int, required=True)
    ap.add_argument("--pgroup")
    ap.add_argument("--exp-name")
    ap.add_argument("--instrument", default="bernina")
    ap.add_argument("--result-filename", default="auto")
    ap.add_argument("--results-directory", default="./")
    ap.add_argument("--result-type", default="zarr", choices=["zarr", "h5"])
    ap.add_argument("--max-element-size", type=int, default=5000)
    ap.add_argument("--poll-interval", type=float, default=15)
    ap.add_argument("--idle-polls-before-final-pass", type=int, default=4)
    ap.add_argument("--max-polls", type=int, default=None)
    ap.add_argument("--parse-version", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--no-merge-data-sources", action="store_true")
    args = ap.parse_args()

    live_reduce_scan(
        run_number=args.run_number,
        pgroup=args.pgroup,
        exp_name=args.exp_name,
        instrument=args.instrument,
        result_filename=args.result_filename,
        results_directory=args.results_directory,
        result_type=args.result_type,
        max_element_size=args.max_element_size,
        poll_interval=args.poll_interval,
        idle_polls_before_final_pass=args.idle_polls_before_final_pass,
        max_polls=args.max_polls,
        parse_version=args.parse_version,
        merge_data_sources=not args.no_merge_data_sources,
    )
