"""Reading a Stream's retained samples while the EventWorker appends.

``.data``/``.eventIds`` are the live ``deque``s the worker appends to.
Iterating one while it does raises ``RuntimeError: deque mutated during
iteration``, and in a live plot that surfaced as

    Live plot update error: deque mutated during iteration

printed once per lost update: the frame was dropped, so plots refreshed
only intermittently and the more streams were running the worse it got
(observed on a real Bernina session with several live plots and a running
mean). DataManager.flat()/snapshot() copy under the append lock instead.
"""

import threading

import pytest

from escape.stream.escape_stream import DataManager


def _hammer(manager, stop, n_steps=3):
    i = 0
    while not stop.is_set():
        manager.append(float(i), i, i % n_steps)
        i += 1


def test_flat_never_races_the_appender():
    m = DataManager(maxlen=50)
    stop = threading.Event()
    writer = threading.Thread(target=_hammer, args=(m, stop), daemon=True)
    writer.start()
    try:
        for _ in range(300):
            values = m.flat()
            # Consistent in itself: a snapshot, not a view being mutated.
            assert all(isinstance(v, float) for v in values)
            assert len(values) <= 50 * 3
    finally:
        stop.set()
        writer.join(5)


def test_flat_with_ids_pairs_values_and_ids_from_one_snapshot():
    """to_frame and every correlation plot index these two in parallel, so
    they must come from the same lock hold - one append apart pairs the
    wrong samples."""
    m = DataManager(maxlen=50)
    stop = threading.Event()
    writer = threading.Thread(target=_hammer, args=(m, stop), daemon=True)
    writer.start()
    try:
        for _ in range(300):
            values, ids = m.flat(with_ids=True)
            assert len(values) == len(ids)
            # The appender writes value == float(id), so a desynchronised
            # pair is detectable rather than merely suspected.
            assert all(v == float(i) for v, i in zip(values, ids))
    finally:
        stop.set()
        writer.join(5)


def test_iterating_the_live_deques_is_what_used_to_break():
    """Documents the hazard the two tests above exist for: the raw attributes
    are still the live deques, and iterating them under load can raise."""
    m = DataManager(maxlen=50)
    stop = threading.Event()
    writer = threading.Thread(target=_hammer, args=(m, stop), daemon=True)
    writer.start()
    try:
        raised = False
        for _ in range(2000):
            try:
                [v for step in m.data for v in step]
            except RuntimeError:
                raised = True
                break
    finally:
        stop.set()
        writer.join(5)
    if not raised:
        pytest.skip("did not lose the race in this run; flat() is safe either way")
