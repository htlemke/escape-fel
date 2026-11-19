"""PulseCache / PulseJoin: aligning sources by pulse id.

Pure unit tests - no dispatcher, no threads in the primitives themselves, so
the alignment rules are pinned down independently of live data (which is how
a 12 s-late Jungfrau ROI combined with a 90 Hz event counter managed to be
unexplainable from the outside).
"""

import threading

import pytest

from escape.stream.pulse_align import PulseCache, PulseJoin, depth_for_delay


# -- depth sizing -----------------------------------------------------------


def test_depth_covers_the_measured_delay_with_margin():
    """A 12 s-late connection is ~1200 pulses behind at 100 Hz, so a cache
    bounded at a few hundred could never align it."""
    assert depth_for_delay(0) >= 200
    assert depth_for_delay(12.1) > 1200
    assert depth_for_delay(12.1) > depth_for_delay(1.0)
    # None/garbage must not explode: a delay is only ever a measurement.
    assert depth_for_delay(None) >= 200


# -- PulseCache -------------------------------------------------------------


def test_cache_backfills_a_later_event_from_an_earlier_one():
    c = PulseCache(depth=10)
    c.put(100, {"fast": 1.0})
    assert c.get(100) == {"fast": 1.0}
    assert c.stats()["hit_rate"] == 1.0


def test_cache_counts_misses_so_a_failing_alignment_is_visible():
    """The point of the counters: downstream, a miss and "no data" look
    identical, and that ambiguity is what made this undiagnosable."""
    c = PulseCache(depth=10)
    c.put(100, {"fast": 1.0})
    assert c.get(999) is None
    s = c.stats()
    assert (s["n_hit"], s["n_miss"]) == (0, 1)
    assert s["hit_rate"] == 0.0
    assert c.get(100) is not None
    assert c.stats()["hit_rate"] == 0.5


def test_cache_is_bounded_in_pulses_and_evicts_oldest_first():
    c = PulseCache(depth=3)
    for pid in range(10):
        c.put(pid, {"v": pid})
    assert len(c) == 3
    assert c.get(9) == {"v": 9}
    assert c.get(0) is None            # evicted
    assert c.stats()["n_evicted"] == 7


def test_cache_resize_adapts_when_a_delay_changes():
    c = PulseCache(depth=1000)
    for pid in range(500):
        c.put(pid, {"v": pid})
    c.set_depth(depth_for_delay(0.0))   # connection is prompt again
    assert len(c) <= depth_for_delay(0.0)
    c.set_depth(depth_for_delay(12.1))  # and late again
    assert c.depth > 1200


# -- PulseJoin --------------------------------------------------------------


def test_join_emits_a_pulse_once_both_sources_have_moved_past_it():
    j = PulseJoin(2)
    assert j.offer(0, 10, 1, {"a": 1}) == []      # source 1 may still speak
    assert j.offer(1, 10, 1, {"b": 2}) == []      # neither has moved past 10
    out = j.offer(0, 11, 2, {"a": 3})
    assert out == []                              # source 1 still at 10
    out = j.offer(1, 11, 2, {"b": 4})
    assert [pid for pid, _, _ in out] == [10]
    assert out[0][2] == {"a": 1, "b": 2}


def test_join_handles_mixed_rates_where_a_source_simply_has_no_data():
    """The bug in the predecessor: it fired only when *all* sources had
    contributed, which never happens for a 45 Hz detector merged with a
    90 Hz counter - those pulses were released only by a timeout."""
    j = PulseJoin(2)
    emitted = []
    for pid in range(1, 12):
        emitted += j.offer(0, pid, pid, {"fast": pid})      # every pulse
        if pid % 2 == 0:
            emitted += j.offer(1, pid, pid, {"slow": pid})  # every other one

    ids = [pid for pid, _, _ in emitted]
    assert ids == sorted(ids)                      # strictly in order
    assert len(ids) >= 8                           # not stuck waiting
    by_id = {pid: v for pid, _, v in emitted}
    assert by_id[2] == {"fast": 2, "slow": 2}      # both, when both had data
    assert by_id[3] == {"fast": 3}                 # fast only, not withheld


def test_join_emits_each_settled_pulse_once_and_in_order():
    j = PulseJoin(2)
    emitted = []
    for pid in (5, 6, 7, 8):
        emitted += j.offer(0, pid, pid, {"a": pid})
        emitted += j.offer(1, pid, pid, {"b": pid})
    ids = [pid for pid, _, _ in emitted]
    assert ids == sorted(set(ids))


def test_a_straggler_past_the_window_is_forwarded_not_discarded():
    """Deliberate: a source can be further behind than the join is holding
    (an unbounded skew between independent backends), and dropping its data
    would lose that source entirely. It is forwarded on its own and counted;
    ordering downstream is guarded by EventWorker._is_new_pulse, which exists
    for precisely this.
    """
    j = PulseJoin(2)
    emitted = []
    for pid in (5, 6, 7, 8):
        emitted += j.offer(0, pid, pid, {"a": pid})
        emitted += j.offer(1, pid, pid, {"b": pid})
    first = [pid for pid, _, _ in emitted][0]

    out = j.offer(1, first, 0, {"b": "late"})
    assert [pid for pid, _, _ in out] == [first]
    assert out[0][2] == {"b": "late"}
    assert j.stats()["n_late"] == 1


def test_a_restarted_sender_is_not_mistaken_for_a_straggler():
    """A sender that reconnects starts counting again (a local TestStream
    starts at 1). Same 20000-pulse convention as EventWorker._is_new_pulse."""
    j = PulseJoin(2)
    for pid in (10 ** 6, 10 ** 6 + 1):
        j.offer(0, pid, pid, {"a": pid})
        j.offer(1, pid, pid, {"b": pid})
    assert j.stats()["n_restarts"] == 0

    j.offer(0, 1, 1, {"a": 1})          # restarted, far below
    assert j.stats()["n_restarts"] == 1
    out = j.offer(1, 2, 2, {"b": 2})    # the new timeline proceeds normally
    assert [pid for pid, _, _ in out] == []
    out = j.offer(0, 2, 2, {"a": 2})
    assert [pid for pid, _, _ in out] == [1]


def test_timeout_pulses_forwards_partials_without_waiting():
    """A caller passing a small timeout_pulses is asking for low latency over
    completeness; the watermark rule alone would make it wait for a silent
    source."""
    j = PulseJoin(2, timeout_pulses=1)
    assert j.offer(0, 1, 1, {"a": 1}) == []
    out = j.offer(0, 2, 2, {"a": 2})     # source 1 has said nothing at all
    assert [pid for pid, _, _ in out] == [1]
    assert j.stats()["n_stale"] == 1


def test_a_silent_source_costs_depth_pulses_not_an_unbounded_stall():
    j = PulseJoin(2, depth=5)
    emitted = []
    for pid in range(1, 20):
        emitted += j.offer(0, pid, pid, {"a": pid})   # source 1 never speaks
    assert emitted, "a stalled source must not hold everything forever"
    assert j.stats()["n_forced"] > 0
    assert j.stats()["held"] <= 5


def test_join_flush_releases_everything_held():
    j = PulseJoin(2)
    j.offer(0, 1, 1, {"a": 1})
    j.offer(0, 2, 2, {"a": 2})
    out = j.flush()
    assert [pid for pid, _, _ in out] == [1, 2]
    assert j.stats()["held"] == 0


def test_join_is_safe_under_concurrent_sources():
    """Each source calls offer() from its own thread in the real handler."""
    j = PulseJoin(2, depth=10_000)
    collected = []
    lock = threading.Lock()

    def feed(source):
        local = []
        for pid in range(1, 400):
            local += j.offer(source, pid, pid, {f"c{source}": pid})
        with lock:
            collected.extend(local)

    threads = [threading.Thread(target=feed, args=(s,)) for s in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)

    ids = [pid for pid, _, _ in collected]
    assert len(ids) == len(set(ids)), "a pulse was emitted twice"
    assert j.stats()["n_emitted"] == len(ids)


def test_a_single_source_join_is_a_pass_through_one_pulse_behind():
    """n_sources=1 must still work (a handler can be configured with one),
    and then a pulse settles as soon as the next one arrives."""
    j = PulseJoin(1)
    assert j.offer(0, 1, 1, {"a": 1}) == []
    out = j.offer(0, 2, 2, {"a": 2})
    assert [pid for pid, _, _ in out] == [1]


def test_n_sources_must_be_sane():
    with pytest.raises(ValueError):
        PulseJoin(0)


# -- wired into the delay-group context -------------------------------------


def _two_group_context():
    from escape.stream.es_wrappers_datahub import (
        _DelayGroup, _GroupedDispatcherContext,
    )

    class _Handler:
        late_threshold_s = 5.0
        receive_timeout = 0.1
        url = None
        refused = set()
        auto_split = True
        _awaiting = {}

        def _set_groups(self, *a, **k):
            pass

    ctx = _GroupedDispatcherContext.__new__(_GroupedDispatcherContext)
    ctx.h = _Handler()
    ctx._offset = 0
    ctx._clock = 0.0
    ctx.groups = [_DelayGroup(["fast"]), _DelayGroup(["slow"])]
    ctx.groups[0].delay, ctx.groups[1].delay = 0.0, 12.1
    ctx._resize_caches()
    return ctx


def test_retention_is_sized_from_the_measured_delay():
    """A 12 s-late connection needs ~1200 pulses of retention. The previous
    bound was wall-clock (max delay + 5 s), which depended on how promptly
    the consumer was polled rather than on the skew."""
    ctx = _two_group_context()
    assert ctx.groups[0].cache.depth > 1200


def test_the_prompt_group_is_still_delivered_immediately():
    """The reason the connections are split at all: a fast consumer must not
    wait for the slow channel."""
    ctx = _two_group_context()
    event = ctx._event(0, ctx.groups[0], (1000, 0, {"fast": 1.0}))
    assert event.group == 0
    assert event.getFromSource("fast") == 1.0


def test_a_late_event_carries_the_prompt_values_for_the_same_pulse():
    ctx = _two_group_context()
    ctx._event(0, ctx.groups[0], (1000, 0, {"fast": 1.0}))
    late = ctx._event(1, ctx.groups[1], (1000, 0, {"slow": 2.0}))
    assert late.getFromSource("fast") == 1.0
    assert late.getFromSource("slow") == 2.0


def test_alignment_report_distinguishes_a_miss_from_absent_data():
    """What diagnose() was missing: downstream, a merge miss and "nothing
    arrived" look identical, so a combined Stream producing nothing could not
    be explained."""
    ctx = _two_group_context()
    ctx._event(0, ctx.groups[0], (1000, 0, {"fast": 1.0}))
    ctx._event(1, ctx.groups[1], (1000, 0, {"slow": 2.0}))   # hit
    ctx._event(1, ctx.groups[1], (999, 0, {"slow": 3.0}))    # never retained
    report = ctx.alignment_report()
    assert len(report) == 1
    assert "hit rate 0.50" in report[0]
    assert "fast" in report[0]
