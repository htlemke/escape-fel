"""Aligning messages from several event sources by pulse id.

Two primitives, because the two callers need genuinely different things and
the single shared one they had before was wrong for both:

:class:`PulseCache`
    Bounded, pulse-id-keyed retention of one source's values, so a *later*
    source's event can be backfilled with what an earlier source said about
    the same pulse. This is what delay groups need
    (``_GroupedDispatcherContext``): a channel whose data reaches the
    dispatcher ~12 s late gets its own connection, and its events then carry
    the prompt channels' values for the same pulse, picked out of this cache.

    Deliberately *not* a join: the prompt connection's own events must keep
    being delivered the instant they arrive, or every fast consumer would
    wait for the slow channel -- which is the whole reason the connections
    were split in the first place.

:class:`PulseJoin`
    Settle-and-emit alignment of N independent sources, for when the merged
    event is the only thing anyone consumes (``MultiSourceEventHandler``,
    merging separate backends).

    A pulse is emitted once every source has delivered some *newer* pulse,
    i.e. once what each source has to say about it is known -- including
    "nothing". That watermark rule is what makes mixed-rate sources work:
    its predecessor fired only when all sources had contributed to a pulse,
    which is never true when a 45 Hz detector and a 90 Hz machine counter
    are merged, so those pulses were only ever released by a staleness
    timeout.

Both count what they drop. That is not decoration: a pulse that arrives
after its slot has been emitted, or one evicted because a source stalled,
is indistinguishable from "no data" downstream, and diagnosing a stream
that quietly produces nothing is otherwise guesswork.
"""

from collections import OrderedDict

import threading

__all__ = ["PulseCache", "PulseJoin"]

# A pulse id is a machine-wide counter at the accelerator's repetition rate.
# Depths below are expressed in pulses, so they need a rate to be derived
# from a delay in seconds; SwissFEL runs at 100 Hz, and overestimating the
# rate only makes a bound slightly generous.
PULSES_PER_SECOND = 100.0

# Extra pulses kept beyond the measured delay, for jitter in arrival time and
# for a delay that is drifting upwards between measurements.
DEPTH_MARGIN_PULSES = 200

# A pulse id this far below the last one means the sender restarted and began
# counting again (a local TestStream starts at 1), not that a straggler
# arrived. Same convention and value as EventWorker._is_new_pulse.
RESTART_GAP_PULSES = 20_000


def depth_for_delay(delay_s, rate=PULSES_PER_SECOND, margin=DEPTH_MARGIN_PULSES):
    """Pulses to retain to cover a source that lags by *delay_s* seconds."""
    return max(int(margin), int(float(delay_s or 0.0) * rate) + int(margin))


class PulseCache:
    """Recent ``pulse_id -> values`` of one source, bounded by pulse count.

    Bounded in *pulses* rather than in seconds: the thing that has to be
    covered is how many pulses the late source is behind, which is a pulse
    count, and deriving it from the measured delay (:func:`depth_for_delay`)
    makes the bound adapt by itself when a connection's lateness changes.
    The previous version bounded by wall-clock age instead, which coupled
    retention to how promptly the *consumer* was polled.
    """

    def __init__(self, depth=1000):
        self._values = OrderedDict()   # pulse_id -> values dict
        self.depth = int(depth)
        self.n_put = 0
        self.n_hit = 0
        self.n_miss = 0
        self.n_evicted = 0

    def __len__(self):
        return len(self._values)

    def set_depth(self, depth):
        """Resize, dropping the oldest pulses if the new depth is smaller."""
        self.depth = max(1, int(depth))
        self._trim()

    def put(self, pulse_id, values):
        self._values[pulse_id] = values
        self._values.move_to_end(pulse_id)
        self.n_put += 1
        self._trim()

    def get(self, pulse_id):
        """Values for *pulse_id*, or None - counted either way."""
        hit = self._values.get(pulse_id)
        if hit is None:
            self.n_miss += 1
        else:
            self.n_hit += 1
        return hit

    def clear(self):
        self._values.clear()

    def _trim(self):
        while len(self._values) > self.depth:
            self._values.popitem(last=False)
            self.n_evicted += 1

    def stats(self):
        """What this cache has been doing, for diagnostics.

        ``hit_rate`` is the share of lookups that found the pulse. A low rate
        on a late connection means its events are *not* in fact carrying the
        prompt channels' values, which is the difference between "this stream
        has no data because nothing arrived" and "because alignment is
        failing" -- previously unanswerable from outside.
        """
        looked = self.n_hit + self.n_miss
        return {
            "depth": self.depth,
            "held": len(self._values),
            "n_put": self.n_put,
            "n_hit": self.n_hit,
            "n_miss": self.n_miss,
            "n_evicted": self.n_evicted,
            "hit_rate": (self.n_hit / looked) if looked else None,
        }


class PulseJoin:
    """Align N sources by pulse id and emit each pulse once, in order.

    ``offer()`` returns the pulses that became ready because of it, as
    ``[(pulse_id, timestamp_ns, values), ...]`` in increasing pulse order -
    returning them rather than invoking a callback keeps the caller in charge
    of threading, and makes the whole thing testable without any.

    A pulse is ready when every source's newest pulse is past it (the
    watermark rule above), or when the buffer is full and it is the oldest -
    so one source falling silent costs *depth* pulses of latency, not an
    unbounded stall.
    """

    def __init__(self, n_sources, depth=1000, timeout_pulses=None):
        """*timeout_pulses*, if given, also releases a pulse once that many
        newer pulses have arrived, whether or not every source has moved past
        it. The watermark rule is about correctness (a source with nothing to
        say must not hold a pulse back); this is about bounded latency when a
        source goes quiet, and it is what callers who set it small are asking
        for - ``timeout_pulses=1`` means "forward partials at once".
        """
        if n_sources < 1:
            raise ValueError("n_sources must be >= 1")
        self.n_sources = int(n_sources)
        self.depth = int(depth)
        self.timeout_pulses = None if timeout_pulses is None else int(timeout_pulses)
        self.n_stale = 0
        self._slots = {}                       # pulse_id -> [ts, values]
        self._newest = [None] * self.n_sources  # per-source watermark
        self._emitted_through = None            # highest pulse id emitted
        self._lock = threading.Lock()
        self.n_emitted = 0
        self.n_forced = 0
        self.n_late = 0
        self.n_restarts = 0

    def set_depth(self, depth):
        with self._lock:
            self.depth = max(1, int(depth))

    def offer(self, source, pulse_id, timestamp_ns, values):
        with self._lock:
            if self._emitted_through is not None and pulse_id <= self._emitted_through:
                if pulse_id < self._emitted_through - RESTART_GAP_PULSES:
                    # The sender restarted and is counting from the start
                    # again. Treating these as stragglers would discard
                    # everything from the new connection for good, which is
                    # what a reconnecting source looks like.
                    self._reset_locked()
                else:
                    # Its slot is already gone. Emit it on its own rather than
                    # dropping it: a source can legitimately be further behind
                    # than the join is holding (an unbounded skew between
                    # backends), and discarding its data would lose the source
                    # entirely. Ordering downstream is guarded by
                    # EventWorker._is_new_pulse, which exists for exactly this.
                    self.n_late += 1
                    return [(pulse_id, timestamp_ns, dict(values))]
            prev = self._newest[source]
            if prev is None or pulse_id > prev:
                self._newest[source] = pulse_id
            slot = self._slots.get(pulse_id)
            if slot is None:
                self._slots[pulse_id] = [timestamp_ns, dict(values)]
            else:
                if slot[0] in (None, 0):
                    slot[0] = timestamp_ns
                slot[1].update(values)
            return self._collect()

    def _collect(self):
        ready = []
        while self._slots:
            oldest = min(self._slots)
            newest = max((w for w in self._newest if w is not None), default=oldest)
            if self._settled(oldest):
                pass
            elif (self.timeout_pulses is not None
                  and newest - oldest >= self.timeout_pulses):
                self.n_stale += 1
            elif len(self._slots) > self.depth:
                self.n_forced += 1
            else:
                break
            ts, values = self._slots.pop(oldest)
            self._emitted_through = oldest
            self.n_emitted += 1
            ready.append((oldest, ts, values))
        return ready

    def _reset_locked(self):
        """Forget the pulse-id timeline (sender restart). Counters are kept:
        they are session totals, not per-connection."""
        self._slots.clear()
        self._newest = [None] * self.n_sources
        self._emitted_through = None
        self.n_restarts += 1

    def _settled(self, pulse_id):
        """True once every source has moved past *pulse_id*.

        A source that has not sent anything at all (watermark None) holds
        everything back on purpose: it may be about to deliver this very
        pulse. The depth bound in _collect() is what keeps that from being
        forever.
        """
        for newest in self._newest:
            if newest is None or newest <= pulse_id:
                return False
        return True

    def flush(self):
        """Emit everything held, settled or not - for shutdown, or when a
        source is known to be gone for good."""
        with self._lock:
            ready = []
            for pid in sorted(self._slots):
                ts, values = self._slots[pid]
                self._emitted_through = pid
                self.n_emitted += 1
                ready.append((pid, ts, values))
            self._slots.clear()
            return ready

    def stats(self):
        with self._lock:
            return {
                "n_sources": self.n_sources,
                "depth": self.depth,
                "held": len(self._slots),
                "n_emitted": self.n_emitted,
                "n_forced": self.n_forced,
                "n_stale": self.n_stale,
                "n_late": self.n_late,
                "n_restarts": self.n_restarts,
                "timeout_pulses": self.timeout_pulses,
                "watermarks": list(self._newest),
            }
