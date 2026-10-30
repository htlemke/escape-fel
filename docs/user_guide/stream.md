# Live-Stream Data Acquisition

`escape.stream` provides live, pulse-by-pulse data acquisition from the
SwissFEL beamline network.  It mirrors the `escape.Array` API so that analysis
code written against stored data can be applied to a live stream with a
mechanical substitution.

---

## How it fits together

An **EventWorker** receives one pulse-aligned stream of all requested
channels in a background thread, through an **event handler** that selects
the transport (dispatcher, local bsread sender, Redis). Every pulse, each
accumulating **Stream** stores its value for that pulse; derived Streams
(`i / i0`, `i[pump]`, `evtset[25]`, ...) compute theirs from the Streams
they are built from. Live plots redraw from the accumulated data.

You never subscribe channels yourself: accumulating or plotting a Stream
subscribes every channel it depends on. How this works internally
(subscriptions, restarts, threads, failure handling) is described in
[Stream internals](../development/stream_architecture.md).

---

## API mirror — Array and Stream

| Operation | `escape.Array` (stored) | `escape.stream.Stream` (live) |
|---|---|---|
| Create | `Array('ch', scan)` | `Stream('ch', ew)` |
| Filter | `arr[bool_mask]` | `stream[mask_stream]` |
| Invert | `~arr` | `~stream` (logical NOT) |
| Arithmetic | `a / b`, `a + b`, … | same — per-event via ProcSource |
| Bin by key | `arr.digitize(bins)` | `stream.digitize(bins)` |
| Categorise | `.categorize(other)` | `.categorize(other)` |
| Access data | `arr.data`, `arr.index` | `stream[-100:]` (last N events) |
| Plot | `arr.plot()`, `.plot_hist()` | `stream.plot_med()`, `.plot_hist()`, `.plot_corr()` |
| Acquisition | N/A — static | `stream.accumulate(True/False)` |
| Lifetime | N/A | `StreamContext(s1, s2).tie_to_figure(fig)` |

---

## Quick start

### Synthetic test data (offline, no `psi-datahub` needed)

```python
from escape.stream import (
    Stream, EventWorker, TestStream, LocalEventHandler,
)
import numpy as np

ts = TestStream(port=9999, interval=0.02)   # ~50 Hz synthetic stream
ts.start()

ew = EventWorker(LocalEventHandler(host='localhost', port=9999))

i0   = Stream('i0',      ew, unit='a.u.')
i    = Stream('i',       ew, unit='a.u.')
t    = Stream('t',       ew, unit='ps')
pump = Stream('pump_on', ew, unit='bool')

for s in [i0, i, t, pump]:
    s.accumulate(True)
```

`LocalEventHandler` only needs `bsread` and is the handler used by
`escape/stream/example_local_stream.ipynb`.  `DataHubLocalEventHandler` is the
`psi-datahub`-backed equivalent (see below) — same interface, but future-proof
against the bsread → Redis/Dragonfly transport migration.

### Live SwissFEL data (dispatcher / Redis)

```python
from escape.stream import Stream, EventWorker, DataHubEventHandler

# bsread via SwissFEL dispatcher (current production backend)
ew = EventWorker(DataHubEventHandler(backend='bsread'))

# Redis/Dragonfly (next-generation backend)
# ew = EventWorker(DataHubEventHandler(backend='redis'))

i0     = Stream('SARBD02-DBPM070:Q1',   ew)
energy = Stream('SARUN18-UIND030:FELPHOTENE', ew)
i0.accumulate(True)
```

### Merging channels from two backends

```python
from escape.stream import MultiSourceEventHandler

ew = EventWorker(MultiSourceEventHandler(
    DataHubLocalEventHandler(host='localhost', port=9999),  # local channels
    DataHubEventHandler(backend='bsread'),                  # live channels
    timeout_pulses=50,   # forward partial events after 50 missed pulse IDs
))
```

---

## Derived quantities and filtering

```python
# Per-shot ratio (new Stream backed by ProcSource)
ratio = i / i0

# Boolean filter: pump-on shots only
i_on  = i[pump]
i_off = i[~pump]     # ~pump = logical NOT

# Accumulate all
for s in [ratio, i_on, i_off]:
    s.accumulate(True)
```

### Channels at different rates

All channels of one EventWorker arrive in one pulse-aligned stream. On a
pulse where a channel has no data (a detector running at a lower rate, or
not running at all), its value is `None`. Derived Streams follow one rule:

- **A derived value exists only on pulses where all its inputs exist.**
  `fast - slow` has a sample only on `slow`'s pulses, and `i[evtset[71]]`
  only on pulses where the event set was received *and* bit 71 is set. A
  lower-rate input never repeats an old value onto later pulses.
- **Running statistics are held.** `slow.running_mean(50)` keeps its last
  value on pulses without new `slow` data, so `fast / slow.running_mean(50)`
  has a value on every `fast` pulse.
- A scan parameter that is `None` on a pulse skips that pulse:
  `slow.digitize(bins).categorize(fast)` only bins `fast` on `slow`'s pulses.
  The parameter's channel is subscribed automatically with the binned Stream.

A requested channel that has no data at all (e.g. a stopped detector) does
not stop the other channels; its Streams just get no data, and a plot's
timeout says so (`never had data`).

### Detector channels that arrive late

Some channels reach the dispatcher seconds after their pulse, e.g.
Jungfrau ROI channels (about 12 s). escape notices this by itself and puts
such channels on a separate dispatcher connection:

- Plots of prompt channels stay live; plots of the late channel run that
  much behind.
- Anything combining both (`jf[0][evtset[71]]`, `jf[0] / i0`) appears when
  the late value arrives, computed with the prompt values of the same
  pulse.
- The first time a late channel is used in a session, prompt channels pause
  for about 15 s while this is detected. After that, escape remembers the
  delay and separates the channel right away.

### Shared Streams, shared channels

- `stream[i]` and `stream[mask]` return the **same** Stream when asked again
  with the same index or mask Stream, including what it has already
  accumulated, so a second `.plot()` starts immediately.
- Accumulation is held **per consumer**. Each plot, `StreamContext`, `with`
  block or `gather()` call is its own consumer; a plain `accumulate(True)`
  is yours. The Stream stops only when the last consumer is gone, so
  closing one plot does not stop another plot of the same Stream.
  `accumulate(False)` without arguments releases all consumers;
  `stream.consumers` lists them.
- Stopping a Stream never unsubscribes a channel that another Stream, plot
  or eco counter still uses.

### When a plot times out

Adding a channel reconnects the stream. Plots wait for that reconnect
before their `timeout` starts. If no data arrives in time, the
`TimeoutError` names the real channels and the state of the event loop:

```
TimeoutError: no data for 'JF03T01V02:roi_intensities[0]' (channels: JF03T01V02:roi_intensities) after 20 s.
  event loop: alive, last event 0.01 s ago (pulse 28610363873), 100 Hz, 2 callbacks, loop 0.0 ms/event
  restart: finished 20.0 s ago
  JF03T01V02:roi_intensities: data in 0 of the last 100 events, never had data
```

`ew.stats()` returns the same numbers as a dict. A lost connection is
re-established automatically. If live data has stopped anyway, restart the
connection by hand; existing Streams and plots carry on:

```python
from escape.stream import reset_default_eventworker
reset_default_eventworker()            # or ew.reset() for your own worker
reset_default_eventworker(hard=True)   # if that doesn't help
```

---

## Value-range filters with tunable limits

`stream.filter(lo, hi)` keeps events where `lo <= stream <= hi` — the same call
as `Array.filter`. The limits are labeled
{class}`~escape.storage.lineage.Param` objects (`"i0 min"`, `"i0 max"`) that are
read **per event**: change one and it applies **from the next pulse on**;
events already accumulated stay as they are.

```python
from escape import live

i0_ok = i0.filter(0.5, 2.0)
ratio = i / i0_ok               # only emits where i0 passes
lo, hi = i0_ok.params           # the two Params
lo.value = 0.8                  # takes effect with the next pulse

live.panel(ratio)               # input fields for every Param upstream of ratio
ratio.plot_med(params="all")    # or: a plot with those fields below it
```

`stream.filter_interactive(0.5, 2.0)` shows a live histogram of the stream
(its rolling buffer of recent events, like `plot_hist`) with a draggable span
bound to the limits, and returns the filtered Stream (the selector is its
`.tool`). It needs an interactive matplotlib backend (`%matplotlib widget`).

To discard what was collected under the old limits whenever they change, use
`on_change="reset"` (or `stream.reset_on_change()` on any derived Stream);
`stream.clear()` discards the accumulated data once.

### Tunable bins

`stream.digitize(bins)` keeps its bin edges as a `Param` too (`"t bins"`, also
`binning.bins_param`), and `stream.digitize_interactive()` (or just
`stream.digitize()` with no bins) shows the live histogram with a draggable
region and the same bin controls as for Arrays — rounded bin size (default),
bin size, or number of bins; typing an integer means *number of bins*, a
non-integer a bin size. It returns the `StreamBinning`, ready for
`.categorize(...)`:

```python
binning = t.digitize_interactive()            # drag the region, choose the bins
ratio_vs_t = binning.categorize(i / i0)
ratio_vs_t.plot_med()
```

Unlike a filter limit, the bins *are* the scan steps, so a change re-bins from
the next pulse on **and discards what was accumulated under the old bins** in
the categorized Stream and every Stream derived from it (they follow
automatically).

Offloading such a pipeline with `to_pipeline_server()` bakes in the limits'
values at that moment — a running server pipeline does not follow later changes.

---

## Selecting one element of an array-valued channel

Some channels are arrays rather than scalars — e.g. `SAR-CVME-TIFALL5:EvtSet`,
a 256-element boolean event-code set. `stream[i]` (or `stream.element(i)`)
returns a new *live* scalar Stream of just that element, updated every
event — not a one-off snapshot. Because it's a normal Stream, it can be used
as a filter mask on another Stream directly:

```python
evtset = Stream('SAR-CVME-TIFALL5:EvtSet', ew)

laser_on = evtset[25]           # live Stream of just bit 25
i0_laser_on = i0[laser_on]      # downselect another Stream by it

i0_laser_on.accumulate(True)    # auto-subscribes both real channels --
                                 # evtset and i0 -- no separate accumulate()
                                 # calls on evtset/laser_on needed
```

Accumulating any derived Stream (arithmetic, `filter()`/`[mask]`, `element()`/
`[i]`, `categorize()`, ...) transitively subscribes every real channel the
computation depends on, however deeply nested.

---

## Running statistics

`Stream.running_mean()` / `running_std()` / `running_median()` / `running_mad()`
(plus `running_nanmean()` / `running_nanstd()` / `running_nanmedian()` /
`running_nanmad()`) return a new live Stream: the current windowed statistic,
recomputed every event, over the last `N_acc` samples. Works for scalar
*or* array-valued channels — an array-valued channel's running mean is itself
an array of the same shape (reduced only over the event/window axis), e.g.
a running-averaged waveform or the per-bit "on fraction" of a boolean
event-code-set array.

```python
rm  = i0.running_mean(N_acc=200)     # windowed mean, last 200 events
rs  = i0.running_std(N_acc=200)
rmed = i0.running_median(N_acc=200)
rmad = i0.running_mad(N_acc=200)     # median absolute deviation, unscaled

rm.accumulate(True)
```

**`N_acc` can be changed at any time, mid-run**, without recreating or
interrupting the stream — it's a plain, freely reassignable attribute on the
*returned* Stream, read fresh on every event:

```python
rm.N_acc = 50   # shrinks (or grows) the window from the next event onward
```

**Weighting** by another live Stream (`sum(w*x)/sum(w)` for mean/std; a
cumulative-weight-crossing weighted median for median/mad) — accumulating the
result auto-subscribes both channels' dependencies, per the previous section:

```python
rmw = i0.running_mean(N_acc=200, weights=pump_intensity)
```

The `nan*` variants ignore NaN samples — for a weighted stat, a pair is
skipped if *either* the value or its weight is NaN.

For a running statistic not covered here (e.g. a true cumulative all-time
average, or an exponential moving average), the same underlying pattern
these are built on is available directly: wrap a small **stateful callable**
with `wrapFunc_singleOutput` (the same machinery `escaped()`-style functions
use in `escape.storage`). The callable's own instance state persists across
calls, since `wrapFunc_singleOutput` gives back one derived Stream driven by
one shared callable instance — not a fresh one per event — and `ProcObj`
dedups by pulse ID, so it runs exactly once per real event:

```python
from escape.stream import wrapFunc_singleOutput

class ExpMovingAvg:
    """Exponential moving average with smoothing factor alpha."""
    def __init__(self, alpha=0.1):
        self.alpha, self.value = alpha, None
    def __call__(self, value):
        self.value = value if self.value is None else \
            self.alpha * value + (1 - self.alpha) * self.value
        return self.value

i0_ema = wrapFunc_singleOutput(ExpMovingAvg(0.05), name='i0_ema', unit=i0.unit)(i0)
i0_ema.accumulate(True)
```

`wrapFunc_singleOutput` (like `running_*()`) always uses a fresh no-scan
`Scan()` for the result — one statistic over the whole accumulation, not per
scan step; pass `scan=i0.scan` inside a custom `ProcObj(...)` call directly
instead for one running statistic *per scan step*.

---

## Binning by a scan parameter

```python
import numpy as np

bins = np.linspace(-1.0, 1.0, 21)   # 20 delay bins

# ratio sorted into delay bins — mirrors Array.digitize().categorize()
ratio_vs_t = t.digitize(bins).categorize(ratio)
ratio_vs_t.accumulate(True)
```

---

## Live plots (ipympl / %matplotlib widget)

```python
# Value-distribution histogram (no scan)
hp = i0.plot_hist(update=0.5, n_bins=40)

# Median vs. scan parameter
mp = ratio_vs_t.plot_med(update=0.5)

# Scatter correlation, defaults to lab_time on the x-axis if omitted (see below)
cp = i.plot_corr(i0, Npoints=400, update=0.5)

# Stop redrawing (the Stream keeps accumulating)
hp.stop()
```

Closing a plot window stops the plot and releases its share of the
acquisition (see "Shared Streams, shared channels" above).

### `pulse_id` / `lab_time` — always-available pseudo-channels

Every event already carries a pulse ID and a wall-clock timestamp, regardless
of which real channels were requested — `EventWorker.pulse_id` /
`.lab_time` (cached per worker) save typing `Stream('pulse_id', ew)` by hand:

```python
pid = ew.pulse_id     # or: from escape.stream import pulse_id; pulse_id(ew)
lt  = ew.lab_time      # or: lab_time(ew)  -- both fall back to the module
                        # default EventWorker if none is passed
```

`plot_corr()` uses this: with no `xVar`, it defaults to a live `lab_time`
Stream — i.e. `i0.plot_corr()` alone is a live "value vs time" trend plot.
Pass `default_x='pulse_id'` to default to pulse ID instead.

### Array-valued channels (e.g. `SAR-CVME-TIFALL5:EvtSet`)

`plot_hist()` and `plot_corr()` auto-detect array-valued data (once data is
available) and route to a 2D live image instead of a value/count histogram
or point scatter — neither of those is meaningful per array element:

```python
evtset = Stream('SAR-CVME-TIFALL5:EvtSet', ew)   # 256-element boolean array

evtset.plot_hist(N_acc=100)          # WaterfallPlot: rows = last 100 events,
                                      # columns = array index

evtset.plot_corr(i0, N_acc=100)      # array vs scalar -> same WaterfallPlot,
                                      # rows ordered by i0's live value

evtset.plot_corr(evtset2)            # array vs array, same shape -> every
                                      # element of the last Npoints matched
                                      # events pooled into one dense scatter
                                      # (there's no 2D representation of a
                                      # true element-by-element correlation)
```

For anything else — a derived per-event trace, or just "show me the current
value" — `Stream.plot(rate_Hz=1)` is the generic fallback: the latest array
snapshot as a line, or a rolling trend for a scalar Stream, redrawn at
`rate_Hz`. Ignores scan structure entirely, so it works on any derived
Stream, including chained ones:

```python
isref = evtset[25]
(array[~isref] / array[isref].running_mean(N_acc=50)).plot(rate_Hz=1)
```

---

## Acquisition lifetime — StreamContext

```python
from escape.stream import StreamContext

# Timed block
with StreamContext(i0, ratio):
    time.sleep(10)   # accumulate for 10 s

# Tied to a figure window: releases when the window is closed
fig, ax = plt.subplots()
ctx = StreamContext(i0, ratio)
ctx.tie_to_figure(fig)
ctx.start()
```

A `StreamContext` is one consumer of its Streams, like a plot: stopping it
does not stop a plot of the same Stream.

---

## Backend selection guide

| Handler | Transport | Notes |
|---|---|---|
| `DataHubLocalEventHandler(host, port)` | bsread PULL | local test streams; no restart on channel add |
| `DataHubEventHandler(backend='bsread')` | PSI dispatcher → bsread SUB | current production default |
| `DataHubEventHandler(backend='redis')` | Redis/Dragonfly | next-generation PSI backend |
| `DataHubEventHandler(backend='auto')` | redis if available, else bsread | auto-select |
| `MultiSourceEventHandler(*handlers)` | any combination | merges N backends by pulse_id |
| `LocalEventHandler(host, port)` | bsread PULL (direct) | legacy; prefer DataHubLocalEventHandler |
| `EventHandler_SFEL()` | bsread + dispatcher | legacy; prefer DataHubEventHandler |

---

## Developing against real data

To test against real SwissFEL channels from outside the PSI network (via
Hoppla's `tools/psi-tunnel` or `escape.stream.dev_proxy`), see
[Stream internals](../development/stream_architecture.md#developing-against-real-data).

---

## Design principles

**The transport is invisible.**
Users never import `bsread` or `datahub`.  Swapping backends is a one-keyword
change to the `EventHandler` constructor; `Stream` objects always present the
same interface.

**Live and stored data share one syntax.**
`stream[pump]`, `stream.digitize(bins).categorize(ratio)`, and `stream.plot_med()`
are syntactically identical to their `escape.Array` equivalents.

**Derived quantities are first-class streams.**
`ratio = i / i0` creates a genuine `Stream` that accumulates the computed ratio
per event — not a snapshot.  Filters and scan binnings follow the same rule.

**Acquisition is explicit; plots are continuous.**
`accumulate(True/False)` is the user's gate on data collection.
Live plots run a timer-based refresh loop and hold their own share of the
acquisition: closing a plot window releases only that share (see "Shared
Streams, shared channels") and does not discard data.
