# Live-Stream Data Acquisition

`escape.stream` provides live, pulse-by-pulse data acquisition from the
SwissFEL beamline network.  It mirrors the `escape.Array` API so that analysis
code written against stored data can be applied to a live stream with a
mechanical substitution.

---

## Architecture at a glance

```
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  DATA SOURCES                                                        ║
  ║  bsread Sender (local)   PSI Dispatcher   Redis/Dragonfly           ║
  ╚═══════════╤══════════════════╤══════════════════╤═════════════════════╝
              │                  │                  │
  ╔═══════════▼══════════════════▼══════════════════▼═════════════════════╗
  ║  EVENT HANDLERS  (swap one keyword to change backend)                 ║
  ║  DataHubLocalEventHandler   DataHubEventHandler   MultiSourceHandler ║
  ╚═══════════════════════════════╤══════════════════════════════════════╝
                                  │
  ╔═══════════════════════════════▼══════════════════════════════════════╗
  ║  EventWorker  ─  one background thread, calls all registered        ║
  ║                  callbacks on each pulse                             ║
  ╚═══════════════════════════════╤══════════════════════════════════════╝
                                  │  per-pulse dispatch
        ┌─────────────────────────┼────────────────────────┐
        ▼                         ▼                        ▼
   EventSource               ProcSource            FilteredEventSource
   ('i0', ew)                (i / i0)              (i, mask=pump)
        │                         │                        │
        ▼                         ▼                        ▼
  ┌──────────┐           ┌──────────────┐         ┌──────────────┐
  │ Stream   │           │ Stream       │         │ Stream       │
  │ i0       │           │ ratio        │         │ i_on         │
  └────┬─────┘           └───────┬──────┘         └──────┬───────┘
       │                         │                        │
       ▼                         ▼                        ▼
  ValueHistPlot            Plot (median)           PlotCorrelation
  plot_hist()              plot_med()              plot_corr()
```

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

# Stop a live plot
hp.stop()
```

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

# Tied to a figure window — stops when the plot is closed
fig, ax = plt.subplots()
ctx = StreamContext(i0)
ctx.tie_to_figure(fig)
ctx.start()
i0.plot_hist(axes=ax, update=0.5)
```

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

For the full architecture diagram and design proposals see the
[stream design review](https://claude.ai/code/artifact/cd017151-c247-442e-a358-79066e3283a2).

---

## Developing against real data: `dev_proxy`

`escape/stream/dev_proxy.py` is a dev-only helper (not part of the public
API) for debugging `escape.stream` against **real** SwissFEL channels from a
machine that isn't itself on the PSI network, given a SOCKS-mode SSH tunnel
into one that is:

```bash
ssh -D 8787 -N your_user@saresb-cons-05
```

```python
from escape.stream.dev_proxy import enable_socks_proxy
enable_socks_proxy(port=8787)   # routes both the dispatcher HTTP lookup and
                                 # the raw bsread/mflow ZMQ data socket

from escape.stream import EventWorker, DataHubEventHandler
ew = EventWorker(DataHubEventHandler(backend='bsread'))
ew.registerSource('SAROP21-PBPS103:INTENSITY')
```

Call `disable_socks_proxy()` before going back to local-only testing with
`TestStream`/`LocalEventHandler` — while enabled, "localhost" means the
tunnel endpoint's loopback, not this machine's. See the module docstring for
why two separate mechanisms are needed (an HTTP(S) proxy env var for the
dispatcher lookup via `requests`, plus a `zmq.Context.socket` patch for the
raw data socket, since ZMQ bypasses Python's own socket/HTTP stack).

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
Live plots run a timer-based refresh loop and do not own the acquisition.
Closing a plot window does not discard data.
