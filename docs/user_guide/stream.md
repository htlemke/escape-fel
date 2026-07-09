# Live-Stream Data Acquisition

`escape.stream_new` provides live, pulse-by-pulse data acquisition from the
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

| Operation | `escape.Array` (stored) | `escape.stream_new.Stream` (live) |
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

### Synthetic test data (offline)

```python
from escape.stream_new import (
    Stream, EventWorker, TestStream, DataHubLocalEventHandler,
)
import numpy as np

ts = TestStream(port=9999, interval=0.02)   # ~50 Hz synthetic stream
ts.start()

ew = EventWorker(DataHubLocalEventHandler(host='localhost', port=9999))

i0   = Stream('i0',      ew, unit='a.u.')
i    = Stream('i',       ew, unit='a.u.')
t    = Stream('t',       ew, unit='ps')
pump = Stream('pump_on', ew, unit='bool')

for s in [i0, i, t, pump]:
    s.accumulate(True)
```

### Live SwissFEL data (dispatcher / Redis)

```python
from escape.stream_new import Stream, EventWorker, DataHubEventHandler

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
from escape.stream_new import MultiSourceEventHandler

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

# Scatter correlation
cp = i.plot_corr(i0, Npoints=400, update=0.5)

# Stop a live plot
hp.stop()
```

---

## Acquisition lifetime — StreamContext

```python
from escape.stream_new import StreamContext

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
[stream_new design review](https://claude.ai/code/artifact/cd017151-c247-442e-a358-79066e3283a2).

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
