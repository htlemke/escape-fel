# escape.stream_new — backend guide

The `escape.stream_new` module separates data transport from analysis logic.
A **`EventHandler`** encapsulates the backend connection; swapping it is the
only change needed when PSI migrates infrastructure.

---

## Quick-start by use case

### a) Synthetic test data (works fully offline)

```python
from escape.stream_new import (
    Stream, EventWorker, TestStream,
    LocalEventHandler,           # bsread-direct (always available)
    DataHubLocalEventHandler,    # datahub wrapper (needs psi-datahub)
)

ts = TestStream(port=9999, interval=0.01)   # ~100 Hz
ts.start()

# Option 1 — original bsread handler (no extra dependency)
ew = EventWorker(LocalEventHandler(host='localhost', port=9999))

# Option 2 — datahub wrapper (same result, future-proof receiver)
ew = EventWorker(DataHubLocalEventHandler(host='localhost', port=9999))

i0 = Stream('i0', ew)
i  = Stream('i',  ew)
t  = Stream('t',  ew)

i0.accumulate(True)
```

### b) Live SwissFEL dispatcher data (current production backend)

```python
from escape.stream_new import Stream, EventWorker, DataHubEventHandler

# bsread via PSI dispatcher — same as the old EventHandler_SFEL
ew = EventWorker(DataHubEventHandler(backend='bsread'))

# Register channels and start collecting
i0     = Stream('SARBD02-DBPM070:Q1', ew)
energy = Stream('SARUN18-UIND030:FELPHOTENE', ew)
i0.accumulate(True)
```

### c) Redis/Dragonfly backend (next-generation, post-bsread)

```python
from escape.stream_new import Stream, EventWorker, DataHubEventHandler

# Exact same API, different transport
ew = EventWorker(DataHubEventHandler(backend='redis'))

i0 = Stream('SARBD02-DBPM070:Q1', ew)
i0.accumulate(True)
```

### d) Special / overridden URL

```python
# Custom dispatcher
ew = EventWorker(DataHubEventHandler(
    backend='bsread',
    url='https://dispatcher-api.psi.ch/sf-databuffer',
))

# Custom Redis server
ew = EventWorker(DataHubEventHandler(
    backend='redis',
    url='sf-daqsync-18:6379',
))

# Specific beamline host:port (bypass dispatcher)
ew = EventWorker(DataHubLocalEventHandler(host='sfbrd-bernina', port=9999))
```

### e) Auto-select (try redis, fall back to bsread)

```python
ew = EventWorker(DataHubEventHandler(backend='auto'))
```

---

### f) Merging channels from two backends

```python
from escape.stream_new import (
    Stream, EventWorker,
    DataHubLocalEventHandler, DataHubEventHandler, MultiSourceEventHandler,
)

# Merge a local test stream with live dispatcher channels
ew = EventWorker(MultiSourceEventHandler(
    DataHubLocalEventHandler(host='localhost', port=9999),   # local channels
    DataHubEventHandler(backend='bsread'),                   # live channels
    timeout_pulses=50,   # forward partial events after 50 missed pulse IDs (~0.5 s)
))

# Both sets of channels are now available on the same EventWorker:
local_ch = Stream('i0',                    ew)
live_ch  = Stream('SARBD02-DBPM070:Q1',   ew)
```

---

## Handler comparison

| Class | Backend | Requires | Notes |
|---|---|---|---|
| `LocalEventHandler` | bsread direct | `bsread` | Legacy; use for test streams |
| `EventHandler_SFEL` | bsread + dispatcher | `bsread` | Legacy; no datahub |
| `DataHubLocalEventHandler` | bsread direct | `psi-datahub` | Default for local/test streams |
| `DataHubEventHandler(backend='bsread')` | dispatcher → bsread | `psi-datahub` | Default for live SwissFEL data |
| `DataHubEventHandler(backend='redis')` | Redis/Dragonfly | `psi-datahub`, `redis` | New PSI backend |
| `DataHubEventHandler(backend='auto')` | redis or bsread | `psi-datahub` | Auto-select |
| `MultiSourceEventHandler(*handlers)` | any combination | `psi-datahub` | Merge N backends by pulse_id |

---

## Installing datahub

```bash
# conda (recommended at PSI)
conda install -c paulscherrerinstitute -c conda-forge datahub

# pip
pip install psi-datahub
```

datahub itself still depends on `bsread` for the `'bsread'` and local backends.
For Redis-only usage `bsread` is not required.

---

## Event message structure

All backends produce the same internal event interface via `DataHubEvent`:

| Field | `getFromSource(name)` | Notes |
|---|---|---|
| any bsread channel | value (float/array/…) | by channel name |
| `'pulse_id'` | integer | PSI 100 Hz pulse counter |
| `'lab_time'` | float (seconds since epoch) | from bsread global timestamp |

---

## Background: PSI infrastructure roadmap

| Era | Transport | escape handler |
|---|---|---|
| ~2015–2024 | bsread ZMQ push/pull + dispatcher | `EventHandler_SFEL`, `LocalEventHandler` |
| 2024– | bsread still active; Redis/Dragonfly being introduced | `DataHubEventHandler(backend='bsread')` or `'redis'` |
| Future | Redis/Dragonfly primary; bsread legacy | `DataHubEventHandler(backend='redis')` |

Using `datahub` means the transport can change (bsread → Redis → whatever comes
next) without any escape user-code changes — only the `backend=` keyword changes.
