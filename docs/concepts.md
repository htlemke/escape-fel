# Core Concepts

## Event-Based Data

At a Free-Electron Laser facility (or any pulsed-source experiment), instruments
record one measurement per X-ray pulse.  Each pulse is identified by a unique
integer **pulse ID** (sometimes called *event ID* or *index*).  Across many
diagnostic instruments the pulse IDs are the same physical quantity but the
instruments may miss individual shots: a camera may deliver data for pulses
1, 2, 4, 5 while a fast digitiser delivers 1, 2, 3, 4, 5.

`escape` represents each measured channel as an {class}`~escape.Array` — a pair
of aligned arrays:

* **`data`** — the measured values, shape `(N, ...)`.
* **`index`** — the corresponding pulse IDs, shape `(N,)`.

Because index alignment is built in, arithmetic between two Arrays automatically
finds the common pulse IDs and only operates on those events.

```python
import numpy as np
import escape

a = escape.Array(data=np.array([8, 4, 3, 7]), index=np.array([0, 1, 3, 4]))
b = escape.Array(data=np.array([2, 2, 2, 2]), index=np.array([0, 1, 2, 4]))

c = a - b
print(c.index)  # [0 1 4]  — only the common pulse IDs
print(c.data)   # [6 2 5]
```

## Scan Steps

An experiment typically consists of many **scan steps** — for example, a delay
scan where the pump-probe delay is varied between steps.  Within each step,
hundreds or thousands of pulses are recorded.

`escape` stores scan metadata inside each Array via a {class}`~escape.Scan`
object (accessible as `array.scan`).  The scan partitions the event stream into
sequential groups and carries the parameter value for each step.

```
Array
├── data   shape (N_total, ...)   ← all events stacked in step order
├── index  shape (N_total,)       ← unique integer ID per event
└── scan
    ├── step_lengths  [n0, n1, n2, …]   ← events per step
    └── parameter     {"delay_s": {"values": [t0, t1, t2, …]}}
```

Individual steps are accessed as ordinary Arrays via `scan[i]`, and per-step
statistics are available directly on the scan object:

```python
means = array.scan.nanmean(axis=0)  # list, one entry per step
```

## Lazy Evaluation with Dask

Raw FEL data is often too large to hold in memory.  `escape` uses
[dask](https://docs.dask.org) arrays as its data backend.  All arithmetic and
reduction operations build a lazy graph — no computation is triggered until you
explicitly call {meth}`~escape.Array.compute` or
{func}`~escape.storage.store`.

```python
result = (sig / i0).nanmean(axis=0)  # still lazy
result = result.compute()             # trigger computation
```

For dask-backed Arrays, {meth}`~escape.Array.map_index_blocks` lets you apply
an arbitrary NumPy function chunk-by-chunk, keeping computation distributed.

## Index Ordering Convention

When two Arrays are combined with an operator, the **first** argument determines
the output index order and step grouping.  The second argument is re-indexed to
match.  This means:

```python
result_ab = a + b   # order from a
result_ba = b + a   # order from b
# result_ab.data may differ from result_ba.data in element order!
```

This convention is consistent across all binary operators, {func}`~escape.escaped`-
decorated functions, and {meth}`~escape.Array.categorize`.
