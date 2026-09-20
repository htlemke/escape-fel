# Live results

Some quantities are best tuned by looking at the result: a filter window on
i0, the bins of a delay scan, a scale factor. `escape` lets an Array remember
**how it was made and from which tunable parameters**, so a plot of the result
can update while you drag a span on a histogram or edit a number.

Opt-in and notebook-oriented (ipympl + ipywidgets): `from escape import live`.

## The idea

- A {class}`~escape.storage.lineage.Param` is a labeled, changeable value.
- Any operation that involves a Param — or an Array that already carries
  lineage — records *which function was called on which inputs*. The Array's
  data is exactly what it always was; `array.lineage` is just extra
  bookkeeping.
- `array.evaluate()` replays the recorded calls with the **current** Param
  values, recomputing only what changed. Plots use this to redraw.
- Recording is free unless used: chains without Params carry no lineage.

```python
%matplotlib widget
from escape import live

res = sig / i0.filter_interactive(0.5, 2)   # histogram of i0 with draggable limits
res.plot(live=True)                         # redraws when the limits change
live.panel(res)                             # input fields for every Param upstream of res
```

`filter`/`digitize` turn plain numbers into labeled Params automatically
(`"i0 min"`, `"i0 max"`, `"i0 bins"`); `res.params` lists everything upstream,
`live.describe(res)` shows the recorded chain.

## Which end you tune from

Both directions work because the Params are shared objects:

- **From the origin:** open a tool on any input (`i0.filter_interactive()`);
  everything derived from it follows.
- **From the result:** `live.panel(res)` (or `res.plot(live=True, params="all")`)
  lists every Param upstream of the result as an input field — pick the one
  to tune.

## Your own functions

Pass a `Param` where the function takes a plain value; the function receives
the plain value. Any `escaped` function, `map_index_blocks` call or reduction
works the same way:

```python
@escape.escaped
def scale(x, factor):
    return x * factor

factor = live.Param(2.0, "factor")            # bounds=(0, 5) gives a slider
scale(sig, factor).plot(live=True, params="all")   # plot + 'factor' field
```

`live.live_plot(res, draw)` makes any drawing function live:
`draw(ax, current_array)` is called on a cleared axes after every change.
Several Params can be set together with `with live.batch(): ...` (one update).

## Switching it off

`live.set_enabled(False)` (or `ESCAPE_LINEAGE=0` in the environment before
importing escape) stops all recording. Params passed to operations still work
— they are replaced by their values — results just aren't live.

## Limits of this first version

- Only results that are Arrays carry lineage. Reductions to plain numbers
  (`nanmean()` over events) don't; plot those with
  `live.live_plot(res, lambda ax, a: ax.plot(a.nanmean()))`.
- A recorded chain keeps its intermediate results in memory — meant for data
  that fits. Large dask data works (the steps are replayed lazily) but each
  filter/digitize computes eagerly, as it always has.
- Live plots clear and redraw the axes (zoom is reset).
- The digitize tool writes its bins to the Param but doesn't follow external
  changes of it.
