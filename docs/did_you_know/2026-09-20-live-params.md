---
added: 2026-09-20
tags: [arrays, interactive]
details: user_guide/live
---

# Results that update while you tune a number

Wrap a value in a `Param` and pass it where an `escaped` function, filter or
digitize takes a plain value. Plots made with `live=True` redraw when it
changes, and `live.panel(result)` gives an input field for every Param the
result depends on.

```python
factor = live.Param(2.0, "factor")
scale(sig, factor).plot(live=True, params="all")
```

More: {doc}`../user_guide/live`
