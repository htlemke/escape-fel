---
added: 2026-09-19
tags: [arrays, interactive]
details: user_guide/array_operations
---

# Pick a filter range by dragging on a histogram

`filter_interactive()` shows a histogram of the Array with a span you can
drag or type into. The filtered result follows the span, and a live plot of
anything computed from it redraws.

```python
good_i0 = i0.filter_interactive(0.5, 2)
(sig / good_i0).plot(live=True)
```

More: {doc}`../user_guide/array_operations`
