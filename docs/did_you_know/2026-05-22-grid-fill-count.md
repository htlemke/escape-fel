---
added: 2026-05-22
tags: [scans, grid]
details: user_guide/grid
---

# See how far a 2-D scan got

For a multi-dimensional scan that stopped early, `grid.fill_count()` tells
you how many grid cells have data, and the grid statistics leave the missing
cells as NaN.

```python
sig.grid.fill_count()   # (filled, total, percent)
```

More: {doc}`../user_guide/grid`
