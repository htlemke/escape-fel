---
added: 2026-09-19
tags: [arrays, interactive, binning]
details: user_guide/array_operations
---

# Choose bins by eye

`digitize_interactive()` shows a histogram; drag the region to bin and set a
bin size (rounded to a grid, or exact) or a number of bins. The bin edges
are drawn on the histogram, and live plots follow your changes.

```python
binned = delay.digitize_interactive(n_bins=20)
binned.scan.plot(live=True)
```

More: {doc}`../user_guide/array_operations`
