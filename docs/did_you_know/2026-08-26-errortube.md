---
added: 2026-08-26
tags: [plotting]
details: user_guide/plot_tools
---

# A line with a shaded error band in one call

`errortube` draws a line with a ±error band (symmetric or `(lower, upper)`).
With `step="mid"` it draws histogram-style steps with a stepped band, the
natural look for binned data.

```python
errortube(x, y, yerr)
errortube(bin_centers, y, yerr, step="mid")
```

More: {doc}`../user_guide/plot_tools`
