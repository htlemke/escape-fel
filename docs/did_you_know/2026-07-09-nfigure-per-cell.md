---
added: 2026-07-09
tags: [plotting, jupyter]
details: user_guide/plot_tools
---

# Re-running a cell replaces its figure instead of piling up new ones

`nfigure()`, `nsubplots()` and `nsubplot_mosaic()` name the figure after the
notebook cell that creates it. Re-running the cell starts from a clean
figure; other cells keep theirs.

```python
fig, axes = nsubplots(1, 2)
```

More: {doc}`../user_guide/plot_tools`
