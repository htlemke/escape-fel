---
added: 2026-08-26
tags: [plotting, fitting]
details: user_guide/plot_tools
---

# Fit any curve on a plot without writing fit code

Figures made with `nfigure()` have a **Fit** button (Qt and `%matplotlib
widget`). Type a model such as `gaussian, linear`, pick the range on the
plot, run the fit, and copy a standalone script that repeats it.

```python
fig, ax = nsubplots()
sig.scan.plot(axis=ax)   # then click Fit in the toolbar
```

More: {doc}`../user_guide/plot_tools`
