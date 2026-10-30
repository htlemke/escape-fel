---
added: 2026-09-12
tags: [plotting, analysis]
details: user_guide/plot_tools
---

# Center and FWHM of a peak or an edge in one click

The **Peak** button draws the center, FWHM and background of the curve on
the axes; a second click opens a panel to set the mode (peak, step, auto)
and background. For an edge, the width is the FWHM of its derivative.

```python
fig = nfigure(peak_button=True)
find_peak(x, y, plot=True)     # the same, as a function
```

More: {doc}`../user_guide/plot_tools`
