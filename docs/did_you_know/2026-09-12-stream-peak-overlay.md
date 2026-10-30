---
added: 2026-09-12
tags: [stream, plotting]
details: user_guide/stream
---

# Live scan plots show peak center and FWHM

`plot_med()` draws the center and FWHM (or edge position and width) on the
live median curve, and its figure has the Fit and Peak buttons. Pass
`peak_overlay=False` for channels without a peak or edge.

```python
ratio_vs_t.plot_med()
```

More: {doc}`../user_guide/stream`
