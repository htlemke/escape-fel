---
added: 2026-10-30
tags: [modules, scans, experimental]
details: user_guide/adaptive_scan
---

# Let the data decide where to measure next (experimental)

`escape.adaptive_scan` proposes the next scan points, with exposure and
order, from the data so far: model-free refinement where the curve is least
known, or fits for steps, oscillations and 2-D peaks, with motor travel and
backlash costs taken into account. Experimental, for testing.

```python
from escape import adaptive_scan as ads
```

More: {doc}`../user_guide/adaptive_scan`
