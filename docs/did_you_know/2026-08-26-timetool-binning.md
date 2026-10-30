---
added: 2026-08-26
tags: [arrays, binning, timing]
details: user_guide/array_operations
---

# Timetool binning for delays that are already corrected

`tools.timetool_binning` adds each step's nominal delay to the timetool
signal by default. If your delay is already fully corrected, pass
`time_vec=None` so the nominal delay is not counted twice.

```python
sig.tools.timetool_binning(t_corrected, time_vec=None, time_bins=50e-15)
```

More: {doc}`../user_guide/array_operations`
