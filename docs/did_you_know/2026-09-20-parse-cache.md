---
added: 2026-09-20
tags: [swissfel, loading, performance]
details: user_guide/swissfel
---

# Reloading a run is about ten times faster

`load_dataset_from_scan` stores the scanned file structure and reuses it on
the next load, by default in the run's `aux/` folder if the DAQ wrote one,
else in the pgroup's work directory. No setting needed; a 121-step run went
from 35 s to 3.5 s.

```python
ds = load_dataset_from_scan(run_number=42, pgroup="p12345")
```

More: {doc}`../user_guide/swissfel`
