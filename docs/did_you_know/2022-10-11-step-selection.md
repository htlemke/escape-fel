---
added: 2022-10-11
tags: [swissfel, loading]
details: user_guide/swissfel
---

# Load only some steps of a long scan

`step_selection` loads a slice or a list of scan steps, for a quick first
look at a long scan.

```python
load_dataset_from_scan(run_number=42, pgroup="p12345", step_selection=slice(0, 10))
```

More: {doc}`../user_guide/swissfel`
