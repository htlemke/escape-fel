---
added: 2026-09-17
tags: [swissfel, loading, monitors]
details: user_guide/swissfel
---

# Every recorded EPICS channel of a run, next to its detector data

`merge_data_sources=True` adds the channels eco's namespace monitor recorded
during the run (motors, temperatures, settings) to the loaded DataSet, as
value-vs-time channels read only when used. Per-pulse data of the same name
always takes precedence.

```python
ds = load_dataset_from_scan(run_number=42, pgroup="p12345", merge_data_sources=True)
```

More: {doc}`../user_guide/swissfel`
