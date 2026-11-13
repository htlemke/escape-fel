---
added: 2026-11-13
tags: [storage]
details: user_guide/dataset
---

# Store thousands of small channels several times faster

When storing many `ArrayTimestamps` with a few points each, skip the
per-array flush and read-back, and flush once at the end. Identical scans are
stored once and shared, so the file is smaller too.

```python
for name, arr in channels.items():
    ds.append(arr, name=name)
    arr.store(flush=False, read_back=False)
ds.results_file.flush()
```

More: {doc}`../user_guide/dataset`
