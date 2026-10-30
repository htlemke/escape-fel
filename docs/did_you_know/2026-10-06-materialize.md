---
added: 2026-10-06
tags: [storage]
details: user_guide/dataset
---

# Keep data after closing its file

Arrays loaded from a results file read from it lazily. `materialize()` loads
the data into memory and detaches the Array, so it keeps working after the
file is closed.

```python
sig = ds.signal.materialize()
ds.close()
```

More: {doc}`../user_guide/dataset`
