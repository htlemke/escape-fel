---
added: 2026-09-17
tags: [storage]
details: user_guide/dataset
---

# Store an array again from scratch

`clear_results_file()` deletes stored data of some or all channels, so the
next `store()` starts from empty instead of stopping with a storage
conflict. It asks before deleting.

```python
ds.clear_results_file(["signal"])
```

More: {doc}`../user_guide/dataset`
