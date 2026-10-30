---
added: 2026-08-26
tags: [arrays, dask]
details: user_guide/array_basics
---

# Know how long compute() will take before you start it

`estimate_compute_time()` computes a few chunks spread over the Array, times
them and extrapolates to all chunks. It gives a rough figure in seconds
without paying for the full computation.

```python
imgs_roi.estimate_compute_time()
```

More: {doc}`../user_guide/array_basics`
