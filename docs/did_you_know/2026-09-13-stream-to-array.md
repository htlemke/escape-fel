---
added: 2026-09-13
tags: [stream, arrays]
details: user_guide/stream
---

# Turn what a Stream collected into an Array

`to_array()` turns the data a Stream has accumulated into an `escape.Array`
with pulse-ID index and scan steps, ready to store, align with offline data
or analyse further.

```python
arr = ratio_vs_t.to_array()
```

More: {doc}`../user_guide/stream`
