---
added: 2026-08-19
tags: [arrays, plotting]
details: user_guide/array_basics
---

# Arrays know their channel name, and can carry a unit

Arrays from `load_dataset_from_scan` are named after their channel. Set
`unit` and `Array.plot()` and `hist()` label the axis as `name (unit)`;
`concatenate` keeps both.

```python
sig.unit = "counts"
sig.plot()
```

More: {doc}`../user_guide/array_basics`
