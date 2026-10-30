---
added: 2026-10-06
tags: [storage]
details: user_guide/dataset
---

# Repair a results file after an interrupted write

If an array in a results file can no longer be read because a store was
interrupted, `check_result_file` lists the state of every array, and with
`repair=True` removes the incomplete parts so the rest loads again.

```python
from escape.storage.dataset import check_result_file
check_result_file("run0042.esc.h5", repair=True)
```

More: {doc}`../user_guide/dataset`
