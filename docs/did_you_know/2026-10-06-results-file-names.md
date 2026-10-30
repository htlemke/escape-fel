---
added: 2026-10-06
tags: [storage]
details: user_guide/dataset
---

# A bare name is enough for a results file

Results-file names are completed for you: `run12` becomes `run12.esc.h5`; an
incomplete suffix such as `.h5` is completed too, with a warning.
`ds.results_filepath` shows the path actually used.

```python
ds = DataSet.create_with_new_result_file("run12")
```

More: {doc}`../user_guide/dataset`
