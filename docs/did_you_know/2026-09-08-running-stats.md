---
added: 2026-09-08
tags: [stream]
details: user_guide/stream
---

# Running mean, median and MAD over the last N pulses

`running_mean`, `running_median`, `running_std`, `running_mad` (and `nan`
versions, optionally weighted) give a Stream of the statistic over the last
`N_acc` pulses, e.g. to normalise by a slowly varying reference.

```python
(i / i0.running_mean(N_acc=100)).plot()
```

More: {doc}`../user_guide/stream`
