---
added: 2026-10-16
tags: [stream]
details: user_guide/stream
---

# Slow channels no longer stall fast ones

Channels that reach the dispatcher seconds late, such as Jungfrau ROI
intensities, are moved to their own connection automatically, so plots of
prompt channels stay live. `diagnose()` on the EventWorker shows the groups.

```python
ew.diagnose()
```

More: {doc}`../user_guide/stream`
