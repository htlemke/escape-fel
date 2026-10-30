---
added: 2026-09-12
tags: [stream]
details: user_guide/stream
---

# Combine a live stream with any eco detector

A Stream can be combined with anything that is read on demand: an eco
Detector or Adjustable, or a function without arguments. The value is read
at most every 0.1 s and reused for the pulses in between.

```python
corrected = i - some_eco_detector
```

More: {doc}`../user_guide/stream`
