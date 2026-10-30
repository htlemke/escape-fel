---
added: 2026-09-08
tags: [stream]
details: user_guide/stream
---

# Pick one element of an array channel

For array-valued channels such as an event code set or ROI intensities,
`stream[3]` gives the Stream of one element. Plotting the whole array shows
a live waterfall.

```python
evtset = Stream("SAR-CVME-TIFALL5:EvtSet", ew)
i_on = i[evtset[71]]
```

More: {doc}`../user_guide/stream`
