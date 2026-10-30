---
added: 2026-07-09
tags: [testing]
details: api/example_data
---

# Try any feature without beamline data

`escape.storage.example_data` generates realistic, seedable FEL data: scans,
pump-probe sets, image scans, grid scans and photon-counting detector
stacks.

```python
from escape.storage.example_data import make_pump_probe_scan
sig, i0, pump_on, delay = make_pump_probe_scan(n_steps=20)
```

More: {doc}`../api/example_data`
