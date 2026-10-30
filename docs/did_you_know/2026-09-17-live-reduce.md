---
added: 2026-09-17
tags: [swissfel, loading]
details: user_guide/swissfel
---

# Reduce a run while it is still being acquired

`escape.swissfel.live_reduce` follows a running scan and appends the
completed steps' small channels to a results file, so the reduced data can
be opened during the run and is ready when it ends.

```bash
python -m escape.swissfel.live_reduce --run-number 42 --pgroup p12345
```

More: {doc}`../user_guide/swissfel`
