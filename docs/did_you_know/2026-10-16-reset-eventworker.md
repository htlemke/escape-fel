---
added: 2026-10-16
tags: [stream, troubleshooting]
details: user_guide/stream
---

# Restart a stalled stream without rebuilding your Streams

`reset_default_eventworker()` reconnects the stream in place: your Streams,
plots and subscriptions keep working. `hard=True` also replaces a stuck
connection.

```python
from escape.stream import reset_default_eventworker
reset_default_eventworker()
```

More: {doc}`../user_guide/stream`
