---
added: 2026-09-09
tags: [stream]
details: user_guide/stream
---

# Run a Stream's computation on the pipeline server

`to_pipeline_server()` generates the processing script for a derived Stream,
deploys it on cam_server and publishes the result as a new bsread stream.
Needs the `cam_server` package.

```python
ratio.to_pipeline_server("bernina_ratio")
```

More: {doc}`../user_guide/stream`
