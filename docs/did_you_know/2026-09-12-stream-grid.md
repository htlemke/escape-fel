---
added: 2026-09-12
tags: [stream, grid]
details: user_guide/stream
---

# A live heatmap for 2-D scans

`escape.stream.Grid` reshapes a Stream binned by two step indices into a
grid. `mean`, `std`, `median`, `sum` and the others return the grid, and
`plot=True` shows it as a live heatmap.

```python
grid = Grid(binned, shape=(10, 8))
grid.mean(plot=True)
```

More: {doc}`../user_guide/stream`
