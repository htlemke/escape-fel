---
added: 2026-08-19
tags: [plotting, jupyter]
details: user_guide/plot_tools
---

# Keep a figure in a side panel while you scroll

In JupyterLab, `detached=True` shows a figure or viewer in a panel beside
the notebook. Re-running the cell replaces that panel instead of opening
another one. Needs `pip install escape-fel[sidecar]`.

```python
fig = nfigure(detached=True, title="delay scan")
StackViewer(images, detached=True)
```

More: {doc}`../user_guide/plot_tools`
