# Toolbar button icons

`fit.svg` and `peak.svg` are the Fit/Peak toolbar buttons' icons
(`escape.plot_utilities._build_fit_icon`/`_build_peak_icon`), loaded at
runtime via `QIcon(path)`. Edit them directly in Inkscape (or any SVG
editor) and save in place -- no code change needed, the loader just reads
whatever is here. If a file is missing, or `QtSvg` isn't available, the
corresponding `_build_*_icon()` falls back to drawing the same shape
procedurally with `QPainter` instead.

Any size/viewBox works (`QIcon` scales to fit the toolbar); the originals
were exported at 256x256 from the procedural drawing code so there'd be
something to start editing from.
