# Interactive Plot Tools

Figures made with `escape`'s figure helpers can carry extra toolbar buttons
for quick analysis of whatever is already plotted: fitting, peak/edge
analysis, frequency analysis, hiding curves, and a history of saved curves.
They work with the interactive matplotlib backends: Qt, and ipympl
(`%matplotlib widget`) in Jupyter. With other backends (inline, Agg) the
buttons are simply not added.

All of this lives in `escape.plot_utilities` (also reachable as
`escape.utilities.<name>`).

## Figures named after their cell

```python
from escape.plot_utilities import nfigure, nsubplots, nsubplot_mosaic

fig = nfigure()                         # figure named after this notebook cell
fig, axes = nsubplots(1, 2, figsize=(8, 3))
```

Called without `num`, these helpers name the figure after the notebook cell
that runs them. Re-running the cell closes and replaces that cell's figure
instead of leaving stale ones behind; different cells get different figures.
Pass `num=` to manage the name yourself (e.g. to reuse a figure across
cells), or `close_previous=False` to keep adding to an existing figure.

## Choosing the buttons

| Keyword (`nfigure`, `nsubplots`, `nsubplot_mosaic`) | Button | Default |
|---|---|---|
| `fit_button` | Fit — lmfit panel | on |
| `peak_button` | Peak — center/FWHM of a peak or step | off |
| `freq_button` | Freq — spectra, spectrogram, wavelets | off |
| `select_button` | Select — show/hide individual curves | off |
| `history_button` | History — save curves, recall them later | off |

```python
fig, ax = nsubplots(peak_button=True, freq_button=True, history_button=True)
```

For a figure made some other way, attach them afterwards:

```python
from escape.plot_utilities import attach_escape_buttons, attach_history_button

attach_escape_buttons(fig, fit=True, peak=True, freq=True, select=True, history=True)
attach_history_button(fig)              # or one button at a time
```

Each button acts on the axes you last clicked in (the first axes if you
haven't clicked yet). To switch the buttons off everywhere, set the
environment variable `ESCAPE_DISABLE_TOOLBAR_BUTTONS=1` before importing
escape.

The live plots of `escape.stream` get the Fit and Peak buttons automatically.

## Fit

Opens an [lmfit](https://lmfit.github.io/lmfit-py/) panel for a curve on the
axes (the last one plotted; click a curve to pick another). The model is
written as a list of components:

```text
gaussian, linear            # a peak on a sloped background
gaussian, gaussian, constant
erf, constant               # an edge / step
```

Repeated components get numbered parameter prefixes (`gaussian1_`,
`gaussian2_`). You can restrict the fit to an x-range, fix or bound
parameters (**Select range on plot**, the min/max/vary fields), check the
start values with **Guess params** and **Preview**, and **Run fit** again from
the current values. After a fit, the code tab (Qt: **Copy code to
clipboard**) gives a standalone Python script that reproduces the fit, ready
to paste into a notebook.

The panel can also be opened directly:

```python
from escape.fit_gui import AxesFitter

fitter = AxesFitter(ax, model_expr="gaussian, linear")
fitter.run_fit()
fitter.engine.result          # lmfit ModelResult
print(fitter.engine.get_code())
```

## Peak

A quick, model-free look at a peak, a dip or a step:

- **First click** draws the analysis on the axes: center, FWHM, the
  width-defining crossing points, the subtracted background (peak) or the two
  levels (step), and a labelled peak point.
- **Second click** opens a small panel to tune it: number of background
  points at each end, background model (`linear` or `offset`), a fixed
  offset (e.g. a known dark level), and the mode (`auto`, `peak`, `step`).

For a step, the center is the 50 % crossing and the width is the FWHM of
the step's derivative (for an error-function edge, the FWHM of the
underlying Gaussian). The same analysis is available as a function:

```python
from escape.plot_utilities import find_peak

res = find_peak(x, y, mode="auto", plot=ax)   # dict with center, fwhm, is_peak, ...
```

## Freq

Frequency analysis of a curve on the axes, e.g. to find a periodic
disturbance in a time trace or an oscillation in a delay scan. Methods:

- `welch` — averaged power spectrum (default)
- `periodogram` — single power spectrum
- `lombscargle` — power spectrum for **unevenly spaced** x
- `spectrogram` — power spectrum vs. x
- `scalogram` — wavelet transform (needs `pip install escape-fel[freq]`)

As with Fit, you can restrict the x-range (**Select range on plot**) and get
the equivalent code.
Directly: `escape.freq_gui.FreqAnalyzer(ax, method="welch")`.

## Select

A checkbox per curve on the axes, with a filter box and All/None. Hidden
curves also leave the legend, and the y-range is refitted to what is
left. Useful for busy plots, e.g. one curve per run.

The button needs at least two curves to be useful, so when the figure is
created empty it appears only after you call `attach_select_button(fig)`
once the curves are drawn.

## History

Saves curves so they can be recalled onto any later plot, for example to
compare today's scan with yesterday's.

- **Save**: tick the curves of the active axes and add an optional note.
  Each save is one HDF5 file in the history folder: by default
  `~/.escape/plot_history`, or the `ESCAPE_PLOT_HISTORY_DIR` environment
  variable, or `escape.plot_history.PLOT_HISTORY_DIR = "/sf/.../res/plot_history"`.
  A shared folder lets colleagues recall each other's curves; every entry
  records who saved it.
- **Recall**: saved curves are listed with the best matches to the current
  plot first (titles, axis labels, overlapping x-range, Array name, scan
  parameter, run). Search by words or date range, then add a curve in its
  original style or as a faded reference with the save time in its label.

Error-band and error-bar plots are saved and redrawn as one item.
`Array.plot()` and `Scan.plot()` store the Array name, scan parameter and run
with each curve; add your own information with
`escape.plot_utilities.tag(artists, run=142, sample="A")`.

The functions behind the dialog are in `escape.plot_history` (`save_items`,
`list_entries`, `recall`).

## Figures in a side panel

In JupyterLab, a figure or viewer can be shown in a separate panel beside
the notebook instead of inline, so it stays visible while you scroll:

```python
fig = nfigure(detached=True, title="delay scan")
viewer = StackViewer(images, detached=True)
```

Re-running the cell replaces that panel instead of opening another one.
Supported by `nfigure`/`nsubplots`/`nsubplot_mosaic` and by `StackViewer`,
`StepViewer`, `StepViewerP` and `MultipleRoiSelector`. Needs the `sidecar`
package (`pip install escape-fel[sidecar]`).

## Lines with error bands

```python
from escape.plot_utilities import errortube

errortube(x, y, yerr)                 # line with a shaded ±yerr band
errortube(x, y, (lower, upper))       # asymmetric band
errortube(bin_centers, y, yerr, step="mid")   # histogram-style steps
```

`step` takes `"pre"`, `"post"` or `"mid"` (as in `plt.step`); `"mid"` is the
usual choice when x holds bin centers. The line and its band are one item for
the Select and History tools.
