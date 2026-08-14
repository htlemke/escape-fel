# EXAFS Data Reduction

`escape.exafs` turns raw XAS data — absorption `μ(E)` versus energy — into the
EXAFS signal `χ(k)` and its Fourier transform `χ(R)`. It is a small, readable
re-implementation of the standard reduction recipe, written to be **understood
step by step** rather than used as a black box, and it adds a **dask-parallel
layer** for reducing whole stacks of spectra (one per FEL pulse or scan step)
lazily over an `escape.Array`.

```{note}
This submodule is optional and imported explicitly:

    from escape import exafs

It depends only on `numpy` and `scipy` (plus `matplotlib` for the quick-look
plots). It is adapted, with attribution, from a standalone teaching package by
the escape author; the algorithms trace back to the references
[below](#references-and-provenance).
```

```{warning}
This package **stops at `χ(R)`** — it does not fit shell parameters against
theoretical scattering paths. For quantitative distances/coordination numbers,
and for difficult data, use the field-standard tools
[Larch](https://xraypy.github.io/xraylarch/) or
[Demeter/Artemis](https://bruceravel.github.io/demeter/) and cross-check.
```

---

## What EXAFS is, in three steps

When an atom absorbs an X-ray just above an absorption edge it ejects a core
electron. If neighbouring atoms are present, the outgoing photoelectron wave
scatters off them and interferes with itself, making `μ(E)` **wiggle** by a few
percent above the edge. Those wiggles encode the distances to, and identities
of, the neighbours. Extracting them is a three-step recipe:

```{mermaid}
flowchart LR
    A["raw μ(E)"] --> B["pre_edge()<br/>normalise to edge step"]
    B --> C["autobk()<br/>spline background μ₀(k)"]
    C --> D["χ(k) = (μ − μ₀)/step"]
    D --> E["ft_windowed()<br/>k-weight + taper + FFT"]
    E --> F["χ(R) pseudo-RDF"]
```

1. **Normalise** — {func}`~escape.exafs.pre_edge` fits and subtracts a pre-edge
   line, fits a post-edge curve to get the **edge step**, and rescales `μ(E)` to
   run from ~0 to ~1 across the edge.
2. **Extract `χ(k)`** — convert energy to the photoelectron wavenumber
   `k = √(ETOK·(E−E₀))` and remove the smooth "bare-atom" background `μ₀(k)`.
   {func}`~escape.exafs.autobk` does this with a spline stiff enough (set by the
   `rbkg` length scale) to be blind to real EXAFS below `rbkg`.
3. **Fourier transform** — {func}`~escape.exafs.ft_windowed` k-weights `χ(k)`,
   tapers it with a window, and FFTs it into `χ(R)`.

```{admonition} Beginner gotcha
:class: important
Peaks in `χ(R)` sit **~0.3–0.5 Å short** of the true bond distance because of
quantum-mechanical scattering phase shifts. `χ(R)` is a *pseudo* radial
distribution — recovering real distances needs a fit against FEFF standards
(out of scope here). The R axis is labelled "not phase-corrected" for this
reason.
```

---

## Quick start (single spectrum)

```python
from escape import exafs
import os, escape.exafs

# bundled Cu K-edge foil data (see provenance below)
data = os.path.join(os.path.dirname(escape.exafs.__file__), "data", "cu_rt01.xmu")
energy, mu = exafs.read_columns(data)

pre = exafs.pre_edge(energy, mu)
print(pre.e0, pre.edge_step)                       # ~8980.5 eV, ~2.83

bkg = exafs.autobk(energy, mu, pre.e0, pre.edge_step, rbkg=1.0)
r, chir, _ = exafs.ft_windowed(bkg.k, bkg.chi, kweight=2,
                               window="hanning", kmin=3, kmax=13, dk=1.0)

exafs.plot_bkg(pre, bkg)
exafs.plot_chik(bkg.k, bkg.chi, kweight=2)
exafs.plot_chir(r, chir)         # first-shell peak ~2.25 Å (Cu–Cu 2.55 Å − phase)
```

Every function takes and returns plain NumPy arrays (or small `dataclass`
results with array attributes) — nothing is hidden in an opaque object, so you
can always inspect the intermediate results.

See the annotated notebooks
[`examples/exafs_quickstart.ipynb`](https://github.com/) and
`examples/exafs_understanding_the_steps.ipynb` (a from-scratch build of every
step), which run on the bundled Cu data.

---

## Reducing a whole run (escape + dask)

At an FEL you have many spectra — one per pulse, scan step, or repeat — stacked
along the event axis of an `escape.Array`. {func}`~escape.exafs.reduce_array`
reduces the whole stack **lazily and in parallel** via
{meth}`~escape.Array.map_index_blocks`, mirroring
{func}`escape.wavefront.propagate_array`:

```python
import numpy as np, escape
from escape import exafs

# mu_arr: escape.Array of shape (n_events, n_energy), shared `energy` axis
# (transmission data? build it with exafs.optical_density(i0, i1) first)

kgrid = exafs.common_k_grid(0, 16, 0.05)

# fix E0 / edge_step from a reference (e.g. the run average) so all events
# share one scale; refine=False keeps the per-spectrum background cheap
chi = exafs.reduce_array(mu_arr, energy, kgrid,
                         e0=8980.5, edge_step=2.83, rbkg=1.0)   # Array (n_events, n_k)

r, chir = exafs.ft_array(chi, kgrid, kweight=2, kmin=3, kmax=13)  # Array (n_events, n_r)

chik_mean = chi.mean(axis=0).compute()
chir_mean = np.abs(chir.data).mean(axis=0).compute()
```

Both results stay lazy dask-backed Arrays sharing the input's event index and
scan metadata, so they compose with the rest of `escape` (grouping by a scan
parameter, filtering, averaging) before you `.compute()`.

---

## What this package does **not** do

* **No shell fitting** against FEFF paths (`S₀²`, `σ²`, `ΔE₀`, `ΔR`) — use
  [Larch](https://xraypy.github.io/xraylarch/)/`lmfit`.
* **No raw beamline reduction** (I0/I1 → μ glitch removal, dead-time,
  self-absorption, multi-scan merging) — do that upstream. `optical_density()`
  is provided for the simple transmission `μ = −ln(I₁/I₀)` case.
* **The background is a simplified AUTOBK** — same physics, results close to
  Larch/Ifeffit for typical data, but cross-check important results.

---

## References and provenance

The `escape.exafs` code is a compact re-implementation of the standard EXAFS
reduction recipe. The algorithms are:

1. **AUTOBK spline background** — M. Newville, P. Līviņš, Y. Yacoby, J. J. Rehr
   & E. A. Stern, "Near-edge X-ray-absorption fine structure of Pb: A
   comparison of theory and experiment", *Physical Review B* **47**, 14126
   (1993). doi:[10.1103/PhysRevB.47.14126](https://doi.org/10.1103/PhysRevB.47.14126)

2. **Fourier transform to `χ(R)`** — D. E. Sayers, E. A. Stern & F. W. Lytle,
   "New Technique for Investigating Noncrystalline Structures: Fourier Analysis
   of the Extended X-Ray—Absorption Fine Structure", *Physical Review Letters*
   **27**, 1204 (1971).
   doi:[10.1103/PhysRevLett.27.1204](https://doi.org/10.1103/PhysRevLett.27.1204)

3. **`ETOK` constant and FT / normalisation conventions** — as in IFEFFIT/Larch:
   M. Newville, "IFEFFIT: interactive XAFS analysis and FEFF fitting",
   *Journal of Synchrotron Radiation* **8**, 322–324 (2001).
   doi:[10.1107/S0909049500016964](https://doi.org/10.1107/S0909049500016964);
   Larch: <https://xraypy.github.io/xraylarch/>.

**Field-standard tools** (recommended for production analysis):

4. Larch — <https://xraypy.github.io/xraylarch/>
5. Demeter/Athena/Artemis — <https://bruceravel.github.io/demeter/> (B. Ravel &
   M. Newville, *J. Synchrotron Rad.* **12**, 537 (2005),
   doi:[10.1107/S0909049505012719](https://doi.org/10.1107/S0909049505012719)).

**Demo data.** `escape/exafs/data/cu_rt01.xmu` is real Cu K-edge EXAFS on a
copper foil at room temperature (APS beamline 13ID), from the
[xraylarch](https://github.com/xraypy/xraylarch) example data set, included so
the notebooks run out of the box.

---

## API

See {doc}`../api/exafs` for the full reference.

| Function | Purpose |
|----------|---------|
| {func}`~escape.exafs.read_columns` | load a 2-column `energy μ` text file |
| {func}`~escape.exafs.pre_edge` | pre-edge line + edge-step normalisation |
| {func}`~escape.exafs.find_e0` / {func}`~escape.exafs.energy_to_k` | E₀ and E→k |
| {func}`~escape.exafs.autobk` | spline background → `χ(k)` |
| {func}`~escape.exafs.window` / {func}`~escape.exafs.ft_windowed` | `χ(k)` → `χ(R)` |
| {func}`~escape.exafs.plot_bkg` / `plot_chik` / `plot_chir` | quick-look plots |
| {func}`~escape.exafs.optical_density` | transmission `μ = −ln(I₁/I₀)` |
| {func}`~escape.exafs.reduce_array` | reduce an escape Array of spectra → `χ(k)` |
| {func}`~escape.exafs.ft_array` | FT an escape Array of `χ(k)` → `χ(R)` |
