# Installation

## conda (recommended)

The easiest install, reusing packages already present in your conda environment:

```bash
conda install -c conda-forge escape-fel
```

## pip

```bash
pip install escape-fel
```

> **Note for conda users:** use `pip install --no-deps escape-fel` to avoid
> pip reinstalling packages that conda has already provided.

## Latest development version (from GitHub)

To get unreleased changes from the `main` branch:

```bash
pip install git+https://github.com/htlemke/escape-fel
```

Or clone and install in editable mode so local edits take effect immediately:

```bash
git clone https://github.com/htlemke/escape-fel
cd escape-fel
pip install -e .
```

## Dependencies

`escape` requires Python ≥ 3.9 and the following packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array mathematics |
| `dask` | Lazy, out-of-core computation |
| `h5py` | HDF5 file I/O |
| `zarr` | Alternative chunked storage |
| `hickle` | HDF5-backed serialisation of arbitrary Python objects |
| `pandas` | Tabular scan parameter display |
| `matplotlib` | Built-in plotting helpers |
| `rich` | Terminal output formatting |
| `tqdm` | Progress bars |

## Building this documentation locally

```bash
pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
```
