# Installation

## From GitHub

```bash
pip install git+https://github.com/htlemke/escape-fel
```

## From a local clone

```bash
git clone https://github.com/htlemke/escape-fel
cd escape-fel
pip install -e .
```

## Dependencies

`escape` requires Python ≥ 3.8 and the following packages:

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

A complete environment specification is provided in
[`environment.yml`](https://github.com/htlemke/escape-fel/blob/master/environment.yml).

## Building this documentation locally

```bash
pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
```
