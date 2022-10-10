#!/usr/bin/env python

import setuptools

VERSION = (0, 0, 1)
VERSION_STR = ".".join([str(x) for x in VERSION])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="escape-fel",
    version=VERSION_STR,
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HLemke",
    author_email="htlemke@gmail.com",
    url="https://github.com/htlemke/escape",
    packages=setuptools.find_packages(),
    requires=["dask", "numpy", "h5py", "zarr", "rich", "tqdm"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
