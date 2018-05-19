#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
VERSION = (0, 0, 1)
VERSION_STR = ".".join([str(x) for x in VERSION])
setup(
    name='escape',
    version=VERSION_STR,
    description="escape",
    long_description="event-synchronous catergorisation and processing environment",
    author='htlemke',
    author_email='htlemke@gmail.com',
    url='https://github.com/htlemke/escape',
    packages=['escape'],
    requires=['bsread','numpy','matplotlib'],
    zip_safe=False,
)
