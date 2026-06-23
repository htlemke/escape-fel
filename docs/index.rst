escape-fel
==========

**Event Synchronous Categorisation And Processing Environment**

``escape`` is a high-level, object-oriented Python framework for processing
event-based measurement data from Free-Electron Laser (FEL) experiments and
similar facilities. It abstracts over large volumes of per-pulse data using
`dask <https://docs.dask.org>`_ for lazy, scalable evaluation and provides
a clean interface for aligning, grouping, analysing, and storing event streams.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/array_basics
   user_guide/array_operations
   user_guide/scan
   user_guide/grid
   user_guide/dataset
   user_guide/custom_processing

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/quickstart
   examples/pump_probe

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index
