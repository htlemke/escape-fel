Plotting & Interactive Widgets
==============================

Notebook-friendly figure helpers and interactive ROI-selection / image-stack
viewers. Everything on this page lives in ``escape.plot_utilities`` and is
re-exported through ``escape.utilities`` (e.g. ``escape.utilities.StackViewer``).

Figure Creation
----------------

.. autofunction:: escape.plot_utilities.nfigure
.. autofunction:: escape.plot_utilities.nsubplots
.. autofunction:: escape.plot_utilities.nsubplot_mosaic

Interactive ROI Selectors
--------------------------

.. autoclass:: escape.plot_utilities.RectangleSelectNB
   :members:
.. autoclass:: escape.plot_utilities.SpanSelectNB
   :members:
.. autoclass:: escape.plot_utilities.PolygonSelectNB
   :members:
.. autoclass:: escape.plot_utilities.LassoSelectNB
   :members:
.. autoclass:: escape.plot_utilities.GinputNB
   :members:
.. autoclass:: escape.plot_utilities.RoiRegion
   :members:
.. autoclass:: escape.plot_utilities.RoiPanel
   :members:
.. autoclass:: escape.plot_utilities.MultipleRoiSelector
   :members:

Histogram Range Selectors
--------------------------

Returned by :meth:`escape.Array.filter_interactive` and
:meth:`escape.Array.digitize_interactive`.

.. autoclass:: escape.hist_select.HistogramFilter
   :members:
.. autoclass:: escape.hist_select.HistogramDigitizer
   :members:

Image-Stack & Step Viewers
----------------------------

.. autoclass:: escape.plot_utilities.StackViewer
   :members:
.. autoclass:: escape.plot_utilities.StepViewer
   :members:
.. autoclass:: escape.plot_utilities.StepViewerP
   :members:

Plotting Helpers
------------------

.. autofunction:: escape.plot_utilities.errortube
.. autofunction:: escape.plot_utilities.dual_x_axis_plot
.. autofunction:: escape.plot_utilities.add_crossing_secondary_axis
.. autofunction:: escape.plot_utilities.add_step_secondary_axis
.. autoclass:: escape.plot_utilities.SecondaryCrossing
   :members:
