Bernina diffractometer
======================

``escape.bernina`` holds the calculations specific to the Bernina
endstation's diffractometer and detectors, on top of
`hklforge <https://github.com/swissfel/hklforge>`_ (the angle solver and
general crystallography). eco uses them on the beamline and the
bernina-scripts prep apps on Binder and in the browser. It is an opt-in
subpackage that needs numpy, pandas and hklforge only, not escape's data
stack: ``pip install "escape-fel[bernina]"``.

.. code-block:: python

   from escape.bernina import diffraction as dif

   s = dif.Sample({"a": 5.431, "b": 5.431, "c": 5.431, "alpha": 90, "beta": 90, "gamma": 90},
                  [((0, 0, 1), (0, 1, 0)), ((1, 0, 0), (0, 0, 1))])   # (hkl) > lab direction
   s.solve((1, 1, 1), 9000, {"mu": 0, "gamma": 0, "phi": 0})          # every solution, as a table

.. automodule:: escape.bernina

.. automodule:: escape.bernina.geometry
   :members:

.. automodule:: escape.bernina.diffraction
   :members:

.. automodule:: escape.bernina.recspace
   :members:

.. automodule:: escape.bernina.detectors
   :members:

.. automodule:: escape.bernina.detector_view
   :members:

.. automodule:: escape.bernina.otti
   :members:
