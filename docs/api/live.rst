Live Results
============

See the :doc:`user guide <../user_guide/live>`. ``escape.live`` is opt-in
(``from escape import live``).

Parameters and lineage
----------------------

.. autoclass:: escape.storage.lineage.Param
   :members: value, set, observe
.. autofunction:: escape.storage.lineage.batch
.. autofunction:: escape.storage.lineage.describe
.. autofunction:: escape.storage.lineage.upstream_params
.. autoclass:: escape.storage.lineage.Node
   :members: get, value, tool

Live plots and panels
---------------------

.. autoclass:: escape.live.LivePlot
   :members:
.. autofunction:: escape.live.live_plot
.. autofunction:: escape.live.panel
.. autofunction:: escape.live.set_enabled
