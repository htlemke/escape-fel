Detector Utilities
==================

See the :doc:`user guide <../user_guide/detectors>`.

.. warning::

   Experimental. ``dynamic_pedestal`` in particular is beta and not yet
   validated against real detector data: treat recovered offsets as a
   hypothesis to verify, not a drop-in calibration.

Droplets
--------

.. automodule:: escape.utilities_detectors.droplets
   :members: find_droplets, fit_photon_number_distribution, PoissonFitResult,
             droplets_to_image_block, droplets_to_sparse_block,
             droplets_to_photon_count_block

Dynamic gain-switching pedestal
-------------------------------

.. automodule:: escape.utilities_detectors.dynamic_pedestal
   :members: decode_jungfrau_raw, apply_calibration, find_gain_switch_offset,
             find_dynamic_pedestal_offsets, aggregate_offsets
