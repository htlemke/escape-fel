Timestamps
==========

``ArrayTimestamps`` and ``ScanTimestamps`` are helper classes for data
acquired against a monotonic timestamp rather than a discrete pulse ID.
They mirror the :class:`~escape.Array` / :class:`~escape.Scan` pair but use
timestamp intervals to define scan steps instead of event-index step lengths.

ArrayTimestamps
---------------

.. autoclass:: escape.ArrayTimestamps
   :members:
   :special-members: __init__, __len__
   :show-inheritance:

ScanTimestamps
--------------

.. autoclass:: escape.storage.storage_timestamps.ScanTimestamps
   :members:
   :special-members: __init__, __getitem__, __len__
   :show-inheritance:
