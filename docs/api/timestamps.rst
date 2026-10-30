Timestamps
==========

``ArrayTimestamps`` and ``ScanTimestamps`` are helper classes for data
acquired against a monotonic timestamp rather than a discrete pulse ID.
They mirror the :class:`~escape.Array` / :class:`~escape.Scan` pair but use
timestamp intervals to define scan steps instead of event-index step lengths.

Channel-monitor data (values recorded on change, e.g. EPICS channels) is
stored this way. ``ArrayTimestamps.plot()`` plots it against clock time on a
date axis, with each scan step's time range shaded:

.. code-block:: python

   mon = ds.some_monitor_channel          # an ArrayTimestamps
   mon.plot()                             # data vs. time, steps shaded
   mon.plot(show_scan_steps=False, timestamp_unit="ns")

Printing an ``ArrayTimestamps`` shows a short text histogram, per scan step
if there are several.

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
