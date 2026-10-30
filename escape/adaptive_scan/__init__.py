"""Adaptive scans: a portfolio of strategies for choosing the next scan points.

Opt-in subpackage (not imported by ``escape/__init__.py``)::

    from escape import adaptive_scan as ads

Strategies (all share grid handling, exposure choice, motion costs and
visiting order, see :mod:`~escape.adaptive_scan.strategy`):

=========================  ======================================================
:class:`UniformScan`       baseline: coarse-to-fine uniform sweeps
:class:`ModelFreeScan`     model-free: refine where linear interpolation is
                           worst (noise-aware curvature); :class:`GradientScan`
                           = slope-driven variant for edges
:class:`StepScan`          step (edge) search and characterisation
:class:`OscillationScan`   step + 1-3 damped oscillations with own decay and
                           grow-in times (alias :class:`AdaptiveScan`)
:class:`PeakScan2D`        2-D peak: centre, size, tilt and asymmetry
=========================  ======================================================

Building blocks:

* :mod:`~escape.adaptive_scan.costs` -- :class:`Axis` / :class:`Motion`
  (move times, backlash), :class:`ExposureModel`, :func:`order_route`,
  :func:`grid_points` (the :func:`escape.utilities.roundto` grid).
* :mod:`~escape.adaptive_scan.model` -- :class:`ImpulsiveResponse` /
  :class:`Oscillation` signal model (also used by
  :func:`escape.storage.get_test_data`).
* :mod:`~escape.adaptive_scan.fit` -- fit of that model, model order by BIC.
* :mod:`~escape.adaptive_scan.peak2d` -- tilted, asymmetric 2-D peak model.
* :mod:`~escape.adaptive_scan.simulate` -- simulated measurements,
  :class:`Instrument` (clock + position) and the :func:`run_plan` loop.

Quick start::

    from escape import adaptive_scan as ads

    motion = ads.Axis(speed=2.0, overhead=0.5, backlash=0.1)   # units/s, s
    instr = ads.Instrument(ads.PumpProbeSimulator(seed=0), motion=motion)
    scan = ads.ModelFreeScan((-1, 5), grid_step=0.02, motion=motion,
                            exposure=ads.ExposureModel("auto"))
    ads.run_plan(scan, instr, budget=300)
    scan.plot()
"""

from . import costs, model, fit, strategy, modelfree, design, peak2d, simulate
from .costs import (
    Axis,
    Motion,
    ExposureModel,
    order_route,
    route_time,
    grid_points,
    snap_to_grid,
)
from .model import Oscillation, ImpulsiveResponse, smeared_exp_step
from .fit import ModelSpec, FitResult, fit_model, fit_response
from .strategy import Plan, ScanStrategy, ModelScan, UniformScan, information_gain
from .modelfree import ModelFreeScan, GradientScan
from .design import OscillationScan, AdaptiveScan, StepScan
from .peak2d import Peak2D, Peak2DFit, fit_peak2d, PeakScan2D
from .simulate import (
    pump_probe_points,
    PumpProbeSimulator,
    GaussianSimulator,
    Instrument,
    run_scan,
    run_plan,
)

__all__ = [
    "costs", "model", "fit", "strategy", "modelfree", "design", "peak2d",
    "simulate",
    "Axis", "Motion", "ExposureModel", "order_route", "route_time",
    "grid_points", "snap_to_grid",
    "Oscillation", "ImpulsiveResponse", "smeared_exp_step",
    "ModelSpec", "FitResult", "fit_model", "fit_response",
    "Plan", "ScanStrategy", "ModelScan", "UniformScan", "information_gain",
    "ModelFreeScan", "GradientScan", "OscillationScan", "AdaptiveScan", "StepScan",
    "Peak2D", "Peak2DFit", "fit_peak2d", "PeakScan2D",
    "pump_probe_points", "PumpProbeSimulator", "GaussianSimulator",
    "Instrument", "run_scan", "run_plan",
]
