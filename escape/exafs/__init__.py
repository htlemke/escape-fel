"""EXAFS data reduction for ``escape`` (mu(E) -> chi(k) -> chi(R)).

A small, dependency-light (numpy + scipy) toolkit for turning raw XAS data
``mu(E)`` into EXAFS ``chi(k)`` and ``chi(R)``, written to be **read and
understood step by step** rather than treated as a black box.  It is packaged
here as an optional ``escape`` helper module (like :mod:`escape.wavefront`) and
adds a dask-parallel :mod:`~escape.exafs.batch` layer for reducing whole stacks
of spectra lazily over an :class:`escape.Array`.

Provenance
----------
The single-spectrum reduction is a compact re-implementation of the standard
EXAFS recipe as established by IFEFFIT / Larch / Athena:

* Background removal follows the **AUTOBK** algorithm --
  M. Newville, P. Livins, Y. Yacoby, J. J. Rehr & E. A. Stern,
  *Phys. Rev. B* **47**, 14126 (1993), doi:10.1103/PhysRevB.47.14126.
* The energy->k constant ``ETOK`` and the FT conventions match
  IFEFFIT/Larch (M. Newville, *J. Synchrotron Rad.* **8**, 322 (2001);
  Larch: https://xraypy.github.io/xraylarch/).
* The Fourier-transform-to-``chi(R)`` idea is due to
  D. E. Sayers, E. A. Stern & F. W. Lytle, *Phys. Rev. Lett.* **27**,
  1204 (1971), doi:10.1103/PhysRevLett.27.1204.

For production analysis (shell fitting against FEFF paths) use
`Larch <https://xraypy.github.io/xraylarch/>`_ /
`Demeter/Artemis <https://bruceravel.github.io/demeter/>`_; this package stops
at ``chi(R)`` and does not fit shell parameters.

Typical workflow::

    from escape import exafs

    energy, mu = exafs.read_columns("my_data.xmu")
    pre  = exafs.pre_edge(energy, mu)                       # 1. normalise
    bkg  = exafs.autobk(energy, mu, pre.e0, pre.edge_step)  # 2. chi(k)
    r, chir, _ = exafs.ft_windowed(bkg.k, bkg.chi, kweight=2,
                                   kmin=3, kmax=13)          # 3. chi(R)
"""

from .constants import ETOK
from .energy_k import energy_to_k, k_to_energy, energy_to_q, q_to_energy, find_e0
from .preedge import pre_edge, PreEdgeResult
from .background import autobk, BackgroundResult
from .fourier import window, ft_windowed
from .utils import kweight_chi, smooth, rebin_to_grid, r_range_mask
from .io import read_columns
from .plotting import plot_bkg, plot_chik, plot_chir
from .batch import (
    optical_density,
    common_k_grid,
    reduce_spectrum,
    reduce_array,
    ft_array,
)

__all__ = [
    "ETOK",
    "energy_to_k", "k_to_energy", "energy_to_q", "q_to_energy", "find_e0",
    "pre_edge", "PreEdgeResult",
    "autobk", "BackgroundResult",
    "window", "ft_windowed",
    "kweight_chi", "smooth", "rebin_to_grid", "r_range_mask",
    "read_columns",
    "plot_bkg", "plot_chik", "plot_chir",
    # escape / dask batch layer
    "optical_density", "common_k_grid", "reduce_spectrum",
    "reduce_array", "ft_array",
]

__version__ = "0.1.0"
