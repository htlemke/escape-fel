"""
Minimal file I/O for XAS data.

Most beamlines and processing tools (Athena, Larch, SixPACK, ...)
export simple whitespace/comma-delimited column text files with a
header of comment lines starting with '#'. This module reads those
without requiring any exotic dependency.
"""

from __future__ import annotations
import numpy as np


def read_columns(path, energy_col=0, mu_col=1, comment="#", delimiter=None):
    """
    Read a plain-text column data file (e.g. .xmu, .dat, .txt, .csv).

    Parameters
    ----------
    path : str
        File path.
    energy_col, mu_col : int
        Zero-based column indices for energy and mu(E). For a typical
        3-column "energy mu i0" file the defaults (0, 1) are correct.
        For files where mu must be computed from I0/I1 (mu = -ln(I1/I0)),
        read the raw columns yourself with `numpy.loadtxt` and compute
        mu, then feed the arrays directly into the rest of the package
        (all functions take plain arrays, not file paths).
    comment : str
        Lines starting with this string are ignored.
    delimiter : str or None
        Column delimiter; None means "any whitespace" (numpy default),
        use ',' for CSV files.

    Returns
    -------
    energy, mu : ndarray
    """
    data = np.loadtxt(path, comments=comment, delimiter=delimiter)
    energy = data[:, energy_col]
    mu = data[:, mu_col]
    order = np.argsort(energy)
    return energy[order], mu[order]
