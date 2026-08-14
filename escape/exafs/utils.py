"""
Small standalone helper functions that don't belong in a specific
processing stage.
"""

from __future__ import annotations
import numpy as np


def kweight_chi(k, chi, kweight=2):
    """Return k^kweight * chi(k) -- the most common way to plot/inspect EXAFS."""
    k = np.asarray(k, dtype=float)
    chi = np.asarray(chi, dtype=float)
    return (k ** kweight) * chi


def smooth(y, window_len=5):
    """
    Simple moving-average smoothing (odd window_len). Useful for a
    quick look at noisy data; NOT a substitute for proper background
    removal or a real low-pass filter for quantitative work.
    """
    y = np.asarray(y, dtype=float)
    if window_len < 3:
        return y.copy()
    if window_len % 2 == 0:
        window_len += 1
    kernel = np.ones(window_len) / window_len
    pad = window_len // 2
    y_padded = np.pad(y, pad, mode="edge")
    return np.convolve(y_padded, kernel, mode="valid")


def rebin_to_grid(x, y, new_x):
    """Linear interpolation of (x, y) onto a new grid `new_x`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    new_x = np.asarray(new_x, dtype=float)
    return np.interp(new_x, x, y)


def r_range_mask(r, rmin, rmax):
    """Boolean mask selecting rmin <= r <= rmax -- handy for windowed fits."""
    r = np.asarray(r, dtype=float)
    return (r >= rmin) & (r <= rmax)
