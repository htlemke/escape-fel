import numpy as np
from bisect import bisect
from random import randint
import matplotlib.pyplot as plt

from .plot_utilities import *



def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, np.sqrt(variance))


def corr_nonlin(data, polypar, data_0=0, correct_0=0):
    """ unses parameters found by corrNonlinGetPar to correct data;
    example of usage (assuming mon is non linear and loff is
    the laseroff filter
    d = ixppy.dataset("xppc3614-r0102.stripped.h5")
    loff = d.eventCode.code_91.filter(True)
    mon = d.ipm3.sum;
    dio = d.diodeU.channel0;
    poly = ixppy.tools.corrNonlinGetPar(dio*loff,mon*loff,plot=True)
    mon_corr = ixppy.corrNonlin(mon,poly)"""
    m = 1 / np.polyval(np.polyder(polypar), data_0)
    return m * (np.polyval(polypar, data) - correct_0) + data_0


def corr_nonlin_get_par(
    data, correct, order=2, data_0=0, correct_0=0, displayWarning=True, plot=False
):
    """ Find parameters for non linear correction
    *data* should be an 1D array (use .ravel() in case) of the
    detectors that is suspected to be non linear
    *correct* is the detector that is sussposed to be linear
    *data_0" is an offset to use for the data (used only if plotting"
    *correct_0* offset of the "linear detector"""
    # poor man wrapping :D #
    try:
        data = data.ravel()
    except AttributeError:
        pass
    try:
        correct = correct.ravel()
    except AttributeError:
        pass
    p = np.polyfit(data, correct, order)
    if order >= 2 and p[-3] < 0:
        logbook(
            "corrNonlinGetPar: consistency problem, second order coefficient should \
    be > 0, please double check result (plot=True) or try inverting the data and the\
    correct arguments",
            level=2,
            func="toolsDetectors.corrNonlinGetPar",
        )
    p[-1] = p[-1] - correct_0
    if plot:
        d = corrNonlin(data, p, data_0=data_0, correct_0=correct_0)
        plt.plot(correct, data, ".", label="before correction")
        plt.plot(correct, d, ".", label="after correction")
        poly_lin = np.polyfit(correct, d, 1)
        xmin = min(correct.min(), 0)
        xtemp = np.asarray((xmin, correct.max()))
        plt.plot(xtemp, np.polyval(poly_lin, xtemp), label="linear fit")
        plt.plot(
            correct,
            d - np.polyval(poly_lin, correct),
            ".",
            label="difference after-linear",
        )
        plt.xlabel("correct")
        plt.ylabel("data")
        plt.legend()
    return p


def edges_to_center(edges):
    edges = np.asarray(edges)
    centers = edges[:-1] + np.diff(edges)
    return centers


def hist_scan(
    data,
    cut_percentage=0,
    N_intervals=20,
    normalize_to=None,
    plot_results=True,
    plot_axis=None,
):
    x_scan = data.scan.parameter
    xscan_name = list(x_scan.keys())[1]
    x_scan = x_scan[xscan_name]["values"]
    [hmin, hmax] = np.percentile(
        data.data.ravel(), [cut_percentage, 100 - cut_percentage]
    )
    hbins = np.linspace(hmin, hmax, N_intervals + 1)
    hdat = [np.histogram(td.data.ravel(), bins=hbins)[0] for td in data.scan]
    if normalize_to is "max":
        hdat = [td / td.max() for td in hdat]
    elif normalize_to is "sum":
        hdat = [td / td.sum() for td in hdat]
    hdat = np.asarray(hdat)
    if plot_results:
        if not plot_axis:
            plot_axis = plt.gca()
        plot2D(x_scan, edges_to_center(hbins), hdat.T)
    return x_scan, hbins, hdat


def plot2D(x, y, C, *args, **kwargs):
    def bin_array(arr):
        return np.hstack([arr - np.diff(arr)[0] / 2, arr[-1] + np.diff(arr)[-1] / 2])

    Xp, Yp = np.meshgrid(bin_array(x), bin_array(y))

    return plt.pcolormesh(Xp, Yp, C, *args, **kwargs)


def get_index_modulo_array(data, mod, offset=0):
    index = data.index
    out_bool = np.mod(index, mod) == offset
    return data[out_bool]

greyscale = [
    " ",
    " ",
    ".,-",
    "_ivc=!/|\\~",
    "gjez2]/(YL)t[+T7Vf",
    "mdK4ZGbNDXY5P*Q",
    "W8KMA",
    "#%$",
]


def hist_asciicontrast(x, bins=50, range=None, disprange=True):
    h, edges = np.histogram(x, bins=bins, range=range)
    if np.sum(h) == 0:
        bounds = np.linspace(min(h), 1, len(greyscale) - 1)
    else:
        bounds = np.linspace(min(h), max(h), len(greyscale) - 1)
    hstr = ""

    for bin in h:
        syms = greyscale[bisect(bounds, bin)]
        hstr += syms[randint(0, len(syms) - 1)]

    if disprange:
        hstr = (
            "{:>10}".format("%0.5g" % (edges[0]))
            + hstr
            + "{:>10}".format("%0.5g" % (edges[-1]))
        )

    return hstr


import numpy as np


def hist_unicode(data, bins=10):
    bars = u" ▁▂▃▄▅▆▇█"
    n, _ = np.histogram(data, bins=bins)
    n2 = np.round(n * (len(bars) - 1) / (max(n))).astype(int)
    res = u" ".join([bars[i] for i in n2])
    return res


class Hist_ascii(object):
    """
  Ascii histogram
  """

    def __init__(self, data, bins=50, percRange=None, range=None):
        """
    Class constructor
    :Parameters:
    - `data`: array like object
    """
        if not percRange is None:
            range = np.percentile(data, percRange)
        self.data = data
        self.bins = bins
        self.h = np.histogram(self.data, bins=self.bins, range=range)

    def horizontal(self, height=4, character="|"):
        """Returns a multiline string containing a
    a horizontal histogram representation of self.data
    :Parameters:
        - `height`: Height of the histogram in characters
        - `character`: Character to use
    >>> d = normal(size=1000)
    >>> h = Histogram(d,bins=25)
    >>> print h.horizontal(5,'|')
    106      |||
            |||||
           |||||||
          ||||||||||
         |||||||||||||
    -3.42       3.09"""
        his = """"""
        bars = 1.0 * self.h[0] / np.max(self.h[0]) * height

        def formnum(num):
            return "{:<9}".format("%0.4g" % (num))

        for l in reversed(range(1, height + 1)):
            line = ""
            if l == height:
                line = formnum(np.max(self.h[0])) + " "  # histogram top count
            else:
                line = " " * (9 + 1)  # add leading spaces
            for c in bars:
                if c >= np.ceil(l):
                    line += character
                else:
                    line += " "
            line += "\n"
            his += line
        his += formnum(self.h[1][0]) + " " * (self.bins) + formnum(self.h[1][-1]) + "\n"
        return his

    def vertical(self, height=20, character="|"):
        """
    Returns a Multi-line string containing a
    a vertical histogram representation of self.data
    :Parameters:
        - `height`: Height of the histogram in characters
        - `character`: Character to use
    >>> d = normal(size=1000)
    >>> Histogram(d,bins=10)
    >>> print h.vertical(15,'*')
              236
    -3.42:
    -2.78:
    -2.14: ***
    -1.51: *********
    -0.87: *************
    -0.23: ***************
    0.41 : ***********
    1.04 : ********
    1.68 : *
    2.32 :
    """
        his = """"""
        xl = ["%.2f" % n for n in self.h[1]]
        lxl = [len(l) for l in xl]
        bars = self.h[0] / max(self.h[0]) * height
        his += " " * (np.max(bars) + 2 + np.max(lxl)) + "%s\n" % np.max(self.h[0])
        for i, c in enumerate(bars):
            line = xl[i] + " " * (np.max(lxl) - lxl[i]) + ": " + character * c + "\n"
            his += line
        return his
