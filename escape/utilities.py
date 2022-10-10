import numpy as np
from bisect import bisect
from random import randint
import matplotlib.pyplot as plt
from pathlib import Path
from .plot_utilities import *
import pickle
import signal
import escape.cell2function as cell2function
from rich.tree import Tree


class StructureGroup:
    def __repr__(self):
        s = object.__repr__(self)
        s += "\n"
        s += "items\n"
        for k in self.__dict__.keys():
            s += "    " + k + "\n"
        return s

    def get_structure_tree(self, base=None):
        if not base:
            base = Tree("")
        for key, item in self.__dict__.items():
            if hasattr(item, "get_structure_tree"):
                item.get_structure_tree(base=base.add(key))
            else:
                base.add(key).add(str(item))
        return base


def dict2structure(t, base=None):
    if not base:
        base = StructureGroup()
    for tt, tv in t.items():
        p = tt.split(".")
        tbase = base
        for tp in p[:-1]:
            if tp in tbase.__dict__.keys():
                if not isinstance(tbase.__dict__[tp], StructureGroup):
                    tbase.__dict__[tp] = StructureGroup()
            else:
                tbase.__dict__[tp] = StructureGroup()
            tbase = tbase.__dict__[tp]
        if hasattr(tbase, p[-1]):
            if not isinstance(tbase.__dict__[p[-1]], StructureGroup):
                tbase.__dict__[p[-1]] = tv
        else:
            tbase.__dict__[p[-1]] = tv
    return base


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average(
        (values - average) ** 2, weights=weights
    )  # Fast and numerically precise
    return (average, np.sqrt(variance))


def corr_nonlin(data, polypar, data_0=0, correct_0=0):
    """unses parameters found by corrNonlinGetPar to correct data;
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
    """Find parameters for non linear correction
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


def center_to_edges(centers):
    centers = np.asarray(centers)
    df = np.diff(centers)
    edges = centers + np.hstack([centers[:1], np.diff(centers)])
    return edges


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
    if normalize_to == "max":
        hdat = [td / td.max() for td in hdat]
    elif normalize_to == "sum":
        hdat = [td / td.sum() for td in hdat]
    hdat = np.asarray(hdat)
    if plot_results:
        if not plot_axis:
            plot_axis = plt.gca()
        plot2D(x_scan, edges_to_center(hbins), hdat.T)
    return x_scan, hbins, hdat


def plot2D(x, y, C, *args, ax=None, **kwargs):
    def bin_array(arr):
        arr = np.asarray(arr)
        return np.hstack([arr - np.diff(arr)[0] / 2, arr[-1] + np.diff(arr)[-1] / 2])

    Xp, Yp = np.meshgrid(bin_array(x), bin_array(y))
    if ax:
        plt.sca(ax)
    out = plt.pcolormesh(Xp, Yp, C, *args, **kwargs)
    try:
        plt.xlabel(x.name)
    except:
        pass
    try:
        plt.ylabel(y.name)
    except:
        pass
    return out


def pickle_figure(filename, figure=None):
    if not figure:
        figure = plt.gcf()

    filename = Path(filename)
    if not filename.suffix == "pickfig":
        filename = filename.parent / (filename.stem + ".pickfig")
        print(filename.as_posix())
    with open(filename, "wb") as f:
        pickle.dump(figure, f)


def unpickle_figure(filename):
    filename = Path(filename)
    with open(filename, "rb") as f:
        return pickle.load(f)


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
    bars = " ▁▂▃▄▅▆▇█"
    n, _ = np.histogram(data, bins=bins)
    n2 = np.round(n * (len(bars) - 1) / (max(n))).astype(int)
    res = " ".join([bars[i] for i in n2])
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


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def polyfit_with_fixed_points(x, y, n, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x ** np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[: n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[: n + 1, : n + 1] = np.take(x_n, idx)
    xf_n = xf ** np.arange(n + 1)[:, None]
    mat[: n + 1, n + 1 :] = xf_n / 2
    mat[n + 1 :, : n + 1] = xf_n.T
    mat[n + 1 :, n + 1 :] = 0
    vec[: n + 1] = yx_n
    vec[n + 1 :] = yf
    params = np.linalg.solve(mat, vec)
    return params[: n + 1][::-1]


def get_corr(data, ref, order=2):
    p = []
    p_fx = []
    std = []
    std_fx = []
    for i in range(1, order + 1):
        if len(data) > 1:
            p.append(np.polyfit(ref, data, i))
            p_fx.append(polyfit_with_fixed_points(ref, data, i, [0], [0]))
            std.append(np.std(data / np.polyval(p[-1], ref)))
            std_fx.append(np.std(data / np.polyval(p_fx[-1], ref)))
        else:
            std.append(np.nan)
            std_fx.append(np.nan)
    return np.asarray(std), np.asarray(std_fx)
