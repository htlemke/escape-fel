from collections.abc import Iterable
from matplotlib import colors
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
import os
import sys
import inspect
from scipy.stats import skew
from scipy.interpolate import interp1d
import scipy


units = {
    0: " ",
    1: "K",
    2: "M",
    3: "G",
    4: "T",
    5: "P",
    6: "E",
    7: "Z",
    8: "Y",
    9: "R",
    10: "Q",
    -1: "m",
    -2: "u",
    -3: "n",
    -4: "p",
    -5: "f",
    -6: "a",
    -7: "z",
    -8: "y",
    -9: "r",
    -10: "q",
}


def num2sci(a):
    a = np.atleast_1d(a)
    exp = np.log10(np.max(a)) // 3
    u = units[exp]
    return a / (10 ** (exp * 3)), u


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

    def get_dict(self):
        d = {}
        for key, item in self.__dict__.items():
            if hasattr(item, "get_dict"):
                d[key] = item.get_dict()
            else:
                d[key] = item
        return d

    def append_members_to_namepace(self, namespace=None):
        if namespace is None:
            frame = inspect.currentframe().f_back
            namespace = frame.f_locals

        print(list(namespace.keys()))
        for k, v in self.__dict__.items():
            if k in namespace.keys():
                print(
                    f"Warning: variable {k} is already in namespace and is now overwritten!"
                )
            namespace[k] = v


def dict2structure(t, base=None):
    """convert flattened dictionary with separator string "." in keys to a StructureGroup object

    Args:
        t (dict): flattened dictionary
        base (StructureGroup, optional): Structuregroup instance to append to. Defaults to None.

    Returns:
        StructureGroup instance
    """
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


try:
    from collections.abc import MutableMapping
except:
    from collections import MutableMapping


def flatten_dictionary(d, parent_key="", sep="."):
    """flatten a disctionary to a single hierarchy, based on a separator string.
    e.g. flatten_dictionary({'a':{'b':1}}) --> {'a.b':1}

    Args:
        d (dict): hierarchical dictionary
        parent_key (str, optional): Optional prefix in tructure. Defaults to "".
        sep (str, optional): separator string. Defaults to ".".

    Returns:
        dict: flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dictionary(dflat, sep="."):
    """Unflatten a disctionary with single hierarchy values, based on a separator in string names.
    e.g. unflatten_dictionary({'a.b':1}) --> {'a':{'b':1}}

    Args:
        dflat (dict): flat dictionary
        sep (str, optional): separator. Defaults to '.'.

    Returns:
        dict: unflattened dictionary
    """
    d = {}
    for tka, tv in dflat.items():
        if isinstance(tka, str):
            tks = tka.split(sep)
            p = d
            for tk in tks[:-1]:
                if not (tk in p.keys()):
                    p[tk] = {}
                p = p[tk]
            p[tks[-1]] = tv
        else:
            d[tka] = tv
    return d


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    if (np.asarray(weights)==0).all():
        return (np.nan, np.nan)
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
    """Create an array from binning edges that define the centers
    of those bins (Length of 1D array reduces by one).

    Args:
        edges (1D list/array): bin edges

    Returns:
        array: bin centers (1D)
    """
    edges = np.asarray(edges)
    centers = edges[:-1] + np.diff(edges)
    return centers


def center_to_edges_old(centers):
    """Create an edges array for digitizing / binning from  a given monotonic array.
    Edges are based on the half difference between neighboring values.

    Args:
        centers (list/array): centers aroung which edges are to be created.

    Raises:
        Exception: is centers are non-monotonic

    Returns:
        array : edges
    """
    centers = np.asarray(centers)
    df = np.diff(centers)
    dfs = np.sign(df)
    if not np.all(dfs == dfs[0]):
        raise Exception("Centers need to be monotonic! ")

    edges = np.hstack([centers[:1], centers]) + np.hstack(
        [-df[:1] / 2, df / 2, df[-1:] / 2]
    )
    return edges

def roundto(v,interval):
    return np.rint(v/interval)*interval

def center_to_edges(a, axis=-1):

    nd = np.ndim(a)
    if isinstance(axis, Iterable):
        o = center_to_edges(a, axis=axis[0])
        for taxis in axis[1:]:
            o = center_to_edges(o, axis=taxis)
        return o
    #     axis = nd+axis

    d = np.diff(a, axis=axis)
    # print(d)
    o = np.concatenate(
        (
            a.take(indices=(0,), axis=axis) - d.take(indices=(0,), axis=axis) / 2,
            a.take(indices=range(0, a.shape[axis] - 1), axis=axis) + d / 2,
            a.take(indices=(-1,), axis=axis) + d.take(indices=(-1,), axis=axis) / 2,
        ),
        axis=axis,
    )
    return o


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


def plot2D(x, y, C, *args, 
           axis=None, 
           diverging=False, 
           log_colors=False, 
           pars_symlognorm = dict(
               linthresh=1.0,
               linscale=1.0,
               vmin='auto',
               vmax='auto',
               base=10,
           ),
           **kwargs):
    """Helper function to create a fals color 3D plot using matplotlib pcolormesh.

    Args:
        x (array-like 1d): may be replaced "auto" for bin number
        y (array-like 1d): may be replaced "auto" for bin number
        C (array-like 2d): [description]
        ax (matplotlib axis): [description]. Defaults to None.
    """
    if axis is None:
        axis = kwargs.pop("ax", None)
    def bin_array(arr):
        arr = np.asarray(arr)
        return np.hstack([arr - np.diff(arr)[0] / 2, arr[-1] + np.diff(arr)[-1] / 2])
    
    C = np.asarray(C)

    if type(x) is str and x == "auto":
        x = np.arange(C.shape[1])
    if type(y) is str and y == "auto":
        y = np.arange(C.shape[0])

    Xp, Yp = np.meshgrid(bin_array(x), bin_array(y))
    if axis:
        plt.sca(axis)
    if diverging:
        if log_colors:
            if pars_symlognorm.get("vmin","auto")=="auto":
                vminmax=np.max(
                    np.ceil(np.log10(np.abs(np.nanmin(C)))).astype(int), 
                    np.ceil(np.log10(np.abs(np.nanmax(C)))).astype(int),
                    )
            out = plt.pcolormesh(Xp, Yp, 
                                 C, *args, **kwargs, 
                                 cmap=kwargs.get("cmap","coolwarm"), 
                                 norm=colors.SymLogNorm(**pars_symlognorm))
        else:
            out = plt.pcolormesh(Xp, Yp, C, *args, **kwargs, cmap=kwargs.get("cmap","coolwarm"), norm=colors.CenteredNorm())
    else:
        if log_colors:
            out = plt.pcolormesh(Xp, Yp, C, *args, **kwargs, norm=colors.LogNorm(vmin=np.nanmin(C[np.nonzero(C)]), vmax=np.nanmax(C)))
        else:
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
    """Polynomial fit to x/y data with partially fixed x and y data.

    Args:
        x (np.array): xdata
        y (np.array): ydata
        n (integer): polynomial order
        xf (_type_): _description_
        yf (_type_): _description_

    Returns:
        array: polynomial coefficients, length of order n.
    """
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


def get_corr(data, ref, order=2, weighted=True):
    p = []
    p_fx = []
    std = []
    std_fx = []
    for i in range(1, order + 1):
        if len(data) > 1:
            p.append(np.polyfit(ref, data, i))
            p_fx.append(polyfit_with_fixed_points(ref, data, i, [0], [0]))

            if weighted:
                avg_tmp, std_tmp = weighted_avg_and_std(data / np.polyval(p[-1], ref), np.polyval(p[-1], ref))
                std.append(std_tmp)
                avg_tmp, std_tmp = weighted_avg_and_std(data / np.polyval(p_fx[-1], ref), np.polyval(p_fx[-1], ref))
                std_fx.append(std_tmp)

            else:
                std.append(np.std(data / np.polyval(p[-1], ref)))
                std_fx.append(np.std(data / np.polyval(p_fx[-1], ref)))
        else:
            std.append(np.nan)
            std_fx.append(np.nan)
    return np.asarray(std), np.asarray(std_fx)


def pgroup2name(pgroup, beamline="bernina"):
    tp = f"/sf/{beamline}/exp/"
    d = Path(tp)
    dirs = [i for i in d.glob("*") if i.is_symlink()]
    names = [i.name for i in dirs]
    targets = [i.resolve().name for i in dirs]
    return names[targets.index(pgroup)]


def name2pgroups(name, beamline="bernina"):
    tp = f"/sf/{beamline}/exp/"
    d = Path(tp)
    dirs = [i for i in d.glob("*") if i.is_symlink()]
    names = [i.name for i in dirs]
    targets = [i.resolve().name for i in dirs]
    eq = [[i_n, i_p] for i_n, i_p in zip(names, targets) if name == i_n]
    ni = [
        [i_n, i_p]
        for i_n, i_p in zip(names, targets)
        if (not name == i_n) and (name in i_n)
    ]
    return eq + ni


def smooth(x, window_len=11, window="hanning", clip_output=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    if clip_output:
        return y[(window_len / 2 - 1) : -(window_len / 2)]
    else:
        return y


def weighted_quantile(
    values, quantiles, sample_weight=None, values_sorted=False, old_style=False
):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


##########


def peakAna(x, y, nb=3, plotpoints=False):
    """nb = number of point (on each side) to use as background"""
    ## get background
    xb = np.hstack((x[0:nb], x[-(nb):]))
    yb = np.hstack((y[0:nb], y[-(nb):]))
    a = np.polyfit(xb, yb, 1)
    b = np.polyval(a, x)
    yf = y - b
    yd = np.diff(yf)

    ## determine whether peak or step
    ispeak = np.abs(skew(yf)) > np.abs(skew(yd))
    if ispeak:
        yw = yf
        xw = x
    else:
        yw = yd
        xw = (x[1:] + x[0:-1]) / 2
        ## get background
        xwb = np.hstack((xw[0:nb], xw[-(nb):]))
        ywb = np.hstack((yw[0:nb], yw[-(nb):]))
        aw = np.polyfit(xwb, ywb, 1)
        bw = np.polyval(aw, xw)
        yw = yw - bw

    Iw = (xw[1:] - xw[0:-1]) * (yw[1:] + yw[0:-1]) / 2
    if sum(Iw) < 0:
        yw = -yw

    ## get parameters
    mm = yw.argmax(0)
    PEAK = xw[mm]
    ywmax = yw[mm]
    gg = (yw[:mm][::-1] < (ywmax / 2)).argmax()
    ip = interp1d(
        yw.take([mm - gg - 1, mm - gg]), xw.take([mm - gg - 1, mm - gg]), kind="linear"
    )
    xhm1 = ip(ywmax / 2)
    gg = (yw[mm:] < (ywmax / 2)).argmax()
    ip = interp1d(
        yw.take([mm + gg, mm + gg - 1]), xw.take([mm + gg, mm + gg - 1]), kind="linear"
    )
    xhm2 = ip(ywmax / 2)

    FWHM = np.abs(xhm2 - xhm1)
    CEN = (xhm2 + xhm1) / 2
    if plotpoints and ispeak is True:
        # plot the found points for center and FWHM edges
        plt.ion()
        plt.hold(True)
        plt.plot(x, b, "g--")
        plt.plot(x, b + ywmax, "g--")
        plt.plot([xhm1, xhm1], polyval(a, xhm1) + [0, ywmax], "g--")
        plt.plot([xhm2, xhm2], polyvfal(a, xhm2) + [0, ywmax], "g--")
        plt.plot([CEN, CEN], polyval(a, CEN) + [0, ywmax], "g--")
        plt.plot([xhm1, xhm2], [polyval(a, xhm1), polyval(a, xhm2)] + ywmax / 2, "gx")
        plt.draw()

    if not ispeak:
        try:
            # findings start of step coming from left.
            std0 = scipy.std(y[0:nb])
            nt = nb
            while (scipy.std(y[0:nt]) < (2 * std0)) and (nt < len(y)):
                nt = nt + 1
            lev0 = scipy.mean(y[0:nt])
            # findings start of step coming from right.
            std0 = scipy.std(y[-nb:])
            nt = nb
            while (scipy.std(y[-nt:]) < (2 * std0)) and (nt < len(y)):
                nt = nt + 1
            lev1 = scipy.mean(y[-nt:])
            gg = np.abs(y - ((lev0 + lev1) / 2)).argmin()
            ftx = y[gg - 2 : gg + 2]
            fty = x[gg - 2 : gg + 2]
            if ftx[-1] < ftx[0]:
                ftx = ftx[::-1]
                fty = fty[::-1]
            ip = interp1d(ftx, fty, kind="linear")
            CEN = ip((lev0 + lev1) / 2)
            gg = np.abs(y - (lev1 + (lev0 - lev1) * 0.1195)).argmin()
            ftx = y[gg - 2 : gg + 2]
            fty = x[gg - 2 : gg + 2]
            if ftx[-1] < ftx[0]:
                ftx = ftx[::-1]
                fty = fty[::-1]
            # print " %f %f %f %f %f" % (ftx[0],ftx[1],fty[0],fty[1],lev1+(lev0-lev1)*0.1195)
            ip = interp1d(ftx, fty, kind="linear")
            H1 = ip((lev1 + (lev0 - lev1) * 0.1195))
            # print "H1=%f" % H1

            gg = np.abs(y - (lev0 + (lev1 - lev0) * 0.1195)).argmin()

            ftx = y[gg - 2 : gg + 2]
            fty = x[gg - 2 : gg + 2]

            if ftx[-1] < ftx[0]:
                ftx = ftx[::-1]
                fty = fty[::-1]
            #    print " %f %f %f %f %f" % (ftx[0],ftx[1],fty[0],fty[1],lev0+(lev1-lev0)*0.1195)
            ip = interp1d(ftx, fty, kind="linear")
            H2 = ip((lev0 + (lev1 - lev0) * 0.1195))
            # print "H2=%f" % abs(H2-H1)
            FWHM = abs(H2 - H1)
            if plotpoints is True:
                # plot the found points for center and FWHM edges
                plt.ion()
                # plt.hold(True)
                plt.plot([x.min(), x.max()], [lev0, lev0], "g--")
                plt.plot([x.min(), x.max()], [lev1, lev1], "g--")
                plt.plot([H2, H2], [lev0, lev1], "g--")
                plt.plot([H1, H1], [lev0, lev1], "g--")
                plt.plot([CEN, CEN], [lev0, lev1], "g--")
                plt.plot(
                    [H2, CEN, H1],
                    [
                        lev0 + (lev1 - lev0) * 0.1195,
                        (lev1 + lev0) / 2,
                        lev1 + (lev0 - lev1) * 0.1195,
                    ],
                    "gx",
                )
                plt.draw()
        except:
            CEN = np.nan
            FWHM = np.nan
            PEAK = np.nan
    return (CEN, FWHM, PEAK)


class ReferenceByRunno:
    def __init__(self, name=None):
        self.name = name
        self.data = dict()

    def append(self, run_number, value):
        self.data[run_number] = value

    def get_closest_before(self, run_number):
        ks = np.asarray(list(self.data.keys()))
        cl_k = ks[ks <= run_number].max()
        return self.data[cl_k]
