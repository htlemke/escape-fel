import numpy as np
from bisect import bisect
from random import randint

greyscale = [
      " ",
      " ",
      ".,-",
      "_ivc=!/|\\~",
      "gjez2]/(YL)t[+T7Vf",
      "mdK4ZGbNDXY5P*Q",
      "W8KMA",
      "#%$"
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

def hist_unicode(data,bins=10):
    bars = u' ▁▂▃▄▅▆▇█'
    n,_ = np.histogram(data,bins=bins)
    n2 = np.round(n*(len(bars)-1)/(max(n))).astype(int)
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
