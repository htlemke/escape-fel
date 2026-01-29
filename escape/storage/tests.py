import numpy as np
import h5py


def testEventIds(N=100 * 60 * 10, Nsteps=100, shuffle=False):
    e = np.arange(N)
    if shuffle:
        np.random.shuffle(e)
    m = int(np.ceil(N / Nsteps))
    return [e[n * m : (n + 1) * m] for n in range(Nsteps)]


# def shuffledata(tei):


def finder1(i1, i2, stepLens=None):
    com = i1[np.isin(i1, i2, assume_unique=True)]
    srt = i2.argsort(axis=0)
    return srt[np.searchsorted(i2, com, sorter=srt)]


def getStepLens(i1, stepLens1, i2):
    ind1 = finder1(i2, i1)
    return np.bincount(np.digitize(ind1, bins=np.cumsum(stepLens1)))


def matcher(i1, i2, stepLens1):
    if type(i1) is list:
        i1 = np.hstack(i1)
    if type(i2) is list:
        i2 = np.hstack(i2)
    ind2 = finder1(i1, i2)
    ind1 = finder1(i2, i1)
    stepLensNew = np.bincount(np.digitize(ind1, bins=np.cumsum(stepLens1)))
    return ind1, ind2, stepLensNew


def getStep(i, stepLens, ind):
    pass


def testFinders():
    i1 = testEventIds()
    i2 = testEventIds(shuffle=True)
    lens1 = [len(t) for t in i1]
    lens2 = [len(t) for t in i2]
    i1 = hstack(i1)
    i2 = hstack(i2)
    # ind = finder1(i2,i1)

    return i1, i2


def createTestfile(fina, shape=(100, 100)):
    with h5py.File(fina, "w") as f:
        i = testEventIds()
        lens = [len(t) for t in i]
        ids = np.hstack(i)
        f["lens"] = lens
        f["ids"] = ids
        for n, tl in enumerate(lens):
            f["ds%04d" % n] = np.random.randn(*((tl,) + shape))


def getDArrays(fina):
    f = h5py.File(fina, "r")
    # print(f.items())
    stepName = [(int(key[2:]), key) for key in f.keys() if key[:2] == "ds"]
    step, name = (t for t in zip(*stepName))
    sorter = np.argsort(step)
    dsets = [f[name[s]] for s in sorter]
    return dsets
