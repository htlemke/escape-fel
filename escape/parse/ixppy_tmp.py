"""Tools for reading ixp.h5 files from deprecated ixppy. (work in progress)"""

from .utilities import findItemnamesGroups
import h5py
from .. import Array
from threading import Thread
import numpy as np
import dask.array as da


def parse_ixp(ixp_file):
    # with h5py.File(ixp_file, "r") as fh:
    fh = h5py.File(ixp_file, "r")
    basegroup = fh["dataset"]
    dss = findItemnamesGroups(basegroup, item_names=["data", "time"])
    datastores = {}
    escArrays = {}
    for name, (datagrp, timegrp) in dss.items():
        datastores[name] = parse_ixp_dataset(datagrp, timegrp)
        container = LazyContainer(datastores[name])
        try:
            datastores[name]["scan_par"] = {
                tn: {"values": tv}
                for tn, tv in parse_ixp_scan_group(datagrp.parent["scan"]).items()
            }
        except:
            pass
        escArrays[name] = Array(
            container.get_data,
            index=container.get_eventIds,
            step_lengths=datastores[name]["step_lengths"],
            # parameter=datastores[name]["scan_par"],
            # scan=dat["scan"],
        )
    return escArrays

    # return ret
    # escArrays = {}
    # containers = {}
    # for name, dat in dstores.items():
    #     containers[name] = LazyContainer(dat)
    #     escArrays[name] = Array(
    #         containers[name].get_data,
    #         index=containers[name].get_eventIds,
    #         step_lengths=dat["stepLengths"],
    #         scan=dat["scan"],
    #     )
    # return escArrays
    #     data=None,
    #     index=None,
    #     step_lengths=None,
    #     parameter=None,
    #     index_dim=None,
    #     name=None,


def parse_ixp_dataset(datagrp, timegrp):
    datastepnames = sorted([tk for tk in datagrp.keys() if tk[0] == "#"])
    timestepnames = sorted([tk for tk in timegrp.keys() if tk[0] == "#"])
    datasets_data = []
    datasets_time = []
    selection = []
    step_lengths = []
    for stepnumber, stepname in enumerate(datastepnames):
        ds_step_data = datagrp[stepname]
        ds_step_time = timegrp[stepname]
        # print(ds_step_data)
        if ds_step_data.ndim > 0 and ds_step_time.ndim > 0:
            if not (ds_step_data.shape[0] == ds_step_time.shape[0]):
                raise Exception("data and time arrays are not of equal length")
        else:
            print(f"cancelled working on step: {stepname}")
            continue
        datasets_data.append(ds_step_data)
        datasets_time.append(ds_step_time)
        selection.append(stepnumber)
        step_lengths.append(ds_step_time.shape[0])
    result = {
        "data": datasets_data,
        "eventIds": datasets_time,
        "selection": selection,
        "step_lengths": step_lengths,
    }
    return result


def parse_ixp_scan_group(scangrp):
    parameters = list(scangrp.keys())
    scan_pars = {}
    for parameter in parameters:
        scan_pars[parameter] = scangrp[parameter][:]
    return scan_pars
    ###


class LazyContainer:
    def __init__(self, dat):
        self.dat = dat

    def get_data(self, **kwargs):
        return da.concatenate([da.from_array(td) for td in self.dat["data"]])

    def get_eventIds(self):
        ids = {}

        def getids(n, dset):
            ids[n] = dset[...].ravel()

        ts = [
            Thread(target=getids, args=[n, td])
            for n, td in enumerate(self.dat["eventIds"])
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return np.concatenate([ids[n] for n in range(len(self.dat["eventIds"]))])
