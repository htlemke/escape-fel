import h5py
import numpy as np
import dask.array as da
from escape import Array
from escape.parse.utilities import findItemnamesGroups


def parse_ixp_file(filename):
    fh = h5py.File(filename, "r")
    grps = findItemnamesGroups(fh, ["data", "time"], get_full_name=True)
    out = {}
    for name, dsets in grps.items():
        out[name] = parse_ixp_group(dsets[0].parent)
    return out


def parse_ixp_group(h5py_group):
    group_data_kys = sorted([tk for tk in h5py_group["data"].keys() if tk[0] == "#"])
    group_time_kys = sorted([tk for tk in h5py_group["time"].keys() if tk[0] == "#"])

    if not group_data_kys == group_time_kys:
        raise Exception("datasets in data and time dont match!")

    parameters = {}
    try:
        for tk in h5py_group["scan"].keys():
            values = h5py_group["scan"][tk][:]
            parameters[tk] = values
    except:
        print("could not extract scan array data")

    dsets_data = []
    dsets_time = []
    step_lengths = []
    par_reduced = {pk: {"values": []} for pk in parameters.keys()}
    for n, tk in enumerate(group_data_kys):
        if not isinstance(h5py_group["time"][tk], h5py.Dataset):
            continue
        if not h5py_group["time"][tk].shape:
            continue
        dsets_data.append(h5py_group["data"][tk])
        dsets_time.append(h5py_group["time"][tk])
        step_lengths.append(len(h5py_group["time"][tk]))
        for pk in parameters.keys():
            par_reduced[pk]["values"].append(parameters[pk][n])

    get_index = lambda s, ns: np.int64(s) * int(1e9) + np.int64(ns)
    index = np.hstack(
        [get_index(tds["seconds"], tds["nanoseconds"]) for tds in dsets_time]
    )

    data = da.concatenate([da.from_array(dsd) for dsd in dsets_data], axis=0)

    # return index, data, step_lengths, par_reduced
    return Array(
        data=data, index=index, step_lengths=step_lengths, parameter=par_reduced
    )
