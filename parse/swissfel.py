import json
import pathlib
from . import utilities
import h5py
from dask import array as da
from .. import Array, Scan
import numpy as np
from copy import copy

def readScanEcoJson_v01(file_name_json):
    p = pathlib.Path(file_name_json)
    assert p.is_file(), 'Input string does not describe a valid file path.'
    with p.open(mode='r') as f:
        s = json.load(f)
    assert len(s['scan_files']) == len(s['scan_values']), 'number of files and scan values don\'t match in {}'.format(file_name_json)
    assert len(s['scan_files']) == len(s['scan_readbacks']), 'number of files and scan readbacks don\'t match in {}'.format(file_name_json)
    return s,p

def parseScanEco_v01(file_name_json,search_paths=['./','./scan_data/','../scan_data'],memlimit_0D_MB=5, memlimit_mD_MB=10, createEscArrays=True):
    """Data parser assuming eco-written files from pilot phase 1"""
    s,scan_info_filepath = readScanEcoJson_v01(file_name_json)
    lastpath = None
    searchpaths = None

    datasets_scan = []
    for files in s['scan_files']:
        datasets = {}
        for f in files:
            fp = pathlib.Path(f)
            fn = pathlib.Path(fp.name)
            if not searchpaths:
                searchpaths = [fp.parent]+\
                    [scan_info_filepath.parent/pathlib.Path(tp.format(fp.parent.name)) \
                    for tp in search_paths]
            for path in searchpaths:
                file_path = path/fn
                if file_path.is_file():
                    if not lastpath:
                        lastpath = path
                        searchpaths.insert(0,path)
                    break
            #assert file_path.is_file(), 'Could not find file {} '.format(fn)
            try:
                fh = h5py.File(file_path.resolve(),mode='r')
                datasets.update(utilities.findItemnamesGroups(fh,['data','pulse_id']))
            except:
                print(f'WARNING: could not read {file_path.absolute().as_posix()}.')
        datasets_scan.append(datasets)

    names = set()
    dstores = {}

    for stepNo,(datasets,scan_values,scan_readbacks,scan_step_info) \
            in enumerate(zip(datasets_scan,s['scan_values'],s['scan_readbacks'],s['scan_step_info'])):
        tnames = set(datasets.keys())
        newnames = tnames.difference(names)
        oldnames = names.intersection(tnames)
        for name in newnames:
            if datasets[name][0].size ==0:
                print("Found empty dataset in {} in cycle {}".format(name,stepNo))
            else:
                size_data = np.dtype(datasets[name][0].dtype).itemsize * datasets[name][0].size /1024**2
                size_element = np.dtype(datasets[name][0].dtype).itemsize * np.prod(datasets[name][0].shape[1:]) /1024**2
                chunk_size = list(datasets[name][0].chunks)
                if chunk_size[0] == 1:
                    chunk_size[0] = int(memlimit_mD_MB//size_element)
                dstores[name] = {}
                dstores[name]['scan'] = Scan(parameter_names=s['scan_parameters']['name'], parameter_Ids=s['scan_parameters']['Id'])
                # dirty hack for inconsitency in writer
                if len(scan_readbacks) > len(dstores[name]['scan']._parameter_names):
                    scan_readbacks = scan_readbacks[:len(dstores[name]['scan']._parameter_names)]
                dstores[name]['scan']._append(copy(scan_values), copy(scan_readbacks), scan_step_info=copy(scan_step_info))
                dstores[name]['data'] = []
                dstores[name]['data'].append(datasets[name][0])
                dstores[name]['data_chunks'] = chunk_size
                dstores[name]['eventIds'] = []
                dstores[name]['eventIds'].append(datasets[name][1])
                dstores[name]['stepLengths'] = []
                dstores[name]['stepLengths'].append(len(datasets[name][0]))
                names.add(name)
        for name in oldnames:
            if datasets[name][0].size ==0:
                print("Found empty dataset in {} in cycle {}".format(name,stepNo))
            else:
                # dirty hack for inconsitency in writer
                if len(scan_readbacks) > len(dstores[name]['scan']._parameter_names):
                    scan_readbacks = scan_readbacks[:len(dstores[name]['scan']._parameter_names)]
                dstores[name]['scan']._append(copy(scan_values), copy(scan_readbacks), scan_step_info=copy(scan_step_info))
                dstores[name]['data'].append(datasets[name][0])
                dstores[name]['eventIds'].append(datasets[name][1])
                dstores[name]['stepLengths'].append(len(datasets[name][0]))
    if createEscArrays:
        escArrays = {}
        containers = {}
        for name,dat in dstores.items():
            containers[name] = LazyContainer(dat)
            escArrays[name] = Array(containers[name].get_data,\
                    eventIds=containers[name].get_eventIds,stepLengths=dat['stepLengths'],scan=dat['scan'])
        return escArrays
    else:
        return dstores


class LazyContainer:
    def __init__(self,dat):
        self.dat = dat
    def get_data(self):
        return da.concatenate([da.from_array(td,chunks=self.dat['data_chunks']) for td in self.dat['data']])
    def get_eventIds(self):
        return np.concatenate([td[...].ravel() for td in self.dat['eventIds']])

            


           








    




