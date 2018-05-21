import json
import pathlib
from . import utilities
import h5py
from dask import array as da
from .. import Array, Scan


def readScanEcoJson_v01(file_name_json):
    p = pathlib.Path(file_name_json)
    assert p.is_file(), 'Input string does not describe a valid file path.'
    with p.open(mode='r') as f:
        s = json.load(f)
    assert len(s['scan_files']) == len(s['scan_values']), 'number of files and scan values don\'t match in {}'.format(file_name_json)
    assert len(s['scan_files']) == len(s['scan_readbacks']), 'number of files and scan readbacks don\'t match in {}'.format(file_name_json)
    return s,p

def parseScanEco_v01(file_name_json,search_paths=['./','./scan_data/','../scan_data'],memlimit_0D_MB=5, memlimit_mD_MB=10):
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
            assert file_path.is_file(), 'Could not find file {} '.format(fn)
            fh = h5py.File(file_path.resolve(),mode='r')
            datasets.update(utilities.findItemnamesGroups(fh,['data','pulse_id']))
        datasets_scan.append(datasets)

    names = set()
    dstores = {}

    for stepNo,(datasets,scan_values,scan_readbacks,scan_step_info) \
            in enumerate(datasets_scan,s['scan_values'],s['scan_readbacks'],s['scan_step_info']):
        tnames = set(datasets.keys())
        newnames = tnames.difference(names)
        oldnames = names.intersection(tnames)
        for name in newnames:
            if datasets[name][0].size ==0:
                print("Found empty dataset in {} in cycle {}".format(name,stepNo))
            else:
                scan = Scan(parameter_names=s['parameter_names'], parameter_Ids=s['parameter_Ids'])
                size_data = np.dtype(datasets[name][0].dtype).itemsize * datasets[name][0].size /1024**2
                size_element = np.dtype(datasets[name][0].dtype).itemsize * np.prod(datasets[name][0].shape) /1024**2
                chunk_size = dataset[name][0].chunks
                if chunk_size[0] == 1:
                    chunk_size[0] = memlimit_mD_MB//size_element
                dstores[name] = {}
                dstores[name]['scan'] = scan
                dstores[name]['data'] = []
                dstores[name]['data'].append(datasets[name][0])
                dstores[name]['data_chunks'] = chunk_size
                dstores[name]['eventIds'] = []
                dstores[name]['eventIds'].append(datasets[name][1])
                dstores[name]['stepLengths'] = []
                dstores[name]['stepLemgths'].append(len(datasets[name][0]))
                #dstores[name] = [da.from_array(datasets[name][0],chunks=datasets[name][0].chunks), \
                #       da.from_array(datasets[name][1], chunks=datasets[name][1].chunks),[len(datasets[name][0])]]
                names.add(name)
        for name in oldnames:
            if datasets[name][0].size ==0:
                print("Found empty dataset in {} in cycle {}".format(name,stepNo))
            else:
                dstores[name]['scan'].append()
                dstores[name]['data'].append(datasets[name][0])
                dstores[name]['eventIds'].append(datasets[name][1])
                dstores[name]['stepLemgths'].append(len(datasets[name][0]))
#                dstores[name][0] = da.concatenate( [ dstores[name][0],
#                    da.from_array(datasets[name][0],chunks=datasets[name][0].chunks) ], axis=0)
#                dstores[name][1] = da.concatenate( [ dstores[name][1],
#                    da.from_array(datasets[name][1],chunks=datasets[name][1].chunks) ], axis=0)
#                dstores[name][2].append(len(datasets[name][0]))

    escArrays = {}
    for name,(data,eventId,steplengths) in dstores.items():
        escArrays[name] = Array(data,eventIds=eventId,stepLengths=steplengths)






        

    
    return escArrays

            


           








    




