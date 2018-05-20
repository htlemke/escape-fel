import json
import pathlib
from . import utilities
import h5py
from dask import array as da
from .. import Array


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
                    [scan_info_filepath.parent/pathlib.Path(tp) \
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
    for stepNo,datasets in enumerate(datasets_scan,s['scan_values','scan_readbacks',s['scan_step_info']]):
        tnames = set(datasets.keys())
        newnames = tnames.difference(names)
        oldnames = names.intersection(tnames)
        for name in newnames:
            if datasets[name][0].size ==0:
                print("Found empty dataset in {} in cycle {}".format(name,stepNo))
            else:
                dstores[name] = [da.from_array(datasets[name][0],chunks=datasets[name][0].chunks), \
                                da.from_array(datasets[name][1], chunks=datasets[name][1].chunks),[len(datasets[name][0])]]
                names.add(name)
        for name in oldnames:
            if datasets[name][0].size ==0:
                print("Found empty dataset in {} in cycle {}".format(name,stepNo))
            else:
                dstores[name][0] = da.concatenate( [ dstores[name][0],
                    da.from_array(datasets[name][0],chunks=datasets[name][0].chunks) ], axis=0)
                dstores[name][1] = da.concatenate( [ dstores[name][1],
                    da.from_array(datasets[name][1],chunks=datasets[name][1].chunks) ], axis=0)
                dstores[name][2].append(len(datasets[name][0]))

    escArrays = {}
    for name,(data,eventId,steplengths) in dstores.items():
        escArrays[name] = Array(data,eventIds=eventId,stepLengths=steplengths)






        

    
    return escArrays

            
def attachToDict(key,value,dictionary):
    if key in dictionary.keys():
        bof


           








    




