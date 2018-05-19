import json
import pathlib
import utilities
import h5py
from dask import array


def readScanEcoJson_v01(file_name_json):
    p = pathlib.Path(file_name_json)
    assert p.is_file(), 'Input string does not describe a valid file path.'
    with p.open(mode='r') as f:
        s = json.load(f)
    assert len(s['scan_files']) == len(s['scan_values']), 'number of files and scan values don\'t match in {}'.format(file_name_json)
    assert len(s['scan_files']) == len(s['scan_readbacks']), 'number of files and scan readbacks don\'t match in {}'.format(file_name_json)
    return s,p

def parseScanEco_v01(file_name_json,search_paths=['./','./scan_data/','../scan_data']):
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


    for datasets in datasets_scan:
        tnames = set(datasets.keys())
        

    
    return s,datasets_scan

            
def attachToDict(key,value,dictionary):
    if key in dictionary.keys():


           








    




