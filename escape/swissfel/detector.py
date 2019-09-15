from jungfrau_utils.corrections import apply_gain_pede_numba
import h5py
from ..storage.storage import ArraySelector
from pathlib import Path
import numpy as np

def ispath(x):
    if type(x) is str or isinstance(x,Path):
        p = Path(x)
        if p.exists():
            return p

def _apply_gain_pede_numba_stack(images,**kwargs):
    result = np.zeros_like(images,dtype=float)
    for n,i in enumerate(images):
        result[n] = apply_gain_pede_numba(i,**kwargs)
    return result

def correct_gain_dark_mask(array,gain=None,dark=None,mask=None):
    gainpath = ispath(gain)
    if gainpath:
        gain = h5py.File(gainpath,'r')['gains'][...]
    darkpath = ispath(dark)
    if darkpath:
        dark = h5py.File(darkpath,'r')['gains'][...]
    maskpath = ispath(mask)
    if maskpath:
        mask = h5py.File(maskpath,'r')['pixel_mask'][...]
    return array.map_event_blocks(_apply_gain_pede_numba_stack,G=ArraySelector(gain,(1,2)),P=ArraySelector(dark,(1,2)),pixel_mask=ArraySelector(mask,(1,2)))
