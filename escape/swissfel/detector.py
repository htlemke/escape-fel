from jungfrau_utils.corrections import apply_gain_pede_numba
from jungfrau_utils.data_handler import JFDataHandler
import h5py
from ..storage.storage import ArraySelector
from pathlib import Path
import numpy as np
import logging


# def ispath(x):
    # if type(x) is str or isinstance(x, Path):
        # p = Path(x)
        # if p.exists():
            # return p


# def _apply_gain_pede_numba_stack(images, **kwargs):
    # result = np.zeros_like(images, dtype=float)
    # for n, i in enumerate(images):
        # result[n] = apply_gain_pede_numba(i, **kwargs)
    # return result


# def correct_gain_dark_mask(array, gain=None, dark=None, mask=None):
    # gainpath = ispath(gain)
    # if gainpath:
        # gain = h5py.File(gainpath, "r")["gains"][...]
    # darkpath = ispath(dark)
    # if darkpath:
        # dark = h5py.File(darkpath, "r")["gains"][...]
    # maskpath = ispath(mask)
    # if maskpath:
        # mask = h5py.File(maskpath, "r")["pixel_mask"][...]
    # return array.map_index_blocks(
        # _apply_gain_pede_numba_stack,
        # G=ArraySelector(gain, (1, 2)),
        # P=ArraySelector(dark, (1, 2)),
        # pixel_mask=ArraySelector(mask, (1, 2)),
    # )

def jf_correct(array,
        cor_gain_dark_mask=True, 
       cor_tile_gaps = True,
        cor_geometry = True,
        comp_parallel = False,
        jf_id=None, 
        gain_file=None, 
        dark_file=None, 
        mask=None, 
        module_map=None,
        **kwargs
        ):
    h = JFDataHandler(jf_id)
    h.gain_file = gain_file
    h.pedestal_file = dark_file
    if mask:
        h.pixel_mask=mask
    if module_map:
        h.module_map = module_map

    def proc_and_mask(*args,**kwargs):
        o = h.process(*args,**kwargs)
        o[np.broadcast_to(h.get_pixel_mask(cor_tile_gaps,cor_geometry),o.shape)]=np.nan
        return o

    
    return array.map_index_blocks(
        proc_and_mask,
        conversion=cor_gain_dark_mask,
        gap_pixels=cor_tile_gaps,
        geometry=cor_geometry,
        parallel=comp_parallel,
        new_element_size=h.get_shape_out(cor_tile_gaps,cor_geometry),
        **kwargs
    )
    
    




# class JfCorrector:
#     def __init__(self,jf_id,gain=None,dark=None,mask=None):
#         self.id = jf_id
#         self.gain = gain
#         self.dark = dark
#         self.mask = mask
           
        
# import h5py
# from jungfrau_utils import apply_gain_pede, apply_geometry
# h5py.enable_ipython_completer()
# from escape.storage import escaped,matchArrays

# def load_calib(fina_gain,fina_ped):
#     with h5py.File(fina_gain,'r') as f:
#          gains = f['gains'].value
#     with h5py.File(fina_ped,'r') as f:
#          pede = f['gains'].value
#          noise = f['gainsRMS'].value
#          mask = f['pixel_mask'].value
#     return gains, pede, mask

# def correct_gain_pedestal(array,gain,pedestal,mask=None):
#     def gaincorrect(data,gain,ped,mask=None):
#         return np.asarray([apply_gain_pede(td,G=gain, P=ped, pixel_mask=mask) for td in data])
#     return array.map_event_blocks(gaincorrect,gain,pedestal,mask=mask,dtype=float)


# def noisefloor_threshold(array,threshold, fill=np.nan):
#     def threshold_data(d,threshold,fill = np.nan):
#         d[d<threshold] = fill
#         return d
#     return array.map_event_blocks(threshold_data,threshold,fill=fill)


# cal_i0 = load_calib('/sf/bernina/config/jungfrau/gainMaps/JF03T01V01/gains.h5','/sf/bernina/config/exp/19h_teitelbaum/res/JF_pedestals/pedestal_20190927_0822.JF03T01V01.res.h5')
# i0 = correct_gain_pedestal(data['JF03T01V01'],*cal_i0[:2],mask=cal_i0[2])
# i0 = noisefloor_threshold(i0,4.5)
# # i0 = i0[:,:512,:512].nansum(axis=(1,2))[:,None]

# cal_i1 = list(load_calib('/sf/bernina/config/jungfrau/gainMaps/JF07T32V01/gains.h5','/sf/bernina/config/exp/19h_teitelbaum/res/JF_pedestals/pedestal_20190927_0822.JF07T32V01.res.h5'))
