# from jungfrau_utils.corrections import apply_gain_pede_numba
from jungfrau_utils.data_handler import JFDataHandler
import h5py
from ..storage.storage import ArraySelector
from pathlib import Path
import numpy as np
import logging
from dask import array as da


def ispath(x):
    if type(x) is str or isinstance(x, Path):
        p = Path(x)
        if p.exists():
            return p


def jf_correct_obj(
    array,
    jf_id=None,
    cor_gain_dark_mask=True,
    cor_tile_gaps=True,
    cor_geometry=True,
    comp_parallel=False,
    gain_file=None,
    dark_file=None,
    mask=None,
    module_map=None,
    **kwargs,
):
    h = JFDataHandler(jf_id)
    h.gain_file = gain_file
    h.pedestal_file = dark_file
    if mask:
        h.pixel_mask = mask
    if module_map:
        h.module_map = module_map

    return h


def jf_correct(
    array,
    jf_id=None,
    cor_gain_dark_mask=True,
    cor_tile_gaps=True,
    cor_geometry=True,
    cor_mask=True,
    comp_parallel=False,
    gain_file=None,
    dark_file=None,
    mask=None,
    module_map=None,
    double_pixels="interp",
    use_numpy=True,
    threshold=None,
    **kwargs,
):
    if use_numpy:

        with h5py.File(gain_file, "r") as fh:
            gain = fh["gains"][:]

        with h5py.File(dark_file, "r") as fh:
            pedestal = fh["gains"][:]

        # figure()
        data_corr = array.map_index_blocks(
            apply_gain_pede_np, gain, pedestal, dtype=np.float64
        )
        if threshold:
            return data_corr.map_index_blocks(
                apply_threshold, threshold=threshold, dtype=np.float64
            )
        else:
            return data_corr
    else:
        h = JFDataHandler(jf_id)
        h.gain_file = gain_file
        h.pedestal_file = dark_file
        if mask:
            h.pixel_mask = mask
        if module_map:
            h.module_map = module_map

        def proc_and_mask(*args, **kwargs):
            o = h.process(*args, **kwargs)
            o[
                ~np.broadcast_to(
                    h.get_pixel_mask(
                        gap_pixels=cor_tile_gaps,
                        double_pixels=double_pixels,
                        geometry=cor_geometry,
                    ),
                    o.shape,
                )
            ] = np.nan
            return o

        return array.map_index_blocks(
            proc_and_mask,
            conversion=cor_gain_dark_mask,
            gap_pixels=cor_tile_gaps,
            geometry=cor_geometry,
            mask=cor_mask,
            parallel=comp_parallel,
            new_element_size=h.get_shape_out(
                gap_pixels=cor_tile_gaps, geometry=cor_geometry
            ),
            dtype=float,
            **kwargs,
        )


def apply_gain_pede_np(image, G, P, pixel_mask=None, mask_value=np.nan):
    # gain and pedestal correction
    mask = int("0b" + 14 * "1", 2)
    mask2 = int("0b" + 2 * "1", 2)

    gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
    data = np.bitwise_and(image, mask)

    m1 = gain_mask == 0
    m2 = gain_mask == 1
    m3 = gain_mask >= 2
    if G is not None:
        g = m1 * G[0] + m2 * G[1] + m3 * G[2]
    else:
        g = np.ones(data.shape, dtype=np.float32)
    if P is not None:
        p = m1 * P[0] + m2 * P[1] + m3 * P[2]
    else:
        p = np.zeros(data.shape, dtype=np.float32)
    res = np.divide(data - p, g)

    if pixel_mask is not None:
        # dv,mv = np.broadcast_arrays(res,pixel_mask) # seems not necessary
        mv = pixel_mask
        if isinstance(image, da.Array):
            mv = da.from_array(mv, chunks=image.chunks[-2:])
        #            if len(image.shape)==3:
        #                res[mv!=0] = mask_value
        #            else:
        res[mv != 0] = mask_value
        res = da.nan_to_num(res, 0)
    return res


def apply_threshold(data, threshold=0):
    data[data < threshold] = 0
    return data


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
