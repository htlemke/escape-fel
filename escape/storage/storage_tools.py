from escape.utilities import MultipleRoiSelector, StepViewer
import numpy as np
from IPython.display import display


class ArrayTools:
    def __init__(self, array):
        self._array = array

    def get_regions_of_interest_rectangular_2D(
        self,
        data_selection=slice(None, 100),
        rois={},
    ):
        def append_rois(s):
            s.result = {}
            for nam, roi in s.rois.items():
                roi = [int(np.round(tr)) for tr in roi]
                s.result[nam] = self._array[:, slice(*roi[2:]), slice(*roi[:2])]

        data = self._array[data_selection].mean(axis=0).compute()
        s = MultipleRoiSelector(data, rois=rois, callbacks_changeanyroi=[append_rois])
        display(s)
        return s


class ScanTools:
    def __init__(self, scan):
        self._scan = scan

    def view_step_averages(
        self,
        data_selection=slice(None, 100),
    ):

        data = self._scan._array
        s = StepViewer(data)
        display(s)
        return s
