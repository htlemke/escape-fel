from escape.utilities import MultipleRoiSelector, StepViewer, StepViewerP
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt


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

    def corr_ana_plot(self, referece, scanpar_name=None, axis=None):
        if not scanpar_name:
            names = list(self._scan.parameter.keys())
            scanpar_name = names[0]
        x = np.asarray(self._scan.parameter[scanpar_name]["values"]).ravel()
        corres = self._scan.correlation_analysis_to(referece)

        if not axis:
            axis = plt.gca()

        std = [tc[0] for tc in corres]
        std_fx = [tc[1] for tc in corres]

        ordercolors = ["b", "r"]
        for to, toc in zip([0, 1], ordercolors):
            axis.plot(
                x,
                [tc[to] for tc in std],
                toc + "--" + ".",
                label=f"poly. order:{to+1}; zero free",
            )
        for to, toc in zip([0, 1], ordercolors):
            axis.plot(
                x,
                [tc[to] for tc in std_fx],
                toc + "-" + "o",
                label=f"poly. order:{to+1}; zero fixed",
            )
        axis.set_xlabel(scanpar_name)
        axis.set_ylabel(self._scan._array.name)
        axis.legend()
        plt.tight_layout()

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

        data = self._scan._array
        sm = MultipleRoiSelector(
            data[data_selection].mean(axis=0).compute(),
            rois=rois,
            # callbacks_changeanyroi=[append_rois],
        )
        s = StepViewerP(data, sm, data_selection=data_selection)
        StepViewerP.rois = property(lambda self: self.output.rois)

        def append_rois():
            s.result = {}
            for nam, roi in sm.rois.items():
                rroi = [int(np.round(tr)) for tr in roi]
                s.result[nam] = self._scan._array[:, slice(*rroi[2:]), slice(*rroi[:2])]
                s.result[nam].name = nam

        sm.callbacks_changeanyroi = [append_rois]
        for rs in sm.roi_selectors:
            rs.callbacks_changeroi = [append_rois]
        append_rois()

        display(s)
        return s

    # def get_rectangular_roi(self,roidef={'test':(0,1,0,1)}):
    #     rroi = [int(np.round(tr)) for tr in roi]
    #     ret = self._array
