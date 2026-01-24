from numbers import Number
from escape import utilities
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
        show=True,
    ):
        def append_rois(s):
            s.result = {}
            for nam, roi in s.rois.items():
                roi = [int(np.round(tr)) for tr in roi]
                s.result[nam] = self._array[:, slice(*roi[2:]), slice(*roi[:2])]

        data = self._array[data_selection].mean(axis=0).compute()
        s = MultipleRoiSelector(data, rois=rois, callbacks_changeanyroi=[append_rois])
        if show:
            display(s)
        return s
    
    def compare_to_reference(
        self, 
        is_reference, 
        N_agg_ref=100, 
        weights=None, 
        cmp_type='ratio',
        axis_survey=None,
    ):  
        array = self._array
        resort = is_reference.get_index_array(N_index_aggregation=N_agg_ref) # new sorting according to pulse id (/ real time) bins for reference taking
        array = resort.categorize(array)
        array = array.scan.tools.has_N_refsig(is_reference) # filter out steps that don't have any reference or signal 

        array_sig = array[~is_reference]
        array_ref = array[is_reference]
        
        if cmp_type =='ratio':
            array_cmp = array_sig.scan / array_ref.scan.weighted_stat(weights)[0]
        if cmp_type =='difference':
            array_cmp = array_sig.scan - array_ref.scan.weighted_stat(weights)[0]

        if axis_survey:
            array_ref.plot(axis=axis_survey,ms=.3,label="Reference, single pulse")
            # array_sig.plot(axis=axis.survey,ms=.3,label="Signal")

            array_ref.plot(axis=axis_survey,ms=.3, label='ref (off) single plse')
            array_ref.scan.plot(axis=axis_survey, label='Reference, aggregated')
            array_sig.scan.plot(axis=axis_survey, label='Signal, aggregated')

        return array_cmp
    
    def timetool_binning(self,timetool, time_vec=None, time_bins=None, tbinsize=20e-15):
        array = self._array

        if not time_vec: 
            t = timetool.scan.par_steps.iloc[:,0] # timevec scan

        t_tt = timetool.scan + t # taking the time of the step and adding the time tool delay for each shot - the real measured delay 
        
        tt_med = timetool.nanmedian() # try to get the average tt values as median, for binning. 

        if isinstance(time_bins,Number):
            time_bins = np.arange(
                utilities.roundto(np.nanmin(t)+tt_med,time_bins),
                utilities.roundto(np.nanmax(t)+tt_med,time_bins)+time_bins,
                tbinsize)
        t_tt_binned = t_tt.digitize(time_bins)
        
        array_tt = (t_tt_binned).categorize(array)
        
        return array_tt

#>>>>>>>>>>>>>>>


def timetool_binning(array,timetool, time_vec=None, time_bins=None, tbinsize=20e-15):

    if not time_vec: 
        t = timetool.scan.par_steps.iloc[:,0] # timevec scan

    t_tt = timetool.scan + t # taking the time of the step and adding the time tool delay for each shot - the real measured delay 
    
    tt_med = timetool.nanmedian() # try to get the average tt values as median, for binning. 

    if isinstance(time_bins,Number):
        time_bins = np.arange(
            utilities.roundto(np.nanmin(t)+tt_med,time_bins),
            utilities.roundto(np.nanmax(t)+tt_med,time_bins)+time_bins,
            tbinsize)
    t_tt_binned = t_tt.digitize(time_bins)
    
    array_tt = (t_tt_binned).categorize(array)
    
    return array_tt


#<<<<<<<<<<<<<<<



        # return (N_sig <= len(self._array[is_sig])) and (N_ref <= len(self._array[is_ref]))


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
    

    def has_N_refsig(self,is_ref,is_sig=None,N_ref = 1, N_sig=1):
        if is_sig is not None:
            sel = is_ref | is_sig
            #TODO
        
        is_ref = self._scan._array.categorize(is_ref).compute()

        valid_steps = []
        for n,step in enumerate(is_ref.scan):
            if (N_ref <= sum(step.data)) and (N_sig <= sum(~step.data)):
                valid_steps.append(n)

        return self._scan[valid_steps]
    

        


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
        show=True,
    ):
        def append_rois(s):
            s.result = {}
            for nam, roi in s.rois.items():
                roi = [int(np.round(tr)) for tr in roi]
                s.result[nam] = self._array[:, slice(*roi[2:]), slice(*roi[:2])]

        if show:
            data = self._scan._array
            sm = MultipleRoiSelector(
                data[data_selection].mean(axis=0).compute(),
                rois=rois,
                callbacks_changeanyroi=[append_rois],
            )
            s = StepViewerP(data, sm, data_selection=data_selection)
            StepViewerP.rois = property(lambda self: self.output.rois)

            def append_rois():
                s.result = {}
                for nam, roi in sm.rois.items():
                    rroi = [int(np.round(tr)) for tr in roi]
                    s.result[nam] = self._scan._array[
                        :, slice(*rroi[2:]), slice(*rroi[:2])
                    ]
                    s.result[nam].name = nam

            sm.callbacks_changeanyroi = [append_rois]
            for rs in sm.roi_selectors:
                rs.callbacks_changeroi = [append_rois]
            append_rois()
            display(s)
        else:

            class Dummy:
                pass

            s = Dummy()
            s.rois = {}
            s.result = {}
            for nam, troi in rois.items():
                roi = [int(np.round(tr)) for tr in troi]
                s.rois[nam] = roi
                s.result[nam] = self._scan._array[:, slice(*roi[2:]), slice(*roi[:2])]

        return s

    # def get_rectangular_roi(self,roidef={'test':(0,1,0,1)}):
    #     rroi = [int(np.round(tr)) for tr in roi]
    #     ret = self._array
