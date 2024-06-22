
import xrayutilities as xu
from collections import deque
import jungfrau_utils




class THC_robot:
    def __init__(self,pixsz = 0.075,JF_ID='JF01T03V01'):
        self.pixsz = pixsz
        self.JF_ID=JF_ID
        self.angles = ['mu','eta','chi','phi','gamma','delta']
        self.qconv=xu.QConversion(('y+', 'x-', 'z+', 'x-'), ('y+', 'x-'), (0, 0, 1))
        
    @property
    def det_sz_px(self)
        jfh = jungfrau_utils.JFDataHandler(self.JF_ID)
        return np.asarray(jfh.get_shape_out())
    @property
    def beam_center_x_px(self):
        return det_sz_px[1]/2
    @property
    def beam_center_y_px(self):
        return det_sz_px[0]/2
    
    @property
    def beam_center_x_frac(self):
        return beam_center_x_px/det_sz_px[1]
    @property
    def beam_center_y_frac(self):
        return beam_center_y_px/det_sz_px[0]

    def plot_geom(self):
        pixsz_plot = pixsz*80
        det_sz_px_plot = np.round(det_sz_px *pixsz /pixsz_plot).astype(int)

        qconv.init_area(
            'x-','y-',
            beam_center_x_frac*det_sz_px_plot[1],
            beam_center_y_frac*det_sz_px_plot[0],
            *det_sz_px_plot[::-1],
            distance=300,
            pwidth1=pixsz_plot,pwidth2=pixsz_plot)