import matplotlib.pyplot as plt
from time import sleep
import ipywidgets as widgets
import numpy as np


class GinputNB:
    def __init__(self, fig=None):
        self.x = []
        self.y = []
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0

    def get(self, Npts=-1.0, plotspec="rd-"):
        self.mcid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.kcid = self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        if plotspec:
            self.line = plt.plot(self.x, self.y, plotspec)[0]
        self.collecting = Npts
        return self

    def onclick(self, event):
        if self.collecting == 0:
            return
        if event.button == 1:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            if self.collecting > 0:
                self.collecting -= 1
        else:
            self.x.pop()
            self.y.pop()
            if self.collecting > 0:
                self.collecting += 1
        if self.line:
            self.line.set_data(self.x, self.y)
            if self.collecting == 0:
                self.stop()
            self.line.figure.canvas.draw()

    def stop(self):
        self.fig.canvas.mpl_disconnect(self.mcid)
        self.fig.canvas.mpl_disconnect(self.kcid)
        self.collecting = 0
        donetx = plt.text(self.x[-1], self.y[-1], "Done!", color=[0, 1, 0])
        self.line.figure.canvas.draw()
        sleep(2)
        donetx.remove()

    def onkey(self, event):
        if event.key == "d":
            self.stop()


import ipywidgets as widgets
from matplotlib.widgets import RectangleSelector

        
def make_box_layout():
     return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
     )

class RectangleSelectNB:
    def __init__(self, fig=None, ax=None, ax_roi=None):
        if not fig:
            fig = plt.gcf()
        self.fig = fig
        #         self.evt= Event()
        self.collecting = 0
        if not ax:
            ax = plt.gca()
        self.ax = ax
        self.ax_roi = ax_roi

        self.selector = RectangleSelector(ax, self.line_select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='data',
                                       interactive=True)
        fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        self.selector.set_active(False)
        
    def toggle_selector(self,event):
        if event.key == 't':
            if self.selector.active:
                print(' RectangleSelector deactivated.')
                self.selector.set_active(False)
                # self.rectangle = Rectangle([10,10],20,20)
                
            else:
                print(' RectangleSelector activated.')
                self.selector.set_active(True)  
        
    def line_select_callback(self, eclick, erelease):
#         if self.selector.active_handle:
#             self.selector.set_active(True)
#         else:
#             self.selector.set_active(False)
#         'eclick and erelease are the press and release events'
#         x1, y1 = eclick.xdata, eclick.ydata
#         x2, y2 = erelease.xdata, erelease.ydata
#         print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
#         print(" The button you used were: %s %s" % (eclick.button, erelease.button))
        if self.ax_roi:
            if self.ax_roi.get_images():
                i = self.ax_roi.get_images()[0]
                i.set_data(self.get_image_roi_data())
            else:
                self.ax_roi.imshow(self.get_image_roi_data())
    
    
    def get_image_roi_data(self):
        i = self.ax.get_images()[0]
        return i.get_array()[self.get_image_slice_selection()]
        

    def get_image_slice_selection(self):
        return slice(int(np.round(self.ymin)),int(np.round(self.ymax))), slice(int(np.round(self.xmin)),int(np.round(self.xmax)))
    
    @property
    def xmin(self):
        return self.selector.extents[0]
    @property
    def xmax(self):
        return self.selector.extents[1]
    @property
    def ymin(self):
        return self.selector.extents[2]
    @property
    def ymax(self):
        return self.selector.extents[3]
    
    
    
class MultipleRoiSelector(widgets.HBox):
    def __init__(self,data,name=''):
        # super().__init__(layout=widgets.Layout(width='90%'))
        super().__init__()
        self.data = data
        self.name = name
        self.roi_selectors = []
        self._tabs_rois = widgets.Tab()
        def f(x):
            self.set_roi_selection_active(self,i=None)
        self._tabs_rois.observe(f,names='selected_index')

        
        self._select_buttons = []
        self.debug = widgets.Output()

        
        
        self._add_roi_button = widgets.Button(
            description='Add roi',
            disabled=False,button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            # icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
        
        
        self._cmap_range = widgets.FloatRangeSlider(
            value=[self.data.min(),self.data.max()],
            description='Colormap range',
            min=self.data.min(),
            max=self.data.max(),
            step=self.data.ptp()/500,
            continuous_upfate=False,
            disabled=False)
        
            
            
        self._add_roi_button.on_click(self.add_roi)
        self._roi_titles = []
        

        self.axs_rois=[]
        self.figs_rois = []
        self.create_data_plot()
        
        
        self.children = [widgets.VBox([
            widgets.HBox([self._clim_slider,self._add_roi_button]),
            self._output_data
        ]),
                         self._tabs_rois]
        self.layout = make_box_layout()
        
    def create_data_plot(self):
        self._output_data = widgets.Output()
        with self._output_data:
            plt.close(self.name)
            fig,ax = plt.subplots(num=self.name
                # constrained_layout=True
            )
            self.fig_data = fig
            self.ax_data = ax
            ih = self.ax_data.imshow(self.data)
            plt.colorbar(mappable=ih)
            plt.tight_layout()
            plt.show(self.fig_data)
        
        mn = self.data.min()
        mx = self.data.max()
        ptp = mx-mn
        
        self._clim_slider = widgets.FloatRangeSlider(
                                value=[mn,mx],
                                min=mn,
                                max=mx,
                                step=ptp/200,
                                description='Colormap range:',
                                disabled=False,
                                continuous_update=True,
                                orientation='horizontal',
                                readout=True,
                                # readout_format='.1f',
                                )
        # widgets.interact(lambda val:self.set_clim(*val),val=self._clim_slider)
        self._clim_slider.observe(lambda val:self.set_clim(*val['new']),names='value')
        
    def set_clim(self,*args,**kwargs):
        
        with self.debug:
            print(*args,**kwargs)
        i = self.ax_data.get_images()[0]
        i.set_clim(*args,**kwargs)
        for n,ax in enumerate(self.axs_rois):
            print(n)
            if ax.get_images():
                i = ax.get_images()[0]
                i.set_clim(*args,**kwargs)
        
        
    
    def add_roi_plot(self):
        ti = len(self.roi_selectors)
        op = widgets.Output()
        with op:
            tfig = plt.figure(constrained_layout=True)
            self.axs_rois.append(tfig.add_subplot())
            plt.show(tfig)
            
        self._select_buttons.append(
            widgets.Button(
                description='Select',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Click me',
                # icon='check' # (FontAwesome names without the `fa-` prefix)
                )
        )
        self._roi_titles.append(widgets.Text(value=f'roi{ti}', placeholder='ROI title', description='Name:',disabled=False))
        self._tabs_rois.children += (widgets.VBox([widgets.HBox([self._select_buttons[-1],self._roi_titles[-1]]),op]),)
        # self._tabs_rois.set_title(len(self._tabs_rois.children)-1,self._roi_title_input.value)
        def update_tab_title(x):
            self._tabs_rois.set_title(ti,x['new'])
        self._tabs_rois.set_title(ti,self._roi_titles[ti].value)
        self._roi_titles[ti].observe(update_tab_title,names='value')
        
        
        
    def set_roi_selection_active(self,i=None):
        for rs in self.roi_selectors:
            rs.selector.set_active(False)
        if not i==None:
            self.roi_selectors[i].selector.set_active(True)
        
    def get_roi_selection_active(self):
        o = []
        for rs in self.roi_selectors:
            o.append(rs.selector.active)
        return o
            
    
    def add_roi(self,dum):
        # self.roi_selectors.append('test')
        ti = len(self.roi_selectors)
        self.add_roi_plot()
        self.roi_selectors.append(RectangleSelectNB(fig=self.fig_data, ax=self.ax_data, ax_roi=self.axs_rois[-1]))
        self._select_buttons[-1].on_click(lambda dum:self.set_roi_selection_active(ti))


def nfigure(name='no name'):
    plt.close(name)
    return plt.figure(name)

