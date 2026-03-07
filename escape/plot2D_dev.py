
from matplotlib import colors, pyplot as plt
from matplotlib.widgets import SpanSelector
import numpy as np


def plot2D(
    x,
    y,
    C,
    *args,
    axis=None,
    diverging=False,
    log_colors=False,
    pars_symlognorm=dict(
        linthresh=1.0,
        linscale=1.0,
        vmin="auto",
        vmax="auto",
        base=10,
    ),
    colorbar_with_histogram=False,
    **kwargs,
):
    """Helper function to create a fals color 3D plot using matplotlib pcolormesh.

    Args:
        x (array-like 1d): may be replaced "auto" for bin number
        y (array-like 1d): may be replaced "auto" for bin number
        C (array-like 2d): [description]
        axis (matplotlib axis): [description]. Defaults to None.
        diverging (bool, optional): Use centered colormap. Defaults to False.
        log_colors (bool, optional): Use logarithmic color scale. Defaults to False.
        pars_symlognorm (dict, optional): parameters for SymLogNorm if diverging and log_colors are True.
            Defaults to  dict( linthresh=1.0, linscale=1.0, vmin='auto', vmax='auto', base=10,)
        kwargs: additional arguments passed to pcolormesh
        Returns: pcolormesh object"""

    if axis is None:
        axis = kwargs.pop("ax", None)

    def bin_array(arr):
        arr = np.asarray(arr)
        return np.hstack([arr - np.diff(arr)[0] / 2, arr[-1] + np.diff(arr)[-1] / 2])

    C = np.asarray(C)

    if type(x) is str and x == "auto":
        x = np.arange(C.shape[1])
    if type(y) is str and y == "auto":
        y = np.arange(C.shape[0])

    Xp, Yp = np.meshgrid(bin_array(x), bin_array(y))
    
    if axis is None:
        axis = plt.gca()
        plt.sca(axis)

    if colorbar_with_histogram:
        spec = axis.get_subplotspec()
        fig = axis.get_figure()
        axis.remove()
        gs = spec.subgridspec(1, 3, width_ratios=[15, 0.6, 2])
        # gs = gridspec.GridSpec(
        axis = fig.add_subplot(gs[0])
        ax_cbar = fig.add_subplot(gs[1])
        ax_hist = fig.add_subplot(gs[2])


    if axis:
        plt.sca(axis)
    
    
    if diverging:
        if log_colors:
            if pars_symlognorm.get("vmin", "auto") == "auto":
                vminmax = np.max(
                    np.ceil(np.log10(np.abs(np.nanmin(C)))).astype(int),
                    np.ceil(np.log10(np.abs(np.nanmax(C)))).astype(int),
                )
            out = plt.pcolormesh(
                Xp,
                Yp,
                C,
                *args,
                **kwargs,
                cmap=kwargs.get("cmap", "coolwarm"),
                norm=colors.SymLogNorm(**pars_symlognorm),
            )
        else:
            out = plt.pcolormesh(
                Xp,
                Yp,
                C,
                *args,
                **kwargs,
                cmap=kwargs.get("cmap", "coolwarm"),
                norm=colors.CenteredNorm(),
            )
    else:
        if log_colors:
            out = plt.pcolormesh(
                Xp,
                Yp,
                C,
                *args,
                **kwargs,
                norm=colors.LogNorm(
                    vmin=np.nanmin(C[np.nonzero(C)]), vmax=np.nanmax(C)
                ),
            )
        else:
            out = plt.pcolormesh(Xp, Yp, C, *args, **kwargs)
    try:
        plt.xlabel(x.name)
    except:
        pass
    try:
        plt.ylabel(y.name)
    except:
        pass

    if colorbar_with_histogram:
        fig.colorbar(out, cax=ax_cbar)

        counts, bins, patches = ax_hist.hist(C.flatten(), bins=100, 
                                            orientation='horizontal', color='lightgray')
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax_hist.yaxis.tick_right()
        # ax_hist.set_title("Contrast Selector")


        def update_elements(vmin, vmax):
            out.set_clim(vmin, vmax)
            for center, patch in zip(bin_centers, patches):
                if vmin <= center <= vmax:
                    norm_val = (center - vmin) / (vmax - vmin) if vmax != vmin else 0
                    patch.set_facecolor(out.cmap(norm_val))
                    patch.set_alpha(1.0)
                else:
                    patch.set_facecolor('lightgray')
                    patch.set_alpha(0.5)
            
            if ax_cbar.get_ylim() != (vmin, vmax):
                ax_cbar.set_ylim(vmin, vmax)
            
            fig.canvas.draw_idle()

        span = SpanSelector(ax_hist, update_elements, 'vertical', useblit=False,
                            props=dict(alpha=0.3, facecolor='blue'), interactive=True)
        
        def on_cbar_lim_changed(event_ax):
            vmin, vmax = event_ax.get_ylim()
            # Programmatically update the selection box in the histogram
            span.extents = (vmin, vmax) 
            update_elements(vmin, vmax)

        ax_cbar.callbacks.connect('ylim_changed', on_cbar_lim_changed)

    return out






    
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.widgets import SpanSelector
# import matplotlib.gridspec as gridspec

# # 1. Setup Data
# data = np.random.randn(100, 100).cumsum(axis=0).cumsum(axis=1)
# cmap = plt.get_cmap('viridis')

# # 2. Layout with constrained_layout=True for dynamic resizing
# fig = plt.figure(figsize=(12, 6), constrained_layout=True)
# gs = gridspec.GridSpec(1, 3, width_ratios=[15, 0.6, 2], figure=fig)
# ax_img = fig.add_subplot(gs[0])
# ax_cbar = fig.add_subplot(gs[1])
# ax_hist = fig.add_subplot(gs[2])

# # 3. Initial Plots
# im = ax_img.imshow(data, cmap=cmap)
# fig.colorbar(im, cax=ax_cbar)

# counts, bins, patches = ax_hist.hist(data.flatten(), bins=100, 
#                                      orientation='horizontal', color='lightgray')
# bin_centers = 0.5 * (bins[:-1] + bins[1:])
# ax_hist.yaxis.tick_right()
# # ax_hist.set_title("Contrast Selector")

# # 4. Sync Function
# def update_elements(vmin, vmax):
#     im.set_clim(vmin, vmax)
#     for center, patch in zip(bin_centers, patches):
#         if vmin <= center <= vmax:
#             norm_val = (center - vmin) / (vmax - vmin) if vmax != vmin else 0
#             patch.set_facecolor(cmap(norm_val))
#             patch.set_alpha(1.0)
#         else:
#             patch.set_facecolor('lightgray')
#             patch.set_alpha(0.1)
    
#     if ax_cbar.get_ylim() != (vmin, vmax):
#         ax_cbar.set_ylim(vmin, vmax)
    
#     fig.canvas.draw_idle()

# # 5. SpanSelector
# span = SpanSelector(ax_hist, update_elements, 'vertical', useblit=False,
#                     props=dict(alpha=0.3, facecolor='blue'), interactive=True)

# # 6. Two-Way Sync: Colorbar to SpanSelector
# def on_cbar_lim_changed(event_ax):
#     vmin, vmax = event_ax.get_ylim()
#     # Programmatically update the selection box in the histogram
#     span.extents = (vmin, vmax) 
#     update_elements(vmin, vmax)

# ax_cbar.callbacks.connect('ylim_changed', on_cbar_lim_changed)

# plt.show()


    
# import matplotlib.pyplot as plt

# # 1. Create a basic 2x1 layout
# fig, (ax_top, ax_bottom) = plt.subplots(2, 1)

# # 2. Get the location spec of the bottom axis
# spec = ax_bottom.get_subplotspec()

# # 3. Create a 1x3 sub-grid inside that specific space
# # Note: You usually want to remove the 'placeholder' axis first
# ax_bottom.remove() 
# sub_gs = spec.subgridspec(1, 3)

# # 4. Fill the sub-grid with new axes
# for i in range(3):
#     fig.add_subplot(sub_gs[0, i])

# plt.show()



# fig = ax.get_figure()

