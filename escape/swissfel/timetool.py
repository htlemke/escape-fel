from scipy.special import erf
import numpy as np
import escape
import matplotlib.pyplot as plt

def get_relative_profiles(tt_prof,xon,Npids=100, fac_references=1.):
    tt_pd = tt_prof[xon]
    tt_bg = tt_prof[~xon]
    indxs = escape.Array(data=tt_prof.index,index=tt_prof.index)
    indsrt = indxs.digitize(np.arange(indxs.data.min(),indxs.data.max(),Npids)).ones()
    return escape.concatenate([on/(off*fac_references) for on,off in zip((indsrt[:,None]*tt_pd).scan,(indsrt[:,None]*tt_bg).scan.mean(axis=0))])

def normstep(step):
    """normalizing a test signal for np.correlate"""
    step = step-np.mean(step)
    step = step/np.sum(step**2)
    return step

def get_max(c,px_w=None):
    """getting maximum from a correlation curve (optionally using polynomial fit)"""
    im = c.argmax()
    mx = c[im]

    if px_w:
        order = 2
        i_f = max(0,im-(px_w//2))
        i_t = min(im+(px_w//2),len(c))
        x = np.arange(i_f,i_t)
        y = c[i_f:i_t]
        p = np.polyfit(x,y,order)
        dp = np.polyder(p)
        im = -dp[1]/dp[0]
    return im,mx

def find_signal(d,ref,px_w = 50, verbose_plot=False):
    """finding signal ref in d. 
    ref is expected to be properly normalized
    return position is corrected to to center location of the reference signal (as found in signal d)"""
    #need to invert both to get correct direction
    x0 = (len(ref)+1)//2
    c = np.correlate(d,ref,'valid')
    p,mx = get_max(c,px_w=px_w)
    if verbose_plot:
        plt.plot(c)
        plt.axvline(p)
        plt.axhline(mx)
    
    
    return p+x0,mx
    
def refine_reference(data,pos,resolution=1):
    """refining the reference signal based on many example datasets (data) and given positions found before"""
    xb = np.arange(len(data[0]))
    xd = xb-np.asarray(pos).ravel()[:,None]
    xd_mn = np.max(xd[:,0])
    xd_mn -= xd_mn%resolution
    xb_mn = xd_mn - resolution/2
    xd_mx = np.min(xd[:,-1])
    xd_mx += (resolution - xd_mx%resolution)
    xb_mx = xd_mx + resolution/2
    xr = np.arange(xd_mn,xd_mx+resolution,resolution)
    xb = np.arange(xb_mn,xb_mx+resolution,resolution)
    bns = np.digitize(xd,xb)
    yr = np.bincount(bns.ravel(),weights=data.ravel())/np.bincount(bns.ravel())
    
    
    return xr,yr[1:-1]



def get_reference_function(width_px=70,reflen = 300):
    rng = reflen/width_px
    ref = -erf(np.linspace(-rng,rng,reflen))/2
    ref = normstep(ref)
    return ref