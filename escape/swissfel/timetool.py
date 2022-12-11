from scipy.special import erf
from scipy.signal.windows import hann
import numpy as np
import escape
import matplotlib.pyplot as plt


def get_relative_profiles(tt_prof, is_reference, Npids=100, fac_references=1.0):
    tt_pd = tt_prof[~is_reference]
    tt_bg = tt_prof[is_reference]
    indxs = escape.Array(data=tt_prof.index, index=tt_prof.index)
    indsrt = indxs.digitize(np.arange(indxs.data.min(), indxs.data.max(), Npids)).ones()
    return escape.concatenate(
        [
            on / (off * fac_references)
            for on, off in zip(
                (indsrt[:, None] * tt_pd).scan,
                (indsrt[:, None] * tt_bg).scan.mean(axis=0),
            )
        ]
    )


def find_steps(
    tt_prof,
    roi=[None],
    is_sigref=None,
    sigref_Nix_avg=100,
    stepref=None,
    step_sigma_px=70,
    step_ref_len=300,
    step_windowing="hann",
    findsig_poly=10,
):
    """All inclusice timetool analysis function adding all analysis steps together."""
    if not stepref:
        stepref = get_reference_function(
            sigma_px=step_sigma_px, reflen=step_ref_len, window=step_windowing
        )

    if is_sigref:
        tt_prof = get_relative_profiles(tt_prof, is_sigref, Npids=sigref_Nix_avg)

    ttres = tt_prof.map_index_blocks(
        lambda d: np.asarray(
            [find_signal(td, stepref, dpx_poly=findsig_poly, roi=roi) for td in d]
        ),
        new_element_size=[2],
        dtype=float,
    )
    pos = ttres[:, 0]
    amp = ttres[:, 1]
    return pos, amp


# d.append(ttres[:,0],name='ttpos')
# d.append(ttres[:,1],name='ttamp')


def normstep(step):
    """normalizing a test signal for np.correlate"""
    step = step - np.mean(step)
    step = step / np.sum(step**2)
    return step


def get_max(c, dpx_poly=None, offset=0, verbose_plot=False):
    """getting maximum from a correlation curve (optionally using polynomial fit)"""
    im = c.argmax()
    mx = c[im]

    if dpx_poly:
        order = 2
        i_f = max(0, im - (dpx_poly // 2))
        i_t = min(im + (dpx_poly // 2), len(c))
        x = np.arange(i_f, i_t)
        y = c[i_f:i_t]
        p = np.polyfit(x, y, order)
        dp = np.polyder(p)
        im = -dp[1] / dp[0]
        if (im < i_f) | (i_t < im):
            im = np.nan
        if verbose_plot:
            plt.plot((xp := np.linspace(i_f, i_t, 50)) + offset, np.polyval(p, xp), "r")

    return im + offset, mx


def find_signal(d, ref, dpx_poly=50, verbose_plot=False, roi=[None]):
    """finding signal ref in d.
    ref is expected to be properly normalized
    return position is corrected to to center location of the reference signal (as found in signal d)"""
    # need to invert both to get correct direction

    x0 = (len(ref) + 1) // 2
    c = np.correlate(d[slice(*roi)], ref, "valid")
    if roi[0]:
        x0 += np.min(roi)
    if verbose_plot:
        plt.plot(np.arange(len(c)) + x0, c)
    p, mx = get_max(c, dpx_poly=dpx_poly, offset=x0, verbose_plot=verbose_plot)
    if verbose_plot:
        plt.axvline(p)
        plt.axhline(mx)

    return p, mx


def refine_reference(data, pos, resolution=1):
    """refining the reference signal based on many example datasets (data) and given positions found before"""
    if isinstance(data, escape.Array) and isinstance(pos, escape.Array):
        tdata, tpos = escape.match_arrays(data, pos)
        tdata, tpos = escape.compute(tdata, tpos)
        tdata = tdata.data
        tpos = tpos.data
    else:
        tdata = data
        tpos = pos
    sel = ~np.isnan(tpos)
    tpos = tpos[sel]
    tdata = tdata[sel, :]
    xb = np.arange(len(tdata[0]))
    xd = xb - np.asarray(tpos).ravel()[:, None]
    xd_mn = np.max(xd[:, 0])
    xd_mn -= xd_mn % resolution
    xb_mn = xd_mn - resolution / 2
    xd_mx = np.min(xd[:, -1])
    xd_mx += resolution - xd_mx % resolution
    xb_mx = xd_mx + resolution / 2
    xr = np.arange(xd_mn, xd_mx + resolution, resolution)
    xb = np.arange(xb_mn, xb_mx + resolution, resolution)
    bns = np.digitize(xd, xb)
    yr = np.bincount(bns.ravel(), weights=tdata.ravel()) / np.bincount(bns.ravel())

    return xr, yr[1:-1]


def get_reference_function(sigma_px=30, reflen=300, window=None):
    rng = reflen / np.sqrt(2) / sigma_px / 2
    ref = -erf(np.linspace(-rng, rng, reflen)) / 2
    if window:
        if window == "hann":
            ref = hann(len(ref)) * ref * 1.64
    ref = normstep(ref)
    return ref
