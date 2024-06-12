import numpy as np
import matplotlib.pyplot as plt
from transitleastsquares import transit_mask
from scipy.interpolate import interp1d
import batman


# Log likelyhood of a model
def lnlike(y, ey, model):
    assert np.all(np.isfinite(np.ascontiguousarray(ey)))
    
    nancut = ~np.isnan(ey) & ~(ey == 0)
    return -.5 * np.nansum((y[nancut]-model[nancut])**2 / ey[nancut]**2)


def DeltaBIC(y, ey, model, modelnull, k=5, knull=1):
    '''Transit model (from the TLS) vs the null model (i.e. a flat line)
    k is number of free parameters
    '''
    # theta = {P,T0,D,Z,baseline}
    BIC_model = k*np.log(len(y)) - 2*lnlike(y, ey, model)   
    BIC_null = knull*np.log(len(y)) - 2*lnlike(y, ey, modelnull)

    # print(BIC_model, BIC_null)
    
    return BIC_model - BIC_null


def sincurve(x, amp, T0, P, offset):
    return amp * np.sin(2*np.pi*(x-T0)/P) + offset


def bin_curve(bjd, fnorm, efnorm, bin_width=10, even_bins=False,
              bin_length=0.007, use_rms=True):
    """Bin a given lightcurve 
    
    BJD MUST BE SORTED
    """
    if even_bins:
        bjd_cuts = np.arange(bjd[0], bjd[-1]+bin_length, bin_length)
        index_cuts = []
        j = 0
        i = 0
        while i < len(bjd) and j+1 < len(bjd_cuts):
            if bjd[i] >= bjd_cuts[j]:
                if bjd[i] >= bjd_cuts[j+1]:
                    j += 1
                else:
                    index_cuts.append(i)
                    j += 1
            i += 1
        index_cuts.append(len(bjd))
        bin_indeces = np.array(index_cuts)

    else:
        bin_indeces = np.arange(0, len(bjd)+bin_width, bin_width)
    
    bin_bjd    = np.zeros(len(bin_indeces)-1)
    bin_fnorm  = np.zeros(len(bin_indeces)-1)
    bin_efnorm = np.zeros(len(bin_indeces)-1)
    
    # Could use weighted mean
    for i in range(len(bin_indeces)-1):
        j, k = bin_indeces[i], bin_indeces[i+1]
        bin_bjd[i] = np.mean(bjd[j:k])
        bin_fnorm[i] = np.median(fnorm[j:k])
        if use_rms:
            bin_efnorm[i] = (np.std(fnorm[j:k])**2/(k-j))**0.5
        else:
            bin_efnorm[i] = (np.mean(efnorm[j:k])**2/(k-j))**0.5
        
    return bin_bjd, bin_fnorm, bin_efnorm



def upper_sigma_clip(series, sig, clip=None, iterative=False):
    """
    Return a boolean array to index by to perform an upper sigma clip
    
    === Parameters ===
    series: 1D numpy array
        Array of series to be sigma clipped
    sig: float or int
        Sigma clip factor
    clip: boolean numpy array
        Starting array for the sigma clip, False data points will be masked.
        Must be the same shape as series.
    iterative: boolean, default False
        If True, continue interatively sigma clipping until all data points
        are within sig standard deviations
        
    === Returns ===
    clip: boolean numpy array
        Boolean array of points to be clipped, same shape as series
    """
    delta = 1
    if clip is None:
        clip = np.ones(series.size, dtype=bool)
    c_size = np.sum(~clip)
    
    while delta:
        c_size = np.sum(~clip)
        median = np.nanmedian(series[clip])
        ## No reason for this to be rms
        rms = np.sqrt(np.nanmedian((series[clip] - median)**2))
        clip = series <= median + rms*sig
        if iterative:
            delta = np.sum(~clip) - c_size
        else:
            delta = 0
    
    return clip


def phase_fold(bjd, P, T0):
    folded_t = (bjd - T0 + P/2) % P / P - 0.5
    return folded_t


def plot_result(result, savefig=None, show=True, fig=None, ax=None,
                title_ext=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    
    ax.axline((0, 6),(1, 6))
    ax.set_xlim(min(result.periods), max(result.periods))
    ax.plot(result.periods, result.power, color='black', lw=0.5)
    for n in range(1, 20):
        ax.axvline(n*result.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(result.period/n, alpha=0.4, lw=1, linestyle="dashed")
    ax.set_xlabel("Period (Days)")
    ax.set_ylabel("SDE")

    title = f'SDE Peak at {round(result.period, 4)} Days'
    if title_ext is not None:
        title += title_ext
    ax.set_title(title)
    
    if savefig:
        fig.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    

def get_Ntransits(P, T0, duration, bjd, countdiff=True):
    intransit = transit_mask(bjd, P, duration, T0)
    
    if countdiff:
        bjd_diff = bjd[intransit][1:] - bjd[intransit][:-1]
        N = sum(bjd_diff > (P-duration)) + 1

    else:
        N = sum(intransit) * (2/(60*24)) / duration

    return N


def transit_duration(M, R, P, Rp, b):
    G = 2942.2062
    a = (P**2 * M / (4*np.pi**2) * G)**(1/3) / R
    duration = P/np.pi * np.arcsin(((Rp+1)**2-b**2)**0.5/a)
    
    return duration


def transit_duration_simple(M, R, P):
    # G in R⊙^3 / M⊙ days^2
    G = 2942.2062
    a = (P**2 * M / (4*np.pi**2) * G)**(1/3) / R
    duration = P/np.pi * np.arcsin(1/a)
    
    return duration

    

## Modeling Functions ##

def batman_model(bjd, T0, P, Rp, b, R, M, u, offset=0):
    # Should make this a function
    G = 2942.2062
    a = (P**2 * M / (4*np.pi**2) * G)**(1/3) / R
    # Compute inclination from 
    inc = np.arccos(b / a) * 180/np.pi

    # Initialize Batman Transit
    bm_params = batman.TransitParams()
    bm_params.per, bm_params.rp, bm_params.inc = P, Rp, inc
    bm_params.t0 = T0
    bm_params.limb_dark = "quadratic"
    bm_params.u = u
    bm_params.a = a
    bm_params.ecc = 0
    bm_params.w = 90

    # Make Model
    m = batman.TransitModel(bm_params, bjd)

    return m.light_curve(bm_params) + offset


def generate_transit_model(best_period, R, M, u):
        
    def transit_model(bjd, T0, Rp, b):        
        light_curve = batman_model(bjd, T0, best_period, Rp, b, R, M, u)
        return light_curve
    
    return transit_model


### Star Conversion Functions ###
### Based on https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt

def m_from_r(radius, radius_err, invert=False):
    r_vals = np.array([13.43 , 12.13 , 11.45 , 10.71 , 10.27, 9.82, 9.42, 8.95,
                       8.47 , 8.06 , 7.72 , 7.5  , 7.16 , 6.48 , 5.71 , 5.02 ,
                       4.06 , 3.89 , 3.61 , 3.46 , 3.36 , 3.27 , 2.94 , 2.86 ,
                       2.49 , 2.45 , 2.193, 2.136, 2.117, 1.861, 1.794, 1.785,
                       1.775, 1.75 , 1.747, 1.747, 1.728, 1.679, 1.622, 1.578,
                       1.533, 1.473, 1.359, 1.324, 1.221, 1.167, 1.142, 1.1  ,
                       1.06 , 1.012, 1.002, 0.991, 0.977, 0.949, 0.927, 0.914,
                       0.853, 0.813, 0.797, 0.783, 0.755, 0.713, 0.701, 0.669,
                       0.63 , 0.615, 0.608, 0.588, 0.544, 0.501, 0.482, 0.446,
                       0.421, 0.361, 0.3  , 0.274, 0.217, 0.196, 0.156, 0.137,
                       0.126, 0.12 , 0.116, 0.114, 0.104,  0.102,  0.101])
    m_vals = np.array([59.   , 48.   , 43.   , 38.   , 35.   , 31., 28., 26,
                       23.6 , 21.9 , 20.2 , 18.7 , 17.7 , 14.8  ,11.8 , 9.9,
                       7.3  , 6.1  , 5.4  , 5.1  , 4.7 ,  4.3  , 3.92, 3.38,
                       2.75 , 2.68 , 2.18 , 2.05 , 1.98,  1.86 , 1.93, 1.88,
                       1.83 , 1.77 , 1.81 , 1.75 , 1.61,  1.5  , 1.46, 1.44,
                       1.38 , 1.33 , 1.25 , 1.21 , 1.18,  1.13 , 1.08, 1.06,
                       1.03 , 1.   , 0.99 , 0.985, 0.98,  0.97 , 0.95, 0.94,
                       0.9  , 0.88 , 0.86 , 0.82 , 0.78,  0.73 , 0.7 , 0.69,
                       0.64 , 0.62 , 0.59 , 0.57 , 0.54 ,  0.5 , 0.47, 0.44,
                       0.4  , 0.37 , 0.27 , 0.23 , 0.18,  0.162, 0.12, 0.102,
                       0.093, 0.09 , 0.088, 0.085, 0.08 ,  0.079,  0.078])
    
    if invert:
        m_interp = interp1d(r_vals, m_vals)
    else:
        m_interp = interp1d(m_vals, r_vals)
    
    mass = float(m_interp(radius))
    mass_err = abs(m_interp(radius+radius_err) - m_interp(radius-radius_err))
    
    return mass, mass_err


def r_from_m(mass, mass_err):
    radius, radius_err = m_from_r(mass, mass_err, invert=True)
    
    return radius, radius_err