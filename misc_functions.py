import numpy as np
import matplotlib.pyplot as plt
from transitleastsquares import transit_mask
import batman


def lnlike(y, ey, model):
    assert np.all(np.isfinite(np.ascontiguousarray(ey)))
    return -.5 * np.nansum((y-model)**2 / ey**2)


def DeltaBIC(y, ey, model, modelnull, k=5, knull=1):
    '''Transit model (from the TLS) vs the null model (i.e. a flat line)
    k is number of free parameters
    '''
    # theta = {P,T0,D,Z,baseline}
    BIC_model = k*np.log(y.size) - 2*lnlike(y, ey, model)   
    BIC_null = knull*np.log(y.size) - 2*lnlike(y, ey, modelnull)
    return BIC_model - BIC_null


def sincurve(x, amp, T0, P, offset):
    return amp * np.sin(2*np.pi*(x-T0)/P) + offset


def bin_curve(bjd, fnorm, efnorm, bin_width=10, even_bins=False,
              bin_length=0.001):
    """Bin a given lightcurve 
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
    
    for i in range(len(bin_indeces)-1):
        j, k = bin_indeces[i], bin_indeces[i+1]
        bin_bjd[i] = np.mean(bjd[j:k])
        bin_fnorm[i] = np.median(fnorm[j:k])
        bin_efnorm[i] = ((np.std(fnorm[j:k])**2+\
                          np.mean(efnorm[j:k])**2)/(k-j))**0.5
        
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


def plot_result(result, savefig=None, show=True, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    
    plt.axline((0, 6),(1, 6))
    ax.set_xlim(min(result.periods), max(result.periods))
    ax.plot(result.periods, result.power, color='black', lw=0.5)
    for n in range(1, 20):
        ax.axvline(n*result.period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(result.period/n, alpha=0.4, lw=1, linestyle="dashed")
    ax.set_xlabel("Period (Days)")
    ax.set_ylabel("SDE")
    
    ax.set_title(f'SDE Peak at {round(result.period, 4)} Days')
    
    if savefig:
        fig.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()
    

def get_Ntransits(P, T0, duration, bjd):
    intransit = transit_mask(bjd, P, duration, T0)
    bjd_diff = bjd[intransit][1:] - bjd[intransit][:-1]
    N = sum(bjd_diff > (P-duration)) + 1
    
    return N
    

## Modeling Functions ##

def batman_model(bjd, T0, P, Rp, b, R, M, u):
    # G in R⊙^3 / M⊙ days^2
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

    return m.light_curve(bm_params)    


def generate_transit_model(best_period, R, M, u):
        
    def transit_model(bjd, T0, Rp, b):        
        light_curve = batman_model(bjd, T0, best_period, Rp, b, R, M, u)
        return light_curve
    
    return transit_model
