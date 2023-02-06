import numpy as np


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
        bin_efnorm[i] = (np.std(fnorm[j:k])**2 + np.mean(efnorm[j:k])**2)**0.5
        
    
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

