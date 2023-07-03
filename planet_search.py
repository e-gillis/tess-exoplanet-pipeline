import numpy as np
import matplotlib.pyplot as plt
import misc_functions as misc


from transitleastsquares import transitleastsquares, transit_mask
from transitleastsquares import catalog_info

def find_transits(bjd, fnorm, threshold=6, max_iterations=5, grazing_search=True, 
                  threads=1, method='noise', **tls_kwargs):
    """Return a list of TLS transit candidates from a given lightcurve.
    
    === Parameters ===
    BJD: 1D numpy array
        BJD of each data point in the observation timeseries
    fnorm: 1D array
        Normalized flux at each BJD. fnorm *must* be detrended to return 
        sensible results
    threshold: float
        Threshold for a peak in SED to be considered a transit candidate
    max_iterations: int
        Maximum number of TLS iterations to run, sucessively masking each 
        significant transit candidate
    **tls_kwargs:
        Keyword arguments to pass into the tls run
        
    === Returns ===
    result_list: List[TLS result]
        List of results for transit candidates with sufficient SED, see
        transitleastsquares.readthedocs.io/en/latest/Python interface.html#return-values
        for more information
    """
    
    # All points are in the transit
    # intransit = np.zeros(len(bjd), dtype=bool)
    
    # Look for the first planet
    model = transitleastsquares(bjd, fnorm)
    result = model.power(**tls_kwargs, use_threads=threads)

    
    # Check if a planet candidate is found
    if result["SDE"] < threshold:
        return []
    
    result_list = [result]
    i = 1
    
    # Start looping finding more planets
    grazing = False
    while i <= max_iterations:
        # Mask transits
        bjd, fnorm = mask_transits(bjd, fnorm, result_list[-1].period, 
                                   result_list[-1].duration, 
                                   result_list[-1].T0, 
                                   method=method)
        
        # Look for planets again with transits masked
        model = transitleastsquares(bjd, fnorm)
        result = model.power(**tls_kwargs, use_threads=threads,
                             transit_template=['default', 'grazing'][grazing])
        
        # plt.scatter(bjd, fnorm, s=0.1)
        # plt.show()
        
        diff = result.power[:-1]-result.power[1:]
        good_spec = sum(diff==0) / len(result.power) < 0.67
        
        # Check if planet found
        if result["SDE"] > threshold and good_spec:
            result_list.append(result)
        # Run a grazing template to see if we missed something?
        elif not grazing and grazing_search: 
            grazing = True
            continue
        else:
            break

        # Increment
        i += 1
        
    # plt.scatter(bjd, fnorm, s=0.1)
    # plt.show()
        
    return result_list


def mask_transits(bjd, fnorm, period, duration, T0, method):
    # Make sure method is there
    assert method in ['remove', 'noise']
    
    # Avoid aliasing at all costs
    bjd, fnorm = bjd.copy(), fnorm.copy()
    intransit = transit_mask(bjd, period, 2*duration, T0)
    
    if method == 'remove':
        return bjd[~intransit], fnorm[~intransit]
    
    elif method == 'noise':
        np.random.seed(42)
        rms = np.mean((fnorm - 1)**2)**0.5
        fnorm[intransit] = np.random.normal(loc=1, scale=rms, 
                                            size=sum(intransit))
        return bjd, fnorm
