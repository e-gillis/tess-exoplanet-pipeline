import numpy as np
import matplotlib.pyplot as plt
import misc_functions as misc
from scipy.optimize import curve_fit
from constants import TLS_THREADS

from transitleastsquares import transitleastsquares, transit_mask
from transitleastsquares import catalog_info

def find_transits(bjd, fnorm, efnorm, star_params, threshold=6, max_iterations=4,
                  grazing_search=True, threads=1, method='noise', **tls_kwargs):
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
    model = transitleastsquares(bjd, fnorm, efnorm)
    result = model.power(**tls_kwargs, use_threads=threads)
    
    # Check if a planet candidate is found
    if result["SDE"] < threshold:
        return []
    
    result_list = [result]
    i = 1
    
    # Start looping finding more planets
    grazing = False
    while i < max_iterations:
        # Mask transits
        if method == "model":
            fnorm = model_mask(bjd, fnorm, efnorm, result, star_params)
        else:
            bjd, fnorm = mask_transits(bjd, fnorm, result_list[-1].period, 
                                       2*result_list[-1].duration, 
                                       result_list[-1].T0, 
                                       method=method)
        
        # Look for planets again with transits masked
        model = transitleastsquares(bjd, fnorm, efnorm)
        result = model.power(**tls_kwargs, use_threads=threads,
                             transit_template=['default', 'grazing'][grazing])
        
        # plt.scatter(bjd, fnorm, s=0.1)
        # plt.show()
        
        diff = result.power[:-1]-result.power[1:]
        good_spec = sum(diff==0) / len(result.power) < 0.9
        
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
    """
    Mask transits in a given lightcurve using the method specified.

    === Arguments ===
    bjd: numpy array
        The bjd timeseries for the lightcurve
    fnorm: numpy array
        Normalized detrended lightcurve
    period: float
        Period of the transit signal
    duration: float
        Duration of the transit
    T0: float
        Middle transit time of the first transit in the timeseries
    method: string ('noise', 'remove')
        Method to mask transits with: Noise overwrites the intransit timeseries
        with gaussian noise matching the mean and rms of the full timeseries;
        Remove excises the intransit data from the timeseries

    """
    # Make sure method is there
    assert method in ['remove', 'noise']
    
    # Avoid aliasing at all costs
    bjd, fnorm = bjd.copy(), fnorm.copy()
    intransit = transit_mask(bjd, period, duration, T0)
    
    if method == 'remove':
        return bjd[~intransit], fnorm[~intransit]
    
    elif method == 'noise':
        np.random.seed(42)
        rms = np.mean((fnorm - 1)**2)**0.5
        fnorm[intransit] = np.random.normal(loc=1, scale=rms, 
                                            size=sum(intransit))
        return bjd, fnorm


def fit_transit_model(bjd, fnorm, efnorm, result, star_params, 
                      r_update=False, durcheck=True):
    """
    Fit a transit model to a lightcurve informed by a TLS result using scipy's
    curve_fit. Retrun the optimal planet paramaters.

    === Arguments ===
    bjd: numpy array
        The bjd timeseries for the lightcurve
    fnorm: numpy array
        Normalized detrended lightcurve
    efnorm: numpy array
        Error on the normalized flux
    result: TLS Result
        TLS result to inform the transit model fitting
    star_params: (float, float, (float, float))
        Radius, mass and limb darkening parameters of the host star
        
    === Returns ===
    T0: float
        Time of mid-transit for the first transit in the timeseries
    period: float
        Period of the transits
    Rp: float
        Ratio of the planet's radius to the star's radius
    b: float
        Impact parameter of the transit
    """
    # Unpack Parameters
    R, M, u = star_params
    dur_guess = misc.transit_duration(M, R, result.period, 
                                      (1-result.depth)**0.5, 1)
    
    if result.duration / dur_guess < 5 and durcheck:
        print("Result Duration 5 times greater than predicted, correcting")
        # Redo result with shorter window around period
        tls_kwargs = {"R_star": R, "M_star": M, "u": u,
                     "period_min": result.period*0.95, 
                     "period_max": result.period*1.05}
        model = transitleastsquares(bjd, fnorm, efnorm, verbose=False)
        res = model.power(**tls_kwargs, use_threads=TLS_THREADS, 
                          show_progress_bar=False)
        result.period, result.duration, result.T0, result.depth = \
        res.period, res.duration, res.T0, res.depth
    
    # Use period, T0, duration, 
    T0, T0_delta = result.T0, result.duration
    period, p_delta = result.period, abs(result.period_uncertainty)

    # Transit model to fit parameters in
    transit_model = lambda bjd, T0, P, Rp, b, offset:\
                   misc.batman_model(bjd, T0, P, Rp, b, R, M, u, offset)

    # Only fit to the intransit data
    intransit = transit_mask(bjd, result.period, result.T0, 2*result.duration)

    # Set Bounds
    bounds = np.array(((T0-T0_delta, T0+T0_delta),
                      (period-p_delta, period+p_delta),
                      (0, 1), 
                      (0, 1),
                      (-np.std(fnorm[intransit]), np.std(fnorm[intransit])))).T
    p0 = (T0, period, (1-result.depth)**0.5, 0.5, 0)

    # Run the curve fit
    popt, pcov = curve_fit(transit_model, bjd[intransit], fnorm[intransit], 
                           p0=p0, bounds=bounds,
                           sigma=efnorm[intransit])
    # Unpack variables and return
    T0, P, Rp, b, offset = popt

    if r_update:
        result.period = P
        result.T0 = T0
        result.depth = 1 - Rp**2
    
    return T0, P, Rp, b, offset


def model_mask(bjd, fnorm, efnorm, result, star_params):
    """
    Model a transit based on a TLS result and mask it out of a given lightcurve

    === Arguments ===
    bjd: numpy array
        The bjd timeseries for the lightcurve
    fnorm: numpy array
        Normalized detrended lightcurve
    efnorm: numpy array
        Error on the normalized flux
    result: TLS Result
        TLS result to inform the transit model fitting
    star_params: (float, float, (float, float))
        Radius, mass and limb darkening parameters of the host star
        
    === Returns ===
    masked_fnorm: numpy array
        Residual flux after the transit has been modeled and subtracted
    """
    T0, P, Rp, b, o = fit_transit_model(bjd, fnorm, efnorm, result, star_params)
    R, M, u = star_params
    transit_model = misc.batman_model(bjd, T0, P, Rp, b, R, M, u, o)

    return fnorm - transit_model + 1
    

def model_mask_checks(fnorm, transit_model, old_result):
    """
    Determine if a model mask has sufficiently removed the SDE signal
    from a TLS result to prevent double results
    
    === Arguments ===
    fnorm: numpy array
        Normalized detrended lightcurve
    transit_model: numpy array
        Trandit model to mask out the transits
    old_result: 
        transitleastsquares result from unmasked fnorm
    """
    model_masked = fnorm - transit_model + 1
    raise NotImplementedError 