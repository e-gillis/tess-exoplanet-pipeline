import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from gls import Gls
from scipy.stats import norm, kstest

from scipy.signal import savgol_filter

import misc_functions as misc

import wotan as w

### Main Detrending Functions ###

def gaussian_detrend(bjd, fnorm, efnorm, SHOTerm=False):
    residual_rotation, Prot = rotation_check(bjd, fnorm, efnorm, verb=True)
    fnorm_detrend = fnorm.copy()
    
    if SHOTerm and residual_rotation:
        count = 0
        while residual_rotation and count < 4:
            map_soln = build_model_SHO(bjd, fnorm_detrend, efnorm, Prot)
            fnorm_detrend -= map_soln["pred"]/1000

            residual_rotation, Prot = rotation_check(bjd, fnorm_detrend, 
                                                     efnorm, verb=True)
            count += 1
        detrended = (count!=0) and (not residual_rotation or count >= 4)
    
    elif residual_rotation:
        map_soln = build_model_RotationTerm(bjd, fnorm_detrend, efnorm, Prot)
        fnorm_detrend -= map_soln["pred"]/1000
        
        residual_rotation, Prot = rotation_check(bjd, fnorm_detrend, 
                                                 efnorm, verb=True)
        detrended = not residual_rotation
    
    else:
        detrended = False
    
    return fnorm_detrend, detrended


def median_detrend(fnorm, split=True, window_length=360):
    # Should use 12 hours like Ment? Yes
    sav_model = savgol_filter(fnorm, window_length, 1) - 1
    fnorm_detrend = fnorm - sav_model
    
    return fnorm_detrend, True


def spline_detrend(bjd, fnorm, efnorm, iterative=False):
    if iterative:
        raise NotImplementedError
            
    bjd_range = max(bjd) - min(bjd)
    residual_rotation, Prot = rotation_check(bjd, fnorm, efnorm)
    
    if residual_rotation:
        max_splines = int(bjd_range/Prot * 8)
    else:
        max_splines = int(20 * bjd_range/28) # More for long lightcurves
    
    fnorm_detrend = w.flatten(bjd, fnorm, method='pspline',
                              return_trend=False,
                              max_splines=max_splines,
                              break_tolerance=0)
        
    return fnorm_detrend, True 


### Helper Functions ###

def rotation_check(bjd, fnorm, efnorm, bin_length=0.05, verb=False):
    # Bin the lightcurve with 20 bins per day
    bbjd, bfnorm, befnorm = misc.bin_curve(bjd, fnorm, efnorm, 
                                           even_bins=True, bin_length=bin_length)
    T = bjd[-1] - bjd[0]
    
    # Minimum period of 10/day, maximum period of 2/full time
    gls = Gls(((bjd, fnorm, efnorm)), fend=10, fbeg=2/(bjd[-1]-bjd[0]))
    
    Prot = gls.best['P']
    
    theta = gls.best['amp'], gls.best['T0']
    Prot, offset = gls.best['P'], gls.best['offset']
    model = misc.sincurve(bbjd, *theta, Prot, offset)
    model_null = np.ones(len(bbjd)) * np.median(bfnorm)
    
    FAP = gls.FAP(gls.pmax)
    dBIC = misc.DeltaBIC(bfnorm, befnorm, model, model_null, k=4)
    rotation = dBIC <= -50 and FAP < 0.01
    
    if verb:
        print(f"Delta BIC: {dBIC},",
              f"{['no', f'{Prot} day'][rotation]} Rotation Detected")
    
    return rotation, Prot


def build_model_SHO(bjd, fnorm, efnorm, Prot):
    fnorm = (fnorm - 1)*1000
    efnorm = efnorm*1000
    
    with pm.Model() as model:
        # Fitting a lightcurve mean
        mean = pm.Normal("mean", mu=0, sd=10)
        
        # Transit jitter and GP parameters
        # Spread of initial Data
        log_sigma_gp = pm.Uniform("log_sigma_gp", lower=-3, 
                                  upper=np.log(np.std(fnorm)))

        # Quantifying measurement uncertainty
        log_sigma_lc = pm.Normal("log_sigma_lc", 
                                 mu=np.log(np.median(efnorm)), sd=.1)
        # Period Fit
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(Prot), sd=.2)
        # Damping and variation timescale
        log_tau_gp = pm.Uniform("log_tau_gp", lower=np.log(10*Prot), upper=20)

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            tau=tt.exp(log_tau_gp))
        
        gp = GaussianProcess(kernel, t=bjd, yerr=tt.exp(log_sigma_lc))
        gp.marginal("gp", observed=fnorm)
        pm.Deterministic("pred", gp.predict(fnorm))
        
        start = model.test_point
        map_soln = pmx.optimize(start=start,
                                vars=[log_sigma_lc, log_sigma_gp, 
                                      log_rho_gp, log_tau_gp])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(start=map_soln,
                                vars=[log_sigma_lc, log_sigma_gp,
                                      log_rho_gp, log_tau_gp])
        map_soln = pmx.optimize(start=map_soln)

    return map_soln


def build_model_RotationTerm_exoplanet(bjd, fnorm, efnorm, Prot):
    fnorm = (fnorm - 1)*1000
    efnorm = efnorm*1000
    
    with pm.Model() as model:
        # The mean flux of the time series
        mean = pm.Normal("mean", mu=0.0, sigma=10.0)

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(efnorm)), sigma=2.0)

        # A term to describe the non-periodic variability
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
        )

        # The parameters of the RotationTerm kernel
        sigma_rot = pm.InverseGamma(
            "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        log_period = pm.Normal("log_period", mu=np.log(Prot), sigma=2.0)
        period = pm.Deterministic("period", tt.exp(log_period))
        log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
        log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
        f = pm.Uniform("f", lower=0.1, upper=1.0)

        # Set up the Gaussian Process model
        kernel = terms.RotationTerm(
            sigma=sigma_rot,
            period=period,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f)
        kernel += terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
        gp = GaussianProcess(
            kernel,
            t=bjd,
            diag=efnorm**2 + tt.exp(2 * log_jitter),
            mean=mean,
            quiet=True)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("gp", observed=fnorm)

        # Compute the mean model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict(fnorm))

        # Optimize to find the maximum a posterior parameters
        map_soln = pmx.optimize()
    
    return map_soln


def build_model_RotationTerm(bjd, fnorm, efnorm, Prot):
    fnorm = (fnorm - 1)*1000
    efnorm = efnorm*1000
    
    with pm.Model() as model:
        # The mean flux of the time series
        mean = pm.Normal("mean", mu=0.0, sigma=10.0)
        
        # Spread of initial Data
        log_sigma_gp = pm.Uniform("log_sigma_gp", lower=-3, 
                                  upper=np.log(np.std(fnorm)))
        
        # Quantifying measurement uncertainty
        log_sigma_lc = pm.Normal("log_sigma_lc", 
                                 mu=np.log(np.median(efnorm)), sd=0.1)
        # Period Fit
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(Prot), sd=0.01)
        
        # Quality Parameters for the oscillators
        Q0 = pm.Normal("Q0", mu=7.5, sigma=2) 
        # log_Q0 = pm.Normal("log_Q0", mu=0, sigma=2)
        log_dQ = pm.Normal("log_dQ", mu=0, sigma=2.0)
        f = pm.Uniform("f", lower=0.8, upper=1)

        
        # Make the kernel
        kernel = terms.RotationTerm(sigma=tt.exp(log_sigma_gp),
                                    period=tt.exp(log_rho_gp),
                                    Q0=Q0,
                                    dQ=tt.exp(log_dQ),
                                    f=f)

        gp = GaussianProcess(kernel, t=bjd, yerr=tt.exp(log_sigma_lc))
        gp.marginal("gp", observed=fnorm)
        pm.Deterministic("pred", gp.predict(fnorm))
        
        start = model.test_point
        map_soln = pmx.optimize(start=start)
    
    return map_soln
        

def ks_noise_test(fnorm_detrend, mu=None, sigma=None, _use_scipy=True):
    """
    Use the KS-test to determine whether a detrended lightcurve is consistent
    with white noise
    
    === Arguments ===
    fnorm_detrend:
        Detrended TESS lightcurve
        
    === Returns ===
    prob_D:
        The probability that the light curve is drawn from gaussuian noise
    """
    
    # fnorm_cdf = cdf(fnorm_detrend)
    if mu is None:
        mu = np.mean(fnorm_detrend)
    if sigma is None:
        sigma = np.std(fnorm_detrend)
    gauss_cdf = lambda x: norm.cdf(x, loc=mu, scale=sigma)

    if _use_scipy:
        prob_D = kstest(fnorm_detrend, gauss_cdf).pvalue

    else:
        fnorm_sort = np.sort(fnorm_detrend)
        diffs = np.zeros(2*len(fnorm_sort))
        
        for i, dat in enumerate(fnorm_sort):
            diffs[i] = abs(gauss_cdf(dat) - (i+1)/len(fnorm_sort))
            diffs[i+len(fnorm_sort)] = abs(gauss_cdf(dat-1e-5) - i/len(fnorm_sort))
    
        # Take max
        D = max(np.abs(diffs)) 
        N = len(fnorm_detrend)
        
        # KS Test probability
        prob_D = 1 - Q_KS((N**0.5 + 0.12 + 0.11/N**0.5)*D)
    
    return prob_D


def cdf(dat):
    """Return a function that gives the cumulative distribution
    """
    return lambda x: sum(dat <= x) / len(dat)


def Q_KS(z):
    """Translated from Numerical Recipes. They do two different power series,
    but we have enough power to just use the more accurate one
    """
    assert(z >= 0)

    if z == 0:
        return 1
    
    # print(z)
    y = np.exp(-1.23370055013616983/z**2)
    P_KS = 2.25675833419102515*np.sqrt(-np.log(y))*\
           (y + y**9 + y**25 + y**49)
    
    return P_KS


### Flare Finding Function ###
def flare_mask(fnorm, n, nsigma):
    """Find Flares in fnorm and return a flare mask with sequences of n
    consecutive points or more nsigma above the median.
    """
    # All points which are outliers
    outlier_mask = fnorm > (nsigma*np.std(fnorm) + np.mean(fnorm))
    outlier_conv = np.convolve(outlier_mask, np.ones(n), mode='valid')
    
    consecutive = outlier_conv == n
    flare_mask = np.zeros(len(fnorm), dtype=bool)
    
    for i in range(n):
        flare_mask[i:i+len(consecutive)] = flare_mask[i:i+len(consecutive)]\
                                           | consecutive
        
    return flare_mask


#### Not Used ####

def chi_squared(x_data, y_data, y_sigma, y_fit, dof):

    chi_array = (y_data - y_fit)**2/y_sigma**2
    chi_squared = 1/(len(y_data)-dof) * np.sum(chi_array)
    
    return chi_squared


def spline_penalty(bjd, fnorm, efnorm, trend, nsplines):
    total_nsplines = sum(nsplines)
    chi_2 = chi_squared(bjd, fnorm, efnorm, trend, total_nsplines)
    return chi_2
     