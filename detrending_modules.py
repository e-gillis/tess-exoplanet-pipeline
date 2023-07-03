import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from gls import Gls

from scipy.signal import savgol_filter

import misc_functions as misc

import wotan as w

### Main Detrending Functions ###

def gaussian_detrend(bjd, fnorm, efnorm):
    residual_rotation, Prot = rotation_check(bjd, fnorm, efnorm)
    fnorm_detrend = fnorm.copy()
    count = 0

    while residual_rotation and count < 3:
        map_soln = build_model_SHO(bjd, fnorm_detrend, efnorm, Prot)
        fnorm_detrend -= map_soln["pred"]/1000

        residual_rotation, Prot = rotation_check(bjd, fnorm_detrend, 
                                                 efnorm)
        count += 1
        
    detrended = (count!=0) and (not residual_rotation)
    
    return fnorm_detrend, detrended


def median_detrend(fnorm, window_length=400):
    sav_model = savgol_filter(fnorm, window_length, 1) - 1
    fnorm_detrend = fnorm - sav_model
    
    return fnorm_detrend, True


def spline_detrend(bjd, fnorm, efnorm, iterative=False):
    if iterative:
            raise NotImplementedError
    
    residual_rotation, Prot = rotation_check(bjd, fnorm, efnorm)
    if residual_rotation:
        max_splines = int((bjd[-1]-bjd[0])//Prot * 8)
    else:
        max_splines = 20
    
    fnorm_detrend = w.flatten(bjd, fnorm, method='pspline',
                              return_trend=False,
                              max_splines=max_splines,
                              break_tolerance=0)
        
    return fnorm_detrend, True 


### Helper Functions ###

def rotation_check(bjd, fnorm, efnorm):
    # Should I bin the light curve?
    bjd, fnorm, efnorm = misc.bin_curve(bjd, fnorm, efnorm)
    T = bjd[-1] - bjd[0]
    
    gls = Gls(((bjd, fnorm, efnorm)), fend=10, fbeg=0.1/(bjd[-1]-bjd[0]))
    
    Prot = gls.best['P']
    
    theta = gls.best['amp'], gls.best['T0']
    Prot, offset = gls.best['P'], gls.best['offset']
    model = misc.sincurve(bjd, *theta, Prot, offset)
    model_null = np.ones(len(bjd)) * offset
    
    dBIC = misc.DeltaBIC(fnorm, efnorm, model, model_null, k=4)
    
    return (dBIC<=-10 and Prot < T/2), Prot



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
            tau=tt.exp(log_tau_gp),
        )
        gp = GaussianProcess(kernel, t=bjd, yerr=tt.exp(log_sigma_lc))
        resid = fnorm
        gp.marginal("gp", observed=resid)
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


def chi_squared(x_data, y_data, y_sigma, y_fit, dof):

    chi_array = (y_data - y_fit)**2/y_sigma**2
    chi_squared = 1/(len(y_data)-dof) * np.sum(chi_array)
    
    return chi_squared


def spline_penalty(bjd, fnorm, efnorm, trend, nsplines):
    total_nsplines = sum(nsplines)
    chi_2 = chi_squared(bjd, fnorm, efnorm, trend, total_nsplines)
    return chi_2
     