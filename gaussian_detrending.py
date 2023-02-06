import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from gls import Gls


def rotation_check(bjd, fnorm, efnorm):
    # Should I bin the light curve?
    bjd, fnorm, efnorm = bin_curve(bjd, fnorm, efnorm)
    
    gls = Gls(((bjd, fnorm, efnorm)), fend=10, fbeg=0.1/(bjd[-1]-bjd[0]))
    
    Prot = gls.best['P']
    
    theta = gls.best['amp'], gls.best['T0']
    Prot, offset = gls.best['P'], gls.best['offset']
    model = sincurve(bjd, *theta, Prot, offset)
    model_null = np.ones(len(bjd)) * offset
    
    
    dBIC = DeltaBIC(fnorm, efnorm, model, model_null, k=4)
    
    return dBIC <= -10, Prot



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
