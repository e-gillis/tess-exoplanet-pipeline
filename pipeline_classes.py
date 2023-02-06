import matplotlib.pyplot as plt
import numpy as np

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from gls import Gls

from transitleastsquares import transitleastsquares, transit_mask
from transitleastsquares import catalog_info


class TransitSearch:
    
    # Should have all star information
    def __init__(self, tic):
        
        self.tic = tic

        full_lc = TIC_LightCurve(tic)
        lc_series = full_lc.get_splits((full_lc.bjd, full_lc.fnorm, 
                                        full_lc.efnorm, full_lc.sectors, 
                                        full_lc.qual_flags, full_lc.texp))
        self.lightcurves = []
        
        for i in range(len(lc_series[0])):
            series_set = [series[i] for series in lc_series]
            self.lightcurves.append(LightCurve(*series_set))
            self.lightcurves[-1].detrend_lc()
        
        # Star Parameters
        Teff, logg, radius, radius_err,\
        mass, mass_err, RA, Dec = get_star_info(tic)
        
        u = catalog_info(TIC_ID=tic)[0]
        
        self.Teff = Teff
        self.logg = logg
        self.radius = radius
        self.radius_err = radius_err
        self.mass = mass
        self.mass_err = mass_err
        self.RA = RA
        self.Dec = Dec
        self.u = u
        
        self.results = None
    
    
    def transit_search(self, threshold=6, max_iterations=5):
        
        self.results = []
        
        for lc in self.lightcurves:
            results_list = find_transits(lc.bjd, lc.fnorm_detrend, 
                                         period_min=1, period_max=8,
                                         show_progress_bar=False, 
                                         R_star=self.radius, M_star=self.mass,
                                         u=self.u)
            
            self.results.append(results_list)
    
    
    # Is there any point in having a continuous lightcurve?
    def vet_transits(self):
        raise NotImplementedError
    
    def plot_transits(self):
        raise NotImplementedError
        
        


class LightCurve():
    
    def __init__(self, bjd, fnorm, efnorm, sectors, qual_flags, texp):
        
        self.bjd = bjd
        self.fnorm = fnorm
        self.efnorm = efnorm
        self.sectors = sectors
        self.qual_flags = qual_flags
        self.texp = texp
        
        self.gauss_detrendeded = False
        self.median_detrendeded = False
        self.detrended = False
        
        self.Prot = []
        
        self.fnorm_detrend = None       
        
    
    def get_splits(self, series, split_bjd=False):
        # Return each dataset in series split up by sectors
        bjd_diff = self.bjd[1:] - self.bjd[:-1]
        sector_diff = np.abs(self.sectors[1:] - self.sectors[:-1])

        # Find sector boundary points
        indeces = np.arange(len(sector_diff))

        sector_jumps = indeces[sector_diff > 1] # consecutive sectors are okay
        # Do we split based on BJD?
        if split_bjd:
            bjd_jumps = indeces[bjd_diff > 10]
            jump_indeces = np.union1d(bjd_jumps, sector_jumps) + 1
        else:
            jump_indeces = sector_jumps + 1
            
        jump_indeces = np.append(np.array(0), np.append(jump_indeces, 
                                                        np.array(len(self.bjd))))

        cut_series = [[] for i in range(len(series))]

        for i in range(len(jump_indeces)-1):
            for j in range(len(series)):
                cut_series[j].append(series[j][jump_indeces[i]:jump_indeces[i+1]])

        return cut_series
    
    ### Need to work on the detrending routines
    def detrend_lc(self):
        detrended = False
        gaussian_detrended = False
        median_detrended = False
        
        try:
            gaussian_detrended = self.gaussian_detrend_lc()
        except:
            median_detrended = self.median_detrend_lc()
    
        if not gaussian_detrended:
            median_detrended = self.median_detrend_lc()
            
        self.detrended = gaussian_detrended or median_detrended
        self.gaussian_detrended = gaussian_detrended 
        self.median_detrended = median_detrended
    
    
    def gaussian_detrend_lc(self):
        
        bjd_splits, fnorm_splits, efnorm_splits =\
                self.get_splits([self.bjd, self.fnorm, self.efnorm])
        
        full_fnorm_detrend = np.zeros(len(self.fnorm))
        index = 0
        detrended = True
        
        for i in range(len(bjd_splits)):
            bjd, fnorm, efnorm = bjd_splits[i], fnorm_splits[i],\
                                 efnorm_splits[i]
            residual_rotation, Prot = rotation_check(bjd, fnorm, efnorm)
            fnorm_detrend = fnorm.copy()
            count = 0
            
            while residual_rotation and count < 3:
                self.Prot.append(Prot)
                map_soln = build_model_SHO(bjd, fnorm_detrend, efnorm, Prot)
                count += 1
                fnorm_detrend -= map_soln["pred"]/1000
                
                residual_rotation, Prot = rotation_check(bjd, fnorm_detrend, 
                                                         efnorm)
            
            detrended = not residual_rotation and detrended
            full_fnorm_detrend[index:index+len(fnorm_detrend)] = fnorm_detrend
            index += len(fnorm_detrend)
            
        self.fnorm_detrend = full_fnorm_detrend
        self.gauss_detrendeded = False
        self.detrended = detrended
        
        return self.detrended

        
    
    def median_detrend_lc(self, window_length=50):
        # Find indeces corresponding to different sectors
        fnorm_splits = self.get_splits([self.fnorm])[0]
        
        # Median 
        full_fnorm_detrend = np.zeros(len(self.fnorm))
        index = 0
        detrended = True
        k = window_length // 2
        
        for i in range(len(fnorm_splits)):
            fnorm = fnorm_splits[i]
            fnorm_detrend = np.zeros(len(fnorm))
            
            for j in range(len(fnorm)):
                fnorm_slice = fnorm[max([0,j-k]):min([len(fnorm_detrend),j+k])]
                fnorm_detrend[j] = fnorm[j]/np.median(fnorm_slice)
                
            full_fnorm_detrend[index:index+len(fnorm_detrend)] = fnorm_detrend
            index += len(fnorm_detrend)
            
        self.median_detrended = True
        self.detrended = True
        self.fnorm_detrend = full_fnorm_detrend
        return self.detrended
    
    
    def plot_curve(self, series, ax_labels=None, show=True, savefig=None):
        bjd_start = self.bjd[0]
        series_splits = self.get_splits(series + [self.bjd])
        nrows, ncols = len(series_splits)-1, len(series_splits[0])

        fig, axs = plt.subplots(figsize=(ncols*6, nrows*3), 
                                sharex='col', sharey='row',
                                nrows=nrows, ncols=ncols, 
                                gridspec_kw={"wspace":0.02, "hspace":0.05},
                                squeeze=False)
        
        if not ax_labels:
            ax_labels = ['']*nrows

        for i in range(nrows):
            axs[i][0].set_ylabel(ax_labels[i])
            for j in range(ncols):
                bjd = series_splits[-1][j] - bjd_start
                axs[i][j].scatter(bjd, series_splits[i][j], s=0.1)
                axs[i][j].set_xlim(min(bjd), max(bjd))
                axs[i][j].grid()

        for j in range(ncols):
            axs[-1][j].set_xlabel("Days since first observation")

        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)
    
    
    def plot_results(self, results, phase_range=0.1, fnorm_range=None, 
                     show=True, savefig=None):
        nrows = len(results)
        fig, axs = plt.subplots(nrows=nrows, ncols=1, 
                                figsize=(6,3*nrows),
                                sharex='col', sharey=True,
                                gridspec_kw={"wspace":0.02, "hspace":0.02},
                                squeeze=False)

            
        for i in range(nrows):
            folded_t = phase_fold(self.bjd, results[i].period, results[i].T0)
            axs[i].scatter(folded_t, self.fnorm_detrend, s=0.2)

            model_t = phase_fold(results[i]["model_lightcurve_time"], 
                                 results[i]['period'], results[i].T0)
            sorts = np.argsort(model_t)
            axs[i].plot(model_t[sorts], 
                        results[i]["model_lightcurve_model"][sorts], 
                        color='r', ls="--")
            axs[i].set_xlim(-phase_range/2, phase_range/2)
            axs[i].grid()
            axs[i].set_ylabel(r"$F_{norm}$")

            f_sorts = np.argsort(folded_t)

            phase_series = bin_curve(folded_t[f_sorts], 
                                     self.fnorm_detrend[f_sorts],
                                     self.efnorm[f_sorts], 
                                     even_bins=True,
                                     bin_length = 0.0025)
            bin_phase, bin_fnorm, bin_efnorm = phase_series

            axs[i].errorbar(bin_phase, bin_fnorm, bin_efnorm,
                            ls='', marker='.')

        axs[-1].set_ylabel("Phase")

        if fnorm_range:
            axs[0].set_ylim(*fnorm_range)

        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)
            

class TIC_LightCurve(LightCurve): 
    
    def __init__(self, tic):
                
        # Make sure get tess data 
        lc_params = get_tess_data(tic)
        bjd, fnorm, efnorm, sectors, qual_flags, texp = lc_params
        
        LightCurve.__init__(self, bjd, fnorm, efnorm, sectors, qual_flags, texp)       
        
        
        
######## Helper Functions ########

        
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


# My functions

def sincurve(x, amp, T0, P, offset):
    return amp * np.sin(2*np.pi*(x-T0)/P) + offset


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


def find_transits(bjd, fnorm, threshold=6, max_iterations=5, **tls_kwargs):
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
    intransit = np.zeros(len(bjd), dtype=bool)
    
    # Look for thf first planet
    model = transitleastsquares(bjd, fnorm)
    result = model.power(**tls_kwargs)
    
    # Check if a planet candidate is found
    if result["SDE"] < threshold:
        return []
    
    result_list = [result]
    i = 1
    
    # Start looping finding more planets
    grazing = False
    while i <= max_iterations:
        # Mask found transit, | is logical or, and acumulates false
        intransit = intransit | transit_mask(bjd, result_list[-1].period, 
                                             2*result_list[-1].duration, 
                                             result_list[-1].T0)
        
        # Look for planets again with transits masked
        model = transitleastsquares(bjd[~intransit], fnorm[~intransit])
        result = model.power(**tls_kwargs, 
                             transit_template=['default', 'grazing'][grazing])
        
        # Check if planet found
        if result["SDE"] > threshold:
            result_list.append(result)
        elif not grazing: # Run a grazing template to see if we missed something?
            grazing = True
            continue
        else:
            break

        # Increment
        i += 1
        
    return result_list


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
