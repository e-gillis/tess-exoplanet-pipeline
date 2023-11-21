import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.optimize import curve_fit
import emcee
from multiprocessing import Pool

# Set TLS minimum grid for fitting
from transitleastsquares import tls_constants, catalog_info, transitleastsquares, transit_mask
tls_constants.MINIMUM_PERIOD_GRID_SIZE = 5

import get_tess_data as gtd
import detrending_modules as dt
import planet_search as ps
import vetting as vet
import mcmc_fitting as mc
import misc_functions as misc

from constants import *

VERSION = "0.3"

class TransitSearch:
    """
    A transit search object to collect lightcurves, search for transits and 
    characterize planet candidates. Stores information from each stage of this
    pipeline
    
    === Attributes ===
    tic: int
        TESS identification number. 
    lightcurves: List(LightCurve)
        List of TESS lightcurves of the target.
    Teff: float
        Effective temperature of the target star.
    logg: float
        log g of the target star.
    radius: float
        Radius of the target star.
    radius_err: float
        Radius uncertainty of the target star.
    mass: float
        Mass of the target star.
    mass_err: float
        Mass of the target star.
    RA: float
        Right Ascension of the target star.
    Dec: float
        Declination of the target star.
    u: (float, float)
        Quadratic limb darkening parameters of the taget star.
    results: List[List[result]]
        Nested list of tls result objects found with an iterative transit 
        search. Order matches the order of the lightcurves.
    results_tags: List[List[int]]
        Nested list of results tags identifying issues with the tls result
        objects.        
    planet_candidates: List[PlanetCandidate]
        List of credible planet candidates
    planet_candidates_reject: List[PlanetCandidate]
        List of planet candidates which have been rejected
    """
    
    def __init__(self, tic):
        """
        Initialization method for the TransitSearch
        
        === Arguments ===
        tic: int
            TESS identification number to retrieve information from
        """
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
        mass, mass_err, RA, Dec = gtd.get_star_info(tic)
        
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
        self.result_tags = None
        
        # Helpful for reproducing results
        self.star_params = (mass, radius, u)
        self.tls_kwargs = {"R_star": self.radius, "M_star": self.mass, "u": self.u,
                           "period_min": MIN_PERIOD, "period_max": MAX_PERIOD}
        
        # Space for planet Candidates
        self.planet_candidates = []
        self.planet_candidates_reject = []
        
    
    def transit_search(self, threshold=6, max_iterations=4, threads=TLS_THREADS,
                       grazing_search=True, progress=True):
        """ 
        Iteratively search for periodic transit-like features in each detrended
        TESS lightcuvre. Save each probable result in the results attribute
        
        === Arguments ===
        threshold: float
            SDE threshold cutoff for saved in the TLS result object. Default: 6
        max_iterations: int
            maximum number of iterations to run the TLS for. Default: 5
        threads: int
            Number of threads to use in the transit search. Default: TLS_THREADS
            from the consants file
        grazing_search: bool
            If true, run grazing TLS searches after normal searches fail to
            find features. Default: True
        progress: bool
            If true, show a progress bar while running the TLS. Default: True
        """
        self.results = []
        
        for lc in self.lightcurves:
            results_list = ps.find_transits(lc.bjd,lc.fnorm_detrend,lc.efnorm,
                                            (self.radius, self.mass, self.u),
                                            grazing_search=grazing_search,
                                            threshold=SDE_CUTOFF,
                                            max_iterations=max_iterations,
                                            show_progress_bar=progress, 
                                            threads=threads,
                                            method=MASK_METHOD,
                                            **self.tls_kwargs)
            
            self.results.append(results_list)
    
    
    def vet_results(self):
        """
        Iterate throigh TLS results and construct a vetting array using the
        vetting criteria below
        """
        if self.results is None:
            print("Must run TLS before vetting results")
        
        self.result_tags = []
    
        for i in range(len(self.lightcurves)):
            lc, results = self.lightcurves[i], self.results[i]
            vetting_array = np.zeros(len(results))
            
            # Check for correlation with main period in detrended
            vetting_array += vet.rotation_signal(lc, results)
            
            # Check that TLS spectrums are good?
            vetting_array += vet.bad_tls_spectrum(results)
            
            # Odd Even Mismatch
            vetting_array += vet.odd_even_mismatch(results)
            
            # SNR cut
            vetting_array += vet.low_snr(results, lc, cutoff=SNR_VET)
            
            # Check TLS edges
            # vetting_array += vet.tls_edge(results)

            vetting_array += vet.inf_period_uncertainty(results)
            
            self.result_tags.append(vetting_array)
            
        # Check for results across different sectors
        lc_lengths = np.array([lc.bjd[-1]-lc.bjd[0] 
                               for lc in self.lightcurves])
        sector_vet = vet.half_sectors_or_more(self.results, lc_lengths)
        
        for i in range(len(self.result_tags)):
            self.result_tags[i] += sector_vet[i]
    
     
    def cc_results(self, vet_results=True):
        """
        Cut and correlate results together based on common periods, depths
        and durations. Optionally run result vetting
        
        === Arguments ===
        vet_results: bool
            If true, run result vetting before correlating results. 
            Default: True
            
        === Returns ===
        correlated_results: List[List[result]]
            Nested list of TLS results which have passed all of the vetting 
            criteria. Each list s
        """
        if vet_results:
            self.vet_results()
        
        cut_results_list = vet.cut_results(self.results, self.result_tags)
        correlated_results = vet.correlate_results(cut_results_list, 
                             ptol=P_TOL, depthtol=DEPTH_TOL, durtol=DUR_TOL)
        
        return correlated_results
    
    
    def get_planet_candidates(self, mask_planets=False, progress=True):
        """
        Find and characterize planet candidates from the set of TLS results.
        
        === Arguments ===
        progress: bool
            If true, show progress bar while running the mcmc fitting for each
            planet candidate over the SNR threshold.
        """
        correlated_results = self.cc_results(vet_results=True)
        planet_candidates = []
        planet_candidates_reject = []
        
        # Collect Planet Candidates
        for c_results in correlated_results:
            pc = PlanetCandidate(self, c_results)
            pc.fit_planet_params(mask_others=mask_planets)
            
            if pc.snr > SNR_MCMC and not np.isnan(pc.snr):
                planet_candidates.append(pc)
            elif pc.snr <= SNR_MCMC and not np.isnan(pc.snr):
                planet_candidates_reject.append(pc)
        
        # Find PCs with overlapping transits
        full_bjd = np.concatenate([lc.bjd for lc in self.lightcurves])
        snr_sorts = np.argsort([pc.snr for pc in planet_candidates])[::-1]
        sorted_pcs = [planet_candidates[i] for i in snr_sorts]
        
        # Check for overlap
        # pcs, cut_pcs = vet.pc_overlap(sorted_pcs, full_bjd)
        # self.planet_candidates.extend(pcs)
        # self.planet_candidates_reject.extend(cut_pcs+planet_candidates_reject)
        
        self.planet_candidates.extend(sorted_pcs)
        self.planet_candidates.extend(planet_candidates_reject)
        
        # Run MCMC
        for pc in self.planet_candidates:
            pc.run_mcmc(nsteps=ITERATIONS, burn_in=BURN_IN, progress=progress, 
                        mask_others=mask_planets)
    
    
    def plot_transits(self):
        raise NotImplementedError
        
        
    def save(self, filename):
        """
        Save this TransitSearch object at filename with the extension .ts. 
        TransitSearch objects can be reloaded with the load_ts function.
        
        === Arguments ===
        filename: string
            filename to which the TransitSearch object is saved.
        """
        # Set version
        self.version = VERSION
        
        with open(filename+'.ts', "wb") as f:
            pickle.dump(self, f)
        return None
        
        


class LightCurve:
    """
    Lightcurve object to store timeseries data taken from the TESS survey
    
    === Attributes ===
    bjd: numpy array
        bjd time for each datapoint
    fnorm: numpy array
        Normalized flux recorded by TESS
    efnorm: numpy array
        Error on the normalized flux recorded by TESS
    sectors: numpy array
        Sector for each exposure
    qual_flags: numpy array
        Quality flags for each exposure
    texp: numpy array
        Exposure time for each exposure
    detrend_methods: List[string]
        List of methods used to detrend the lightcurve 
    fnorm_detrend: numpy array
        Detrended flux using various methods recorded in detrend_methods
    """
    
    def __init__(self, bjd, fnorm, efnorm, sectors, qual_flags, texp):
        
        self.bjd = bjd
        self.fnorm = fnorm
        self.efnorm = efnorm
        self.sectors = sectors
        self.qual_flags = qual_flags
        self.texp = texp
        
        self.detrended = False
        self.detrend_methods = []
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
    def detrend_lc(self, split=False):
        # Check for rotation, if detected try GP
        if not split:
            gaussian_detrended = False      
            try:
                gaussian_detrended = self.gaussian_detrend_lc()
            except:
                pass

            if not gaussian_detrended:
                median_detrended = self.median_detrend_lc()
            else:
                median_detrended = self.median_detrend_lc(cont=True)

        else:
            bjd_s, fnorm_s, efnorm_s = self.get_splits([self.bjd, self.fnorm,
                                                        self.efnorm])
            fnorm_detrend_s = []
            
            for i in range(len(bjd_s)):
                bjd, fnorm, efnorm = bjd_s[i], fnorm_s[i], efnorm_s[i]
                gd = False
                try:
                    fnorm_detrend, gd = dt.gaussian_detrend(bjd, fnorm, efnorm)
                    if 'gaussian' not in self.detrend_methods:
                        self.detrend_methods.append('gaussian')
                except:
                    pass
                
                if not gd:
                    fnorm_detrend = dt.median_detrend(bjd, fnorm, efnorm)
                    if 'median' not in self.detrend_methods:
                        self.detrend_methods.append('median')
                
                fnorm_detrend_s.append(fnorm_detrend)
            
            self.fnorm_detrend = np.concatenate(fnorm_detrend_s)
                
        # Sigma Clip with noise
        clip = misc.upper_sigma_clip(self.fnorm_detrend, sig=5)
        self.fnorm_detrend[~clip] = np.random.normal(loc=1, 
                                    scale=np.std(self.fnorm_detrend[clip]),
                                    size=sum(~clip))

        # Normalize detrended curve
        self.fnorm_detrend += 1 - np.median(self.fnorm_detrend)
            
            
    # Lightcurve is already split!
    def gaussian_detrend_lc(self, cont=False):
        if cont:
            fnorm = self.fnorm_detrend
        else:
            fnorm = self.fnorm
        
        self.fnorm_detrend, self.detrended = dt.gaussian_detrend(self.bjd, 
                                             fnorm, self.efnorm)
        
        if self.detrended:
            self.detrend_methods.append('gaussian')
        
        return self.detrended
    
    
    def median_detrend_lc(self, window_length=360, cont=False):
        if cont:
            fnorm = self.fnorm_detrend
        else:
            fnorm = self.fnorm
        
        self.fnorm_detrend, self.detrended = dt.median_detrend(fnorm, 
                                                        window_length)
        self.detrend_methods.append('median')
        return self.detrended
    
    
    def spline_detrend_lc(self, cont=False):
        if cont:
            fnorm = self.fnorm_detrend
        else:
            fnorm = self.fnorm
            
        self.fnorm_detrend, self.detrended = dt.spline_detrend(self.bjd, fnorm,
                                                               self.efnorm)
        self.detrend_methods.append('spline')
        return self.detrended
    
    
    def plot_curve(self, series, ax_labels=None, show=True, bjd_start=None,
                   savefig=None):
        if bjd_start is None:
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

        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=400)
        
        if show:
            plt.show()

    
    
    def plot_results(self, results, fnorm_range=None, show=True, savefig=None):
        nrows = len(results)
        fig, axs = plt.subplots(nrows=nrows, ncols=1, 
                                figsize=(6,3*nrows),
                                sharex='col', sharey=True,
                                gridspec_kw={"wspace":0.02, "hspace":0.02},
                                squeeze=True)

            
        for i in range(nrows):
            folded_t = misc.phase_fold(self.bjd, results[i].period, 
                                       results[i].T0)
            axs[i].scatter(folded_t, self.fnorm_detrend, s=0.2)

            model_t = misc.phase_fold(results[i]["model_lightcurve_time"], 
                                      results[i]['period'], results[i].T0)
            sorts = np.argsort(model_t)
            axs[i].plot(model_t[sorts], 
                        results[i]["model_lightcurve_model"][sorts], 
                        color='r', ls="--")
            axs[i].set_xlim(-results[i]['duration'], results[i]['duration'])
            axs[i].grid()
            axs[i].set_ylabel(r"$F_{norm}$")

            f_sorts = np.argsort(folded_t)

            phase_series = misc.bin_curve(folded_t[f_sorts], 
                                     self.fnorm_detrend[f_sorts],
                                     self.efnorm[f_sorts], 
                                     even_bins=True,
                                     bin_length = results[i]['duration']/10)
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
        lc_params = gtd.get_tess_data(tic)
        bjd, fnorm, efnorm, sectors, qual_flags, texp = lc_params
        
        LightCurve.__init__(self, bjd, fnorm, efnorm, sectors, 
                            qual_flags, texp)    
        
        

class PlanetCandidate:
    """
    A Planet Candidate object to model candidate planets and collect planet
    characteristics. Must be linked to a Transit Search object to provide 
    lightcurves and host star parameters.
    
    === Attributes ===
    ts: TransitSearch
        Transit search object to provide lightcurves and star parameters.
    correlated_results: List[TLS results]
        List of transitleastsquares results to provide the base for the transit
        search
    best_result: TLS results
        transitleastsquares result that best characterizes this planet candidate
    T0: float
        BJD of the first transit midtime in the lightcurves in self.ts
    P: float
        Period of the planet candidate
    Rp: float
        Ratio of the planet's radius to the star's radius. Stellar radius and 
        uncertainty are is accessible through self.ts.radius and 
        self.ts.radius_err
    b: float
        Impart parameter of the transit
    offset: float
        Offset of lightcurve ephemeris during transit from 1
    duration: float
        Duration of the transit in days
    snr: float
        Signal to noise ratio of the planet candidate
    mcmc_full_chain: numpy array
        Full unflattened MCMC chain with the trajectory of each walker over
        the 5 mcmc parameters
    mcmc_chain: numpy array
        Flatterned mcmc chain with burn in removed. Appropriate for 
        characterizing the posterior distribution of planet parameters
    priors: List[scipy.stats._continuous_distns]
        Prior distributions for each of the 5 parameters
    """
    def __init__(self, ts, correlated_results):
        """
        Initialization method for the PlanetCandidate
        
        === Arguments ===
       ts: TransitSearch
            Transit search object to provide lightcurves and star parameters.
        correlated_results: List[TLS results]
            List of transitleastsquares results to provide the base for the transit
            search
        """
        # Get Data in this class
        self.ts = ts
        self.results = correlated_results
        self.best_result = None
        
        # Planet Properties
        self.T0 = None
        self.period = None
        self.Rp = None
        self.b = None
        self.offset = None
        self.duration = None
        self.snr = None
        
        # MCMC Things
        self.full_mcmc_chain = None
        self.mcmc_chain = None
        self.priors = None


    def mask_planet(self, bjd, fnorm):
        """Mask this planet candidate out of a given timeseries, return the
        masked timeseries.
        
        === Arguments ===
        bjd: numpy array
            bjd time for each datapoint
        fnorm: numpy array
            Normalized flux at each datapoing
            
        === Returns ===
        masked_fnorm: 
            Normalized flux with planet candidate ephemeris removed
        """
        if self.period is None:
            print("Planet Parameters have not been modeled!")
            return fnorm

        # Make the model
        pmodel = misc.batman_model(bjd, self.T0, self.period, self.Rp, self.b, 
                                   self.ts.radius, self.ts.mass, self.ts.u,
                                   offset=self.offset)
        # Subtract the model
        masked_fnorm = fnorm - pmodel + 1
        return fnorm
        
    
    def fit_planet_params(self, mask_others=False):
        """
        Find a best fit model for planet parameters and assign parameters to
        the appropriate attributes. Best fit model is strongly informed by the
        best result of the planet search
        
        === Arguments ===
        mask_others: Boolean, default False
            If True, mask other planet candidates in the transit search (except
            this one!) out of the timeseries before proceeding with fitting
        """
        # Get lightcurves from the transit search object
        bjd, fnorm, efnorm = np.concatenate([lc.bjd for lc in 
                                             self.ts.lightcurves]),\
             np.concatenate([lc.fnorm_detrend for lc in self.ts.lightcurves]),\
             np.concatenate([lc.efnorm for lc in self.ts.lightcurves])
        R, M, u = self.ts.radius, self.ts.mass, self.ts.u
        
        # Mask Existing planet candidates who don't match this one!
        if mask_others:
            for pc in self.ts.planet_candidates:
                # Comparing results
                if [r.period for r in pc.results]!=\
                   [r.period for r in self.results]:
                    fnorm = pc.mask_planet(bjd, fnorm)
        
        # Get period from correlated results
        SNRs = np.array([result.snr for result in self.results])
        
        # Take max right now, maybe do SNR later
        best_result = self.results[np.argmax(SNRs)] 
        
        P, P_delta = best_result.period, best_result.period_uncertainty
        
        # Fit period with limited TLS
        model_full = transitleastsquares(bjd, fnorm)
        best_result = model_full.power(period_min=P-P_delta, 
                                       period_max=P+P_delta,
                                       R_star=self.ts.radius, 
                                       M_star=self.ts.mass, 
                                       u=self.ts.u, 
                                       show_progress_bar=False)
        if np.isnan(best_result.period):
            print("Fitting Failed, no transits found by TLS")
            return None
        self.best_result = best_result

        # Get a best-fit model
        self.T0, self.period, self.Rp, self.b, self.offset =\
            ps.fit_transit_model(bjd, fnorm, efnorm, best_result, 
                                 (R, M, u), durcheck=False)

        depth = self.Rp**2
        
        # Compute the duration
        self.duration = misc.transit_duration(M, R, self.period, 
                                              self.Rp, self.b)
        
        # Count the transits
        N = misc.get_Ntransits(self.period, self.T0, self.duration, bjd)
        
        # Get the noise
        binned_fnorm = misc.bin_curve(bjd, fnorm, efnorm, even_bins=True, 
                                      bin_length=self.duration)[1]
        noise = np.std(binned_fnorm)
        self.snr = depth / noise * N**0.5
    
    
    def run_mcmc(self, nsteps=4000, nwalkers=48, burn_in=2000, progress=True,
                 mask_others=True):
        """
        Run a Monte Carlo Markov Chain to characterize the posterior 
        distribution of planet parameters. self.fit_planet_params must be run
        before this method. 
        
        === Arguments ===
        nsteps: int
            Number of steps for the chain to take
        nwalkers: int
            Number of walkers to explore the parameter space
        burn_in: int
            Length of the burn in of the chain to be chopped off
        progress: Boolean, default True
            If True, show progress bar of MCMC
        """
        # Prepare priors and walker positions
        priors, lc_arrays, walkers = mc.ps_mcmc_prep(self, self.ts, nwalkers,
                                                     mask_others=True)
        
        star_params = (self.ts.radius, self.ts.radius_err, 
                       self.ts.mass, self.ts.mass_err, self.ts.u)
        self.priors = priors
        
        # Run the MCMC
        with Pool(TLS_THREADS) as pool:
            print(len(priors))
            ensam = emcee.EnsembleSampler(nwalkers, len(priors), mc.transit_log_prob,
                                          pool=pool, args=(star_params,lc_arrays,priors))
            ensam.run_mcmc(walkers, nsteps=nsteps, progress=progress)
        
        # Save the chain
        self.full_mcmc_chain = ensam.get_chain()
        self.mcmc_chain = self.full_mcmc_chain[burn_in:].reshape((-1, len(priors)))
        
        # Update Parameters
        # self.T0, self.P, self.Rp, self.b = np.median(self.mcmc_chain, axis=0)
        # R, M = self.ts.radius, self.ts.mass
        # self.duration = misc.transit_duration(M, R, self.period, 
        #                                       self.Rp, self.b)      

        
    def plot_results(self, savefig=None, show=True, title=None):
        fig, axs = plt.subplots(ncols=1, nrows=len(self.results), 
                                figsize=(6, 3*len(self.results)),
                                squeeze=False)
        
        for i, result in enumerate(self.results):
            misc.plot_result(result, show=False, fig=fig, ax=axs[0][i])
            
        if savefig is not None:
            fig.savefig(savefig, bbox_inches='tight')
        if show:
            plt.show()
            
    
    def highlight_transits(self, title=None, show=True, savefig=None):
        """
        Plot the timeseries from each lightcurve and highlight transits in
        the timeseries based on the characterized parameters. 

        === Arguments ===
        title: str or None
            If not None, title of the plot. Default: None
        show: bool
            If True, show the plot after it is created. Default: True
        savefig: str or None
            If not None, path to save the plot at. Default: None
        """
        fig, axs = plt.subplots(ncols=1, nrows=len(self.ts.lightcurves),
                                figsize=(6, 3*len(self.ts.lightcurves)))

        for i, lc in enumerate(self.ts.lightcurves):
            axs[i].scatter(lc.bjd, lc.fnorm_detrend, color="tab:blue", s=0.1)
            mask = transit_mask(lc.bjd, self.period, self.duration, self.T0)
            
            axs[i].scatter(lc.bjd[mask], lc.fnorm_detrend[mask], 
                           color="tab:orange", s=2)
            
            axs[i].set_ylabel("Detrended Flux")
            axs[i].set_xlim(min(lc.bjd), max(lc.bjd))
            axs[i].grid()

        axs[-1].set_xlabel("BJD")
        
        if savefig is not None:
            fig.savefig(savefig, bbox_inches='tight')
        if show:
            plt.show()
            
    
    
    def chain_evos_plot(self, savefig=None, show=True, title=None):
        if self.full_mcmc_chain is None:
            print("run_mcmc method must be run first!")
            return None

        mc.plot_chain_evo(self.full_mcmc_chain,
                       title=title, savefig=savefig, show=show)


    def chain_dists_plot(self, savefig=None, show=True, title=None):
        if self.mcmc_chain is None:
            print("run_mcmc method must be run first!")
            return None
        
        mc.plot_chain_dists(self.mcmc_chain, self.priors, 
                            title=title, savefig=savefig, show=show)
        
    def model_plot(self, savefig=None, show=True, title=None):
        if self.mcmc_chain is None:
                print("run_mcmc method must be run first!")
                return None
        
        mc.plot_model(self, self.ts,savefig=savefig, show=show, title=title)
    
    def corner_plot(self, savefig=None, show=True, title=None):
        if self.mcmc_chain is None:
            print("run_mcmc method must be run first!")
            return None
        
        mc.plot_chain_corner(self.mcmc_chain, 
                             savefig=savefig, show=show, title=title)
    
    
### Loading back objects

def load_ts(path):      
    with open(path, "rb") as f:
        loaded_ts = pickle.load(f)
    return loaded_ts

    