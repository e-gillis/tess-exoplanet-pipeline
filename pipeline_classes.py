import matplotlib.pyplot as plt
import numpy as np
import pickle
from inspect import ismethod

from scipy.optimize import curve_fit
import emcee
from multiprocessing import Pool

# Set TLS minimum grid for fitting
from transitleastsquares import tls_constants, catalog_info, period_grid
from transitleastsquares import transitleastsquares, transit_mask
tls_constants.MINIMUM_PERIOD_GRID_SIZE = 2

from exofop.download.identifiers import TIC
import math

# Internal Package Imports
import get_tess_data as gtd
import detrending_modules as dt
import planet_search as ps
import vetting as vet
import mcmc_fitting as mc
import misc_functions as misc

from constants import *

VERSION = "0.4"

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
    planet_candidates_plausible: List[PlanetCandidates]
        List of plausible but poor fitting planet candidates for manual review
    
    lcs, pcs, pcs_r, pcs_p:
        Aliases for lightcurves, planet_candidates, planet_candidates_reject
        and planet_candidates_plausible
    """
    
    def __init__(self, tic, detrend=True):
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
            if detrend:
                self.lightcurves[-1].detrend_lc()
        
        # Star Parameters
        Teff, logg, radius, radius_err,\
        mass, mass_err, RA, Dec = gtd.get_star_info(tic)
        
        assert not (np.isnan(radius) and np.isnan(mass)),\
        "Radius and Mass cannot both be NaN, catalog query failed"
        
        
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
        
        # Check For NaNs
        if np.isnan(self.mass_err):
            self.mass_err = 0
        if np.isnan(self.radius_err):
            self.radius_err = 0
        if np.isnan(self.radius):
            self.radius, self.radius_err =misc.m_from_r(self.mass,self.mass_err)
        if np.isnan(self.mass):
            self.mass, self.mass_err =misc.r_from_m(self.radius,self.radius_err)
        
        
        self.results = None
        self.result_tags = None
        
        # Helpful for reproducing results
        self.star_params = (mass, radius, u)
        self.tls_kwargs = {"R_star": self.radius, "M_star": self.mass, 
                           "u": self.u, "period_min": MIN_PERIOD, 
                           "period_max": MAX_PERIOD}
        
        # Space for planet Candidates
        self.planet_candidates = []
        self.planet_candidates_reject = []
        self.planet_candidates_plausible = []
        
        # Set Aliases
        self.lcs, self.pcs, self.pcs_r, self.pcs_p =\
        self.lightcurves, self.planet_candidates,\
        self.planet_candidates_reject, self.planet_candidates_plausible
        
        
    
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
            vetting_array += vet.duplicate_period(results)
            
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
        # Reset planet candidate search
        self.clear_pcs()
        
        # Correlate Results
        correlated_results = self.cc_results(vet_results=True)
        # planet_candidates = []
        # planet_candidates_reject = []
        
        if sum([len(c) for c in correlated_results]) == 0:
            return None
        
        # Collect Planet Candidates
        for c_results in correlated_results:
            if len(c_results) == 0:
                continue
            
            pc = PlanetCandidate(self, c_results)
            pc.fit_planet_params(mask_others=mask_planets)
            
            # Planet candidates need to be added here to mask them!
            if not np.isnan(pc.snr) and pc.snr > SNR_MCMC:
                self.planet_candidates.append(pc)
            elif not np.isnan(pc.snr) and pc.snr <= SNR_MCMC:
                self.planet_candidates_reject.append(pc)
                pc.flags.append("Low SNR")
            else:
                self.planet_candidates_reject.append(pc)
                pc.flags.append("NaN SNR")
        
        # Find PCs with overlapping transits
        full_bjd = np.concatenate([lc.bjd for lc in self.lightcurves])
        snr_sorts = np.argsort([pc.snr for pc in self.planet_candidates])[::-1]
        sorted_pcs = [self.planet_candidates[i] for i in snr_sorts]
                
        # Check for overlap
        # pcs, cut_pcs = vet.pc_overlap(sorted_pcs, full_bjd)
        # self.planet_candidates = pcs
        # self.planet_candidates_reject.extend(cut_pcs)
        
        # Run MCMC
        i = 0
        while i < len(self.pcs):
            try:
                self.pcs[i].run_mcmc(nsteps=ITERATIONS, burn_in=BURN_IN, 
                            progress=progress, mask_others=mask_planets)
                
            except ValueError:
                self.pcs[i].flags.append("MCMC Failed, ValueError")
                self.pcs_p.append(self.pcs.pop(i))
                continue
            
            i += 1

        # Rejection loop instead of just one pass?
        rejected_last = True
        while rejected_last:
            i = 0
            rejected_last = False
            while i < len(self.pcs):
                ilast = i

                if self.pcs[i].deltaBIC_model(use_mcmc_params=True, use_offset=True) > -10:
                    self.pcs[i].flags.append("Offset Null Model Favoured")
                    self.pcs_p.append(self.pcs.pop(i))
                elif self.pcs[i].deltaBIC_model(use_mcmc_params=True, use_offset=False) > -10:
                    self.pcs[i].flags.append("Median Null Model Favoured")
                    self.pcs_p.append(self.pcs.pop(i))
                elif pc.snr > 10:
                    i += 1
                elif self.pcs[i].red_chi2_model(dfrac=3, use_mcmc_params=True) > 1.5:
                    self.pcs[i].flags.append("Poor fit by chi2 test")
                    self.pcs_p.append(self.pcs.pop(i))
                elif self.pcs[i].KS_residuals(use_mcmc_params=True) < 0.2 and\
                    self.pcs[i].flags.append("Non-Gaussian White Noise")
                    self.pcs_p.append(self.pcs.pop(i))
                else:
                    i += 1

                rejected_last = rejected_last or (i == ilast)
                
                
    
    def plot_transits(self):
        raise NotImplementedError
        
    
    def plot_model(self, batman_params, savefig=None, show=True, title=None):
        """Plot a transit model given parameters
        """
        T0, P, Rp, b, offset = batman_params
        R, M, u = self.radius, self.mass, self.u
        
        plt.figure(figsize=(12,4))
        bjd, fnorm, efnorm = np.concatenate([lc.bjd for lc in self.lightcurves]),\
                np.concatenate([lc.fnorm_detrend for lc in self.lightcurves]),\
                np.concatenate([lc.efnorm for lc in self.lightcurves])
        
        bjd_folded = (bjd - T0 + P/2) % P - P/2
        duration = misc.transit_duration(M, R, P, Rp, b)
        sort = np.argsort(bjd_folded)
        bjd_folded, fnorm, efnorm = bjd_folded[sort], fnorm[sort], efnorm[sort]
         
        bin_bjd, bin_fnorm, bin_efnorm =  misc.bin_curve(bjd_folded, fnorm, efnorm,
                                                even_bins=True, 
                                                bin_length=duration/8)
        
        plt.scatter(bjd_folded, fnorm, s=1)
        plt.errorbar(bin_bjd, bin_fnorm, bin_efnorm, ls='', 
                     capsize=3, marker='.', color='red')

        bm_curve = misc.batman_model(bjd_folded, 0, P, Rp, b, R, M, u, offset)
        plt.plot(bjd_folded, bm_curve, color='k', lw=4)

        plt.xlim(-1.5*duration, 1.5*duration)
        plt.ylabel("Normalized Flux")
        plt.xlabel("Days Since Transit Middle")

        if title:
            plt.title(title)

        if savefig:
            plt.savefig(savefig, bbox_inches='tight')

        if show:
            plt.show()
            
    
    def clear_pcs(self):
        """Clear planet candidates and aliases to rerun panet fitting
        """
        self.planet_candidates = []
        self.planet_candidates_plausible = []
        self.planet_candidates_reject = []
        
        self.pcs, self.pcs_p, self.pcs_r = self.planet_candidates,\
                                           self.planet_candidates_plausible,\
                                           self.planet_candidates_reject
        
        
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
    trend: numpy array
        Trend which is subtracted from the normalized lighttcurve to obtain the
        detrended lightcurve. Trend does not contain information about 
        resampled sigma clipped points and flares.
    """
    
    def __init__(self, bjd, fnorm, efnorm, sectors, qual_flags, texp):
        
        dat_arrays = [fnorm, efnorm, sectors, qual_flags, texp]
        assert np.all([len(a) == len(bjd) for a in dat_arrays]),\
               "Lightcurve data arrays must all be the same length"

        self.bjd = bjd
        self.fnorm = fnorm
        self.efnorm = efnorm
        self.sectors = sectors
        self.qual_flags = qual_flags
        self.texp = texp
        
        self.detrended = False
        self.detrend_methods = []
        self.fnorm_detrend = None
        self.trend = None
        
    
    def get_splits(self, series, split_bjd=False, consecutive_sectors=True):
        # Return each dataset in series split up by sectors
        bjd_diff = self.bjd[1:] - self.bjd[:-1]
        sector_diff = np.abs(self.sectors[1:] - self.sectors[:-1])

        # Find sector boundary points
        indeces = np.arange(len(sector_diff))

        sec_dif = int(consecutive_sectors)
        sector_jumps = indeces[sector_diff > sec_dif] # consecutive sectors are okay
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

        # Save the trend before sigma clipping and masking
        self.trend = self.fnorm - self.fnorm_detrend

        # Mask flares with noise:
        flare_mask = dt.mask_flares(self.fnorm_detrend, self.bjd, width=100)
        self.fnorm_detrend[~flare_mask] = np.random.normal(loc=1, 
                                          scale=np.std(self.fnorm_detrend[flare_mask]),
                                          size=sum(~flare_mask))   
                
        # Sigma Clip with noise
        clip = misc.upper_sigma_clip(self.fnorm_detrend, sig=5)
        self.fnorm_detrend[~clip] = np.random.normal(loc=1, 
                                    scale=np.std(self.fnorm_detrend[clip]),
                                    size=sum(~clip))        

        # Normalize detrended curve
        self.fnorm_detrend += 1 - np.median(self.fnorm_detrend)
        
        
    def mask_flares(self, n, nsigma, method='noise'):
        assert method in ["noise", "remove"],\
               "method argument must be 'noise' or 'remove'"
        
        # Split based on sector for more uniform noise properties
        fnorms = self.get_splits([self.fnorm_detrend], 
                                 consecutive_sectors=False)[0]
        if method == 'noise':
            np.random.seed(42)
            for i in range(len(fnorms)):
                fnorm = fnorms[i]
                f_mask = dt.flare_mask(fnorm, n, nsigma)
                fnorm[f_mask] = np.random.normal(loc=np.mean(fnorm[~f_mask]),
                                                 scale=np.std(fnorm[~f_mask]))
                fnorms[i] = fnorm
            self.fnorm_detrend = np.concatenate(fnorms)
            return None
        
        flare_masks = []
        for fnorm in fnorms:
            flare_masks.append(dt.flare_mask(fnorm, n, nsigma))
        f_mask = np.concatenate(flare_masks)
        
        self.bjd = self.bjd[~f_mask]
        self.fnorm = self.fnorm[~f_mask]
        self.efnorm = self.efnorm[~f_mask]
        self.qual_flags = self.qual_flags[~f_mask]
        self.texp = self.texp[~f_mask]
        self.sectors = self.sectors[~f_mask]
                    
            
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
    flags: List[string]
        List of flags set externally to validate planet candidates
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
        self.flags = []
        
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
        return masked_fnorm
        
    
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
        SNRs = np.array([result.SDE for result in self.results])
        
        # do SNR, but could refine this
        best_result = self.results[np.argmax(SNRs)] 
        
        P, P_delta = best_result.period, best_result.period_uncertainty
        
        # Require at least 150 sample periods in the trial range
        oversampling_factor = 5
        time_span = bjd[-1]-bjd[0]
        while len(period_grid(self.ts.radius, self.ts.mass,time_span,
                              P-P_delta,P+P_delta,oversampling_factor)) < 150:
            oversampling_factor *= 2
                
        # Fit period with limited TLS
        model_full = transitleastsquares(bjd, fnorm, efnorm)
        best_result = model_full.power(period_min=P-P_delta, 
                                       period_max=P+P_delta,
                                       R_star=self.ts.radius, 
                                       M_star=self.ts.mass, 
                                       u=self.ts.u, 
                                       show_progress_bar=False,
                                       oversampling_factor=oversampling_factor)
        # If nothing was found by the TLS
        if np.isnan(best_result.period):
            print("Fitting Failed, no transits found by TLS")
            self.snr = np.nan
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
                                                     mask_others=mask_others)
        
        star_params = (self.ts.radius, self.ts.radius_err, 
                       self.ts.mass, self.ts.mass_err, self.ts.u)
        self.priors = priors
        
        # Run the MCMC
        with Pool(TLS_THREADS) as pool:
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

    ### Statistical Test Methods ###

    def _stat_test_prep(self, dfrac, use_mcmc_params, mask_others, fold_bjd):
        """Prep function to return the bjd, cut, fnorm, efnorm and model to run a 
        statistical test on
        """
        # Assemble the timeseries
        bjd, fnorm, efnorm =\
        np.concatenate([lc.bjd for lc in self.ts.lightcurves]),\
        np.concatenate([lc.fnorm_detrend for lc in self.ts.lightcurves]),\
        np.concatenate([lc.efnorm for lc in self.ts.lightcurves])
        
        # Mask Existing planet candidates who don't match this one!
        if mask_others:
            for pc in self.ts.planet_candidates:
                # Comparing results
                if [r.period for r in pc.results]!=\
                   [r.period for r in self.results]:
                    fnorm = pc.mask_planet(bjd, fnorm)
                
        # Variables for model
        P, T0, Rp, b, offset = self.period,self.T0,self.Rp,self.b,self.offset
        R, M, u = self.ts.radius, self.ts.mass, self.ts.u
        
        if use_mcmc_params:
            T0, P, Rp, b, offset = np.median(self.mcmc_chain, axis=0)
        
        # Fold and cut BJD
        bjd_folded = (bjd + P/2 - T0) % P - P/2
        cut = np.abs(bjd_folded) < dfrac*self.duration
        
        # Transit model and null model
        model = misc.batman_model(bjd, T0, P, Rp, b,
                                  R, M, u, offset)

        if fold_bjd:
            return bjd_folded, cut, fnorm, efnorm, model

        return bjd, cut, fnorm, efnorm, model

    
    def deltaBIC_model(self, dfrac=1, use_offset=True, use_mcmc_params=False, 
                       mask_others=True, pertransit=False):
        """Return the deltaBIC of the transit model being favored over a 
        constant with median equal to the median of the signal
        """
        bjd, cut, fnorm, efnorm, model = self._stat_test_prep(dfrac, 
                                         use_mcmc_params, mask_others, 
                                         fold_bjd=True)
        if use_mcmc_params:
            offset =  np.median(self.mcmc_chain, axis=0)[-1]
        else:
            offset = self.offset
        
        if use_offset:
            model_null = np.ones(len(model)) + offset
        else:
            model_null = np.ones(len(model))*np.median(fnorm[cut])

        if not pertransit:
            return misc.DeltaBIC(fnorm[cut], efnorm[cut], model[cut], 
                                 model_null[cut], k=5)

        ci = np.where(cut[:-1] != cut[1:])[0] + 1
        if cut[0]: ci = np.concatenate([[0], ci])
        if cut[-1]: ci = np.concatenate([ci, [len(cut)]])

        dBICs = np.zeros(len(ci) // 2)
        for i in range(0, len(ci), 2):
            j, k = ci[i], ci[i+1]
            dBIC = misc.DeltaBIC(fnorm[j:k], efnorm[j:k], model[j:k],
                                 model_null[j:k], k=5)
            dBICs[i//2] = dBIC
        return dBICs
        

    def deltaBIC_model_list(self, dfrac=1, use_offset=True, 
                            use_mcmc_params=False, mask_others=True):
        """
        """
        return None
    
    
    def red_chi2_model(self, dfrac=1, use_mcmc_params=True, mask_others=True,
                       pertransit=False):
        """Compute the reduced Chi-squared of the transit model
        """
        bjd, cut, fnorm, efnorm, model = self._stat_test_prep(dfrac, 
                                         use_mcmc_params, mask_others, 
                                         fold_bjd=True)

        if not pertransit:
            red_chi2 = sum(((fnorm[cut]-model[cut]) / efnorm[cut])**2) / (sum(cut)-5)
            return red_chi2

        ci = np.where(cut[:-1] != cut[1:])[0] + 1
        if cut[0]: ci = np.concatenate([[0], ci])
        if cut[-1]: ci = np.concatenate([ci, [len(cut)]])

        red_chi2s = np.zeros(len(ci) // 2)
        for i in range(0, len(ci), 2):
            j, k = ci[i], ci[i+1]
            red_chi2s[i//2] = sum(((fnorm[j:k]-model[j:k]) /\
                                   efnorm[j:k])**2) / (k-j-5)
        
        return red_chi2s

    
    def KS_residuals(self, dfrac=1, use_mcmc_params=True, mask_others=True,
                     pertransit=False):
        """Use the KS Test to determine if the residuals of the lightcurve are 
        gaussian when the transit model is removed
        """
        bjd, cut, fnorm, efnorm, model = self._stat_test_prep(dfrac, 
                                         use_mcmc_params, mask_others, 
                                         fold_bjd=True)
        fnorm_resid = fnorm - model
        
        if not pertransit:
            prob_D = dt.ks_noise_test(fnorm_resid[cut])
            return prob_D

        ci = np.where(cut[:-1] != cut[1:])[0] + 1
        if cut[0]: ci = np.concatenate([[0], ci])
        if cut[-1]: ci = np.concatenate([ci, [len(cut)]])

        prob_Ds = np.zeros(len(ci) // 2)
        for i in range(0, len(ci), 2):
            j, k = ci[i], ci[i+1]
            prob_Ds[i//2] = dt.ks_noise_test(fnorm_resid[j:k])
        return prob_Ds
        
    
    ### Plotting Methods ###
    
    def plot_results(self, savefig=None, show=True, title=None):
        """Plot the TLS results that motivate this planet candidate

        Add titles to each suplot with sector information
        """
        fig, axs = plt.subplots(ncols=1, nrows=len(self.results), 
                                figsize=(6, 3*len(self.results)),
                                squeeze=False)

        f_bjd, f_sector = np.concatenate([lc.bjd for lc in self.ts.lightcurves]),\
                          np.concatenate([lc.sectors for lc in self.ts.lightcurves])

        for i, result in enumerate(self.results):

            # Sector array at closest bjd to each transit time
            sectors = np.array([f_sector[np.argmin(abs(f_bjd-tt))] 
                                for tt in result.transit_times])
            sectors = np.unique(sectors)
            misc.plot_result(result, show=False, fig=fig, ax=axs[i][0], 
                             title_ext=f", Sectors: {str(list(sectors))[1:-1]}")
        plt.tight_layout()
            
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
        # This makes the figure as one column, remake to n columns
        fig, axs = plt.subplots(ncols=len(self.ts.lightcurves), nrows=2,
                                figsize=(6*len(self.ts.lightcurves), 6),
                                sharey='row', sharex='col', 
                                gridspec_kw={"wspace":0.02, "hspace":0.05},
                                squeeze=False)
    
        for i, lc in enumerate(self.ts.lightcurves):
    
            # Plot detrended lightcurves with intransit points orange        
            axs[1][i].scatter(lc.bjd, lc.fnorm_detrend, color="tab:blue", 
                              s=0.1, alpha=0.7)
            mask = transit_mask(lc.bjd, self.period, self.duration, self.T0)
            
            axs[1][i].scatter(lc.bjd[mask], lc.fnorm_detrend[mask], 
                           color="tab:orange", s=3)
    
            # Plot the lightcurve with trend overplotted and orange transits
            # highlighted orange
            axs[0][i].scatter(lc.bjd, lc.fnorm, color="tab:blue", s=0.1, alpha=0.7)
            
            big_gaps = (lc.bjd[1:] - lc.bjd[:-1]) > 0.1
            big_gaps = np.concatenate((big_gaps, np.array([False])))
            
            nantrend = np.copy(lc.trend)
            nantrend[big_gaps] = np.ones(sum(big_gaps))*np.nan
            
            axs[0][i].plot(lc.bjd, nantrend+1, color='k', lw=1)
    
            mask = mask.astype(int)
            transit_edges = abs(mask[1:] - mask[:-1]) > 1e-10
            transit_edges = np.concatenate((transit_edges, np.array([False])))
            # Make sure the number of edges is even
            if sum(transit_edges) % 2 != 0:
                continue
            transit_bjds = (lc.bjd[transit_edges]).reshape(sum(transit_edges)//2, 2)
    
            
            for bjds in transit_bjds:
                axs[0][i].axvspan(bjds[0], bjds[1], color="tab:orange", alpha=0.5, 
                                  zorder=-1)
    
            axs[-1][i].set_xlim(min(lc.bjd), max(lc.bjd))
            axs[-1][i].set_xlabel("BJD")
    
        axs[0][0].set_ylabel("Normalized Flux")
        axs[1][0].set_ylabel("Detrended Flux")
        
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
        
    def model_plot(self, savefig=None, show=True, title=None, depthnorm=None):
        if self.mcmc_chain is None:
                print("run_mcmc method must be run first!")
                return None
        
        mc.plot_model(self, self.ts, savefig=savefig, show=show, title=title,
                     depthnorm=depthnorm)
    
    def corner_plot(self, savefig=None, show=True, title=None):
        if self.mcmc_chain is None:
            print("run_mcmc method must be run first!")
            return None
        
        mc.plot_chain_corner(self.mcmc_chain, 
                             savefig=savefig, show=show, title=title)
        
    def make_all_plots(self, savedir):
        """Save all relevant plots in a directory
        """
        ptag = round(self.period, 2)
        self.chain_evos_plot(f'{savedir}/{ptag:0.2f}_chain_evo_{self.ts.tic}.pdf', False)
        self.chain_dists_plot(f'{savedir}/{ptag:0.2f}_dists_{self.ts.tic}.pdf', False)
        self.model_plot(f'{savedir}/{ptag:0.2f}_model_{self.ts.tic}.pdf', 
                        False, depthnorm=2.2)
        self.corner_plot(f'{savedir}/{ptag:0.2f}_corner_{self.ts.tic}.pdf', False)
        plt.close('all')        
    
    
### Loading back objects

def load_ts(path):      
    with open(path, "rb") as f:
        loaded_ts = pickle.load(f)
    return loaded_ts

def load_ts_update(path):
    updated_ts = TransitSearchUpdate(load_ts(path))
    return updated_ts
    

# Class to update with new methods
class TransitSearchUpdate(TransitSearch):
    """Class to update a saved TransitSearch object with new methods while
    retaining all attributes. Will also update the planet candidates and 
    lightcurves
    """
    def __init__(self, ts):
        # Space for planet Candidates
        self.lightcurves = []
        self.planet_candidates = []
        self.planet_candidates_reject = []
        self.planet_candidates_plausible = []
        
        # Set Aliases
        self.lcs, self.pcs, self.pcs_r, self.pcs_p =\
        self.lightcurves, self.planet_candidates,\
        self.planet_candidates_reject, self.planet_candidates_plausible
        
        # Go through all the attributes and update them
        for attr_name in dir(ts):
            attr = getattr(ts, attr_name)
            
            # Check if it's a method or builtin
            if ismethod(attr) or attr_name[:2]=="__":
                continue
           
            # Planet Candidates and lightcurves
            elif attr_name in ['planet_candidates','planet_candidates_reject']:
                for pc in attr:
                    getattr(self, attr_name)\
                    .append(PlanetCandidateUpdate(pc, self))
                    
            elif attr_name == 'lightcurves':
                for lc in attr:
                    self.lightcurves.append(LightCurveUpdate(lc))
                    
            else:
                setattr(self, attr_name, attr)
        
        self.version = VERSION
                
                
class LightCurveUpdate(LightCurve):
    """Class to update a saved Lightcurve object with new methods while
    retaining all attributes.
    """
    def __init__(self, lc):
        # Get all the attributes
        for attr_name in dir(lc):
            attr = getattr(lc, attr_name)
            
            # Check if it's a method or builtin
            if ismethod(attr) or attr_name[:2]=="__":
                continue
                
            else:
                setattr(self, attr_name, attr)

                
class PlanetCandidateUpdate(PlanetCandidate):
    """Class to update a saved PlanetCandidate object with new methods while
    retaining all attributes.
    """
    def __init__(self, pc, ts):
        # Get all the attributes
        for attr_name in dir(pc):
            attr = getattr(pc, attr_name)
            
            # Check if it's a method or builtin
            if ismethod(attr) or attr_name[:2]=="__":
                continue
            
            elif attr_name=='ts':
                setattr(self, 'ts', ts)
                
            else:
                setattr(self, attr_name, attr)


                
""" Injection Recovery Modules """
class InjecrecTS(TransitSearch):

    def __init__(self, tic):
        # Inherited Attributes
        TransitSearch.__init__(self, tic, detrend=False)
        # List of (T0, P, Rp, b)
        self.injected = np.zeros(5)*np.nan
        # recovery dictionary
        self.recovery_dict = {}

    
    def mask_planet(self, T0, P, duration):
        """Mask the planet signals in the lightcurves by sampling noise 
        characterized by the data before and after the transit
        """
        np.random.seed(42)
        for lc in self.lightcurves:
            intransit = transit_mask(lc.bjd, P, duration*1.1, T0)
            transit_edges = np.convolve(intransit, np.ones(40), mode='same')>1
            transit_edges = np.logical_xor(intransit, transit_edges)
    
            # Or just go through the lightcurve
            tphase = T0 % P
            bjd_start, bjd_end = lc.bjd[0]-tphase-P/2, lc.bjd[0]+P/2-tphase

            while bjd_start < lc.bjd[-1]:
                bjd_cut = (lc.bjd > bjd_start) & (lc.bjd < bjd_end)
                
                # Should have one transit, mask it
                transit_cut = bjd_cut & intransit
                if np.any(transit_cut):
                    edge_cut = bjd_cut & transit_edges
                    mean, std = np.median(lc.fnorm[edge_cut]),\
                                np.std(lc.fnorm[edge_cut])
                    
                    lc.fnorm[transit_cut] = np.random.normal(loc=mean, scale=std,
                                            size=sum(bjd_cut & intransit))
                
                # Increment by period length
                bjd_start += P
                bjd_end += P

            lc.detrended = False
            lc.trend, lc.fnorm_detrend, lc.detrend_methods = None, None, []
        
        return None

    
    def mask_TOIs(self):
        """Retrieve TOI information from exofop and mask data corresponding to
        each TOI parameter
        """
        retries = 0
        while retries < 5:
            try:
                exofop_tic = TIC(self.tic)
                tab = exofop_tic.lookup()
                break
            except TimeoutError:
                retries += 1
                
        T0s, Ps, durs = tab['Transit Epoch (BJD)'].to_numpy(dtype=float),\
                        tab['Period (days)'].to_numpy(dtype=float),\
                        tab['Duration (hours)'].to_numpy(dtype=float)/24  

        for i in range(len(tab)):
            self.mask_planet(T0s[i], Ps[i], durs[i])

        return None
        
    
    def mask_pcs(self, plausible=True, reject=True):
        """Mask the signals associated with the planet candidates in this 
        transit search
        """
        pcs = self.pcs
        if plausible:
            pcs += self.pcs_p
        if reject:
            pcs += self.pcs_r

        for pc in pcs:
            T0, P, duration = pc.T0, pc.period, pc.duration
            self.mask_planet(T0, P, duration)

    
    def inject_planet(self, T0, P, Rp, b, detrend=True):
        """Inject a planet signal into the lightcurve
        """
        for lc in self.lightcurves:
            model = misc.batman_model(lc.bjd, T0, P, Rp, b,
                                      self.radius, self.mass, self.u)
            lc.fnorm += model - 1

            if detrend:
                lc.detrend_lc()

        # Set injection parameters
        self.injected = (T0, P, Rp, b, 0)
        
        return None
    
    
    def check_injection(self, tolerance=0.02, plausible=False):
        """Check the injected planet parameters, add data to the recovery dict
        """
        found = False
        found_params = np.zeros(5)*np.nan
        
        pcs = self.pcs
        if plausible:
            pcs += self.pcs_p

        for pc in pcs:
            period_ratio = max(pc.period / self.injected[1], 
                               self.injected[1] / pc.period)
            if math.isclose(period_ratio, round(period_ratio), rel_tol=tolerance):
                found = True
                found_params = np.array([pc.T0, pc.period, pc.Rp, pc.b, pc.offset])
                
        self.recovery_dict[self.injected] = [found, found_params]

        if found:
            phase_diff = min((found_params[0]-self.injected[0]) % self.injected[1],
                             (self.injected[0]-found_params[0]) % self.injected[1])
            param_diffs = found_params[:-1] / np.array(self.injected[:-1])
            param_diffs[0] = phase_diff
        else:
            param_diffs = np.nan*np.ones(4)

        self.recovery_dict[self.injected].append(param_diffs)
        self.recovery_dict[self.injected].append(len(pcs) - int(found))
        
        return None

    
    def reset_injection(self):
        """Reset the injected planet parameters
        """
        if np.any(np.isnan(np.array(self.injected))):
            print("NaN in transit params, cannot reset")
            return None
        
        T0, P, Rp, b = self.injected[:-1]
        
        for lc in self.lightcurves:
            model = misc.batman_model(lc.bjd, T0, P, Rp, b,
                                      self.radius, self.mass, self.u)
            lc.fnorm -= model - 1

        # Set injection parameters
        self.injected = np.zeros(5)*np.nan


    def restore_lc_data(self, retain_recovery=True):
        """Restore lightcurve data that may have been deleted
        """
        recovery_dict = self.recovery_dict.copy()

        self.__init__(self.tic)
        self.recovery_dict = recovery_dict


    def save(self, filename, remove_data=False):
        """Save the injection recovery, optionally delete lightcurve data
        """
        if remove_data:
            self.lightcurves = []
    
        with open(filename+'.ts', "wb") as f:
            pickle.dump(self, f)
            
        return None


def load_injecrec(path, reload_data=False):
    """Reload a saved injection recovery
    """
    with open(path, "rb") as f:
        loaded_injecrec = pickle.load(f)

    if reload_data:
        loaded_injecrec.restore_lc_data(retain_recovery=True)
    
    return loaded_injecrec


class InjecrecTSUpdate(TransitSearchUpdate, InjecrecTS):
    """Class to inherit different methods
    """
    def __init__(self, ts):
        # Inherited Attributes
        TransitSearchUpdate.__init__(self, ts)

        # List of (T0, P, Rp, b)
        self.injected = np.zeros(5)*np.nan
        # recovery dictionary
        self.recovery_dict = {}

                