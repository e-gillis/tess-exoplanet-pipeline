import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from get_tess_data import *
from gaussian_detrending import * 
from planet_search import *
from misc_functions import *


class TransitSearch:
    """
    D O C S T R I N G
    """
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
            period_max = min((lc.bjd[-1] - lc.bjd[0])/2, 28)
            results_list = find_transits(lc.bjd, lc.fnorm_detrend, 
                                         period_min=1, period_max=period_max,
                                         show_progress_bar=True, 
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
    
    
    def gaussian_detrend_lc(self, cont=False):
        
        bjd_splits, efnorm_splits = self.get_splits([self.bjd, self.efnorm])
        if cont:
            fnorm_splits = self.get_splits([self.fnorm_detrend])[0]
        else:
            fnorm_splits = self.get_splits([self.fnorm])[0]
        
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
                fnorm_detrend -= map_soln["pred"]/1000
                
                residual_rotation, Prot = rotation_check(bjd, fnorm_detrend, 
                                                         efnorm)
                count += 1
            
            detrended = (count!=0) and (not residual_rotation) and (detrended)
            full_fnorm_detrend[index:index+len(fnorm_detrend)] = fnorm_detrend
            index += len(fnorm_detrend)
            
        self.fnorm_detrend = full_fnorm_detrend
        self.gauss_detrendeded = False
        self.detrended = detrended
        
        return self.detrended
    
    
    def median_detrend_lc(self, window_length=400, cont=False):
        # Find indeces corresponding to different sectors
        if cont:
            fnorm_splits = self.get_splits([self.fnorm_detrend])[0]
        else:
            fnorm_splits = self.get_splits([self.fnorm])[0]

        # Median 
        full_fnorm_detrend = np.zeros(len(self.fnorm))
        index = 0
        k = window_length // 2

        for i in range(len(fnorm_splits)):
            fnorm = fnorm_splits[i]
            
            print(len(fnorm))
            if window_length % 2 == 0: window_length += 1
            
            sav_model = savgol_filter(fnorm, window_length, 1) - 1
            fnorm_detrend = fnorm - sav_model

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
        
        